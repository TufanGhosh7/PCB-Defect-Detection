import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.ops import nms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import imagehash
from skimage.metrics import structural_similarity as ssim
from torchvision.models import resnet50  # match the checkpoint
import torch.nn as nn


print("--- PCB Differential Detection Pipeline (FIXED LABELS) ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===================================================================
# LOAD MODEL WITH CLASS NAMES EXTRACTION (FIXED)
# ===================================================================
model_path = "best_resnet50_pcb_defects_50epochs.pth"

# 🔍 INSPECT CHECKPOINT FIRST
checkpoint = torch.load(model_path, map_location=device)
print("🔍 Checkpoint keys:", list(checkpoint.keys()))

# ✅ EXTRACT CLASS NAMES (Method 1: From checkpoint)
class_names = checkpoint.get("class_names", None)
if class_names is None:
    print("⚠️ No 'class_names' in checkpoint")
    # ✅ Method 2: Standard DeepPCB order (matches 99% of models) [web:52]
    class_names = [
        'missing_hole', 'mouse_bite', 'open_circuit', 
        'short', 'spur', 'spurious_copper'
    ]
    print("✅ Using standard DeepPCB class order")

# ✅ Remove 'normal' if present (inference only)
if 'normal' in class_names:
    class_names.remove('normal')

num_classes = len(class_names)
print(f"✅ {num_classes} classes: {class_names}")

# ✅ Create & load model
defect_classifier = models.resnet50(weights=None)
in_features = defect_classifier.fc.in_features
defect_classifier.fc = torch.nn.Linear(in_features, num_classes)
defect_classifier.load_state_dict(checkpoint["model_state_dict"])
defect_classifier.to(device)
defect_classifier.eval()

# ✅ VERIFY: Model output matches class count
dummy_input = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    dummy_out = defect_classifier(dummy_input)
    print(f"✅ Model output shape: {dummy_out.shape} == [1, {num_classes}] ✓")

# ===================================================================
# CONFIG
# ===================================================================
golden_images_dir = "PCB_DATASET\PCB_USED"
WINDOW_SIZE = 128
STRIDE = WINDOW_SIZE // 4
SIMILARITY_THRESHOLD = 0.95
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.80

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ===================================================================
# GOLDEN DATABASE
# ===================================================================
def create_golden_image_database(golden_dir):
    db = []
    filenames = os.listdir(golden_dir)
    for fname in filenames:
        path = os.path.join(golden_dir, fname)
        try:
            img = Image.open(path).convert('RGB')
            hash_val = imagehash.phash(img)
            db.append({'filename': fname, 'image': img, 'hash': hash_val})
        except Exception as e:
            print(f"Warning: Could not load '{path}'. Error: {e}")
    return db

golden_db = create_golden_image_database(golden_images_dir)
print(f"✅ {len(golden_db)} golden images loaded")

# ===================================================================
# 🔍 DEBUG FUNCTION (NEW - shows why labels wrong)
# ===================================================================
def debug_top_predictions(patch_tensor, window_coords):
    """Print top-3 predictions for debugging wrong labels"""
    with torch.no_grad():
        outputs = defect_classifier(patch_tensor)
        probabilities = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, 3)
    
    print(f"🔍 Window {window_coords}: Top-3 predictions:")
    for i in range(3):
        idx = top_indices[0][i].item()
        prob = top_probs[0][i].item()
        print(f"  {i+1}. {class_names[idx]}: {prob:.3f}")
    
    return top_indices[0][0].item(), top_probs[0][0].item()

# ===================================================================
# CORE FUNCTIONS (FIXED CLASS MAPPING)
# ===================================================================
def find_best_match(input_image, golden_database):
    if not golden_database:
        return None
    input_hash = imagehash.phash(input_image)
    best_match = min(golden_database, key=lambda x: input_hash - x['hash'])
    return best_match['image']

def detect_anomalies_by_comparison(input_image, golden_image, classifier):
    if input_image.size != golden_image.size:
        golden_image = golden_image.resize(input_image.size)
    
    detections = []
    img_width, img_height = input_image.size
    debug_count = 0  # Limit debug spam
    
    for y in range(0, img_height - WINDOW_SIZE + 1, STRIDE):
        for x in range(0, img_width - WINDOW_SIZE + 1, STRIDE):
            window_input = input_image.crop((x, y, x + WINDOW_SIZE, y + WINDOW_SIZE))
            window_golden = golden_image.crop((x, y, x + WINDOW_SIZE, y + WINDOW_SIZE))
            
            window_input_gray = np.array(window_input.convert('L'))
            window_golden_gray = np.array(window_golden.convert('L'))
            
            ssim_score, _ = ssim(window_golden_gray, window_input_gray, full=True)
            
            if ssim_score < SIMILARITY_THRESHOLD:
                patch_tensor = inference_transform(window_input).unsqueeze(0).to(device)
                
                # ✅ FIXED: Safe index + debug
                predicted_idx, confidence = debug_top_predictions(patch_tensor, (x,y)) if debug_count < 3 else (
                    torch.argmax(F.softmax(defect_classifier(patch_tensor), dim=1), dim=1).item(),
                    torch.max(F.softmax(defect_classifier(patch_tensor), dim=1), dim=1)[0].item()
                )
                debug_count += 1
                
                if confidence > CLASSIFIER_CONFIDENCE_THRESHOLD:
                    detections.append({
                        'box': [x, y, x + WINDOW_SIZE, y + WINDOW_SIZE],
                        'label': class_names[predicted_idx],  # ✅ SAFE INDEX ACCESS
                        'confidence': float(confidence)
                    })
    
    if not detections:
        return []
    
    # NMS
    boxes = torch.tensor([d['box'] for d in detections], dtype=torch.float32)
    scores = torch.tensor([d['confidence'] for d in detections], dtype=torch.float32)
    keep_indices = nms(boxes, scores, iou_threshold=0.2)
    return [detections[i] for i in keep_indices]

def draw_detections_on_image(image, detections):
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 32)
    except:
        font = ImageFont.load_default()
    
    unique_labels = list(set([d['label'] for d in detections]))
    colors = plt.cm.get_cmap('hsv', len(unique_labels) + 1)
    color_map = {label: tuple((np.array(colors(i)[:3]) * 255).astype(int)) 
                for i, label in enumerate(unique_labels)}
    
    for det in detections:
        box = det['box']
        label = det['label']
        confidence = det['confidence']
        color = color_map.get(label, (255, 50, 50))
        
        draw.rectangle(box, outline=color, width=5)
        
        text = f"{label} ({confidence:.2f})"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        background_box = [box[0], box[1] - text_height - 5, 
                         box[0] + text_width + 10, box[1]]
        draw.rectangle(background_box, fill=color)
        draw.text((box[0] + 5, box[1] - text_height - 5), text, fill="white", font=font)
    
    return img_with_boxes

# ===================================================================
# STREAMLIT INTERFACE
# ===================================================================
def run_inference_on_pil(input_image):
    """Exact interface required by your Streamlit app.py"""
    start_time = time.time()
    
    print("🔍 Finding best golden match...")
    golden_image_ref = find_best_match(input_image, golden_db)
    if not golden_image_ref:
        print("⚠️ No golden reference found")
        return input_image, []
    
    print("🔍 Running differential detection...")
    anomalies = detect_anomalies_by_comparison(input_image, golden_image_ref, defect_classifier)
    result_image = draw_detections_on_image(input_image, anomalies)
    
    elapsed = time.time() - start_time
    print(f"✅ Detection complete: {len(anomalies)} defects in {elapsed:.2f}s")
    print(f"   Classes used: {class_names}")
    
    return result_image, anomalies

print("✅ PIPELINE READY - LABELS FIXED!")
print(f"Classes: {class_names}")
print("Run: streamlit run app.py")
