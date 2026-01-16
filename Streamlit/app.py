import io
from datetime import datetime

import streamlit as st
from PIL import Image

from inference_new import run_inference_on_pil  # backend function
# from inference import run_inference_on_pil


def build_log_text(anomalies):
    """Plain-text/CSV-style log."""
    ts = datetime.now().isoformat()
    lines = [f"PCB Defect Detection Log - {ts}", ""]
    if not anomalies:
        lines.append("No anomalies detected.")
        return "\n".join(lines)

    lines.append("label,confidence,x1,y1,x2,y2")
    for d in anomalies:
        lines.append(
            f"{d['label']},{d['confidence']:.4f},"
            f"{d['box'][0]},{d['box'][1]},{d['box'][2]},{d['box'][3]}"
        )
    return "\n".join(lines)


def main():
    st.set_page_config(page_title="PCB Defect Detection", layout="wide")
    st.title("PCB Differential Defect Detection")

    uploaded_file = st.file_uploader(
        "Upload PCB image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input image")
            st.image(input_image, use_container_width=True)

        if st.button("Run detection"):
            with st.spinner("Running inference..."):
                result_image, anomalies = run_inference_on_pil(input_image)

            with col2:
                st.subheader("Result")
                st.image(result_image, use_container_width=True)

            if anomalies:
                st.subheader("Detected defects")
                st.dataframe([
                    {
                        "label": d["label"],
                        "confidence": round(d["confidence"], 3),
                        "x1": d["box"][0],
                        "y1": d["box"][1],
                        "x2": d["box"][2],
                        "y2": d["box"][3],
                    }
                    for d in anomalies
                ])

            # Download section
            st.subheader("Download outputs")

            # Image as PNG bytes
            img_buf = io.BytesIO()
            result_image.save(img_buf, format="PNG")
            img_bytes = img_buf.getvalue()

            st.download_button(
                label="Download result image (PNG)",
                data=img_bytes,
                file_name="pcb_result.png",
                mime="image/png",
                key="download-image",
            )

            log_text = build_log_text(anomalies)
            st.download_button(
                label="Download detection log (TXT)",
                data=log_text,
                file_name="pcb_detection_log.txt",
                mime="text/plain",
                key="download-log",
            )


if __name__ == "__main__":
    main()
