import streamlit as st
import pandas as pd
from PIL import Image
from ultralytics import YOLO

from reporting import summarize_detections, total_detections


st.set_page_config(
    page_title="LabLens AI",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 LabLens AI")
st.subheader("Workspace Object Audit")

st.write(
    "Upload a workspace image to identify recognizable objects, "
    "review their locations, and generate an inventory report."
)

st.info(
    "This is an AI prototype. Its detections do not certify "
    "that a workspace is safe."
)


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


uploaded_file = st.file_uploader(
    "Upload a workspace image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Minimum confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.35,
    step=0.05
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("### Original image")
    st.image(image, use_container_width=True)

    if st.button("Analyze workspace", type="primary"):

        with st.spinner("Analyzing image with YOLOv8..."):

            model = load_model()

            results = model.predict(
                source=image,
                conf=confidence,
                verbose=False
            )

            result = results[0]

            annotated_image = result.plot()[:, :, ::-1].copy()

            detection_rows = []

            for box in result.boxes:

                class_id = int(box.cls[0].item())
                score = float(box.conf[0].item())

                detection_rows.append({
                    "Object": result.names[class_id],
                    "Confidence": round(score, 4),
                    "Confidence (%)": round(score * 100, 2)
                })

        st.markdown("### Detection results")

        left, right = st.columns(2)

        with left:
            st.image(
                annotated_image,
                caption="YOLOv8 annotated image",
                use_container_width=True
            )

        with right:
            st.markdown("#### Object inventory")

            summary = pd.DataFrame(
                summarize_detections(detection_rows),
                columns=["Object", "Count"]
            )

            st.dataframe(summary, use_container_width=True)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Objects detected",
            total_detections(detection_rows)
        )

        col2.metric(
            "Object categories",
            len(summary)
        )

        col3.metric(
            "Confidence threshold",
            f"{confidence:.0%}"
        )

        st.markdown("#### Individual detections")

        details = pd.DataFrame(
            detection_rows,
            columns=["Object", "Confidence", "Confidence (%)"]
        )

        st.dataframe(details, use_container_width=True)

        st.download_button(
            label="Download CSV report",
            data=details.to_csv(index=False).encode("utf-8"),
            file_name="lablens_detection_report.csv",
            mime="text/csv"
        )

        if not detection_rows:
            st.warning(
                "No objects were detected above this confidence "
                "threshold. Try another image or a lower threshold."
            )

        st.caption(
            "The pretrained model may miss objects or classify "
            "them incorrectly."
        )

else:
    st.warning("Upload an image to begin your workspace audit.")
