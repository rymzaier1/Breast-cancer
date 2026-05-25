import streamlit as st
import requests

from PIL import Image

st.set_page_config(
    page_title="Breast Cancer "
)

st.title("Breast Cancer Detection ")

uploaded_file = st.file_uploader(
    "Upload Mammography",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Mammography"
    )

    if st.button("Analyse"):

        with st.spinner("AI analyzing image..."):

            files = {
                "file": uploaded_file.getvalue()
            }

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                files=files
            )

            result = response.json()

            st.success(
                f"Prediction: {result['prediction']}"
            )

            st.info(
                f"Confidence: {result['confidence']}%"
            )

            st.image(
                result["gradcam"],
                caption="Grad-CAM Heatmap"
            )