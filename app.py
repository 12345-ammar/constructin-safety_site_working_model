import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile

st.set_page_config(page_title="Construction Site Safety Monitoring", layout="wide")

st.title("Construction Site Safety Monitoring System")

# Model load karein
@st.cache_resource
def load_model():
    return YOLO("construction_site_safely_model.pt")

model = load_model()

option = st.selectbox("Input Mode Chunein", ["Image", "Video", "Live Streaming (Webcam)"])

if option == "Image":
    uploaded_file = st.file_uploader("Ek image upload karein", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        results = model(img_array)
        res_img = results[0].plot()

        boxes = results[0].boxes
        total_count = len(boxes)
        
        # Classes extract karein jo model ne detect ki hain
        detected_classes = [model.names[int(cls)] for cls in boxes.cls]
        
        col1, col2 = st.columns(2)
        col1.metric("Total Detections", total_count)
        col2.metric("Detected Items", ", ".join(set(detected_classes)) if detected_classes else "Kuch nahi mila")

        st.image(res_img, channels="BGR", use_column_width=True)
        
        # Custom Alert message
        st.warning(f"Detected items: {len(detected_classes)} cheezein mil chuki hain. Red boxes unhe show kar rahe hain jo missing ya violate kar rahe hain.")

elif option == "Video":
    video_file = st.file_uploader("Ek video upload karein", type=["mp4", "avi"])
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        
        col1, col2 = st.columns(2)
        metric_total = col1.empty()
        metric_alert = col2.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model(frame)
            annotated_frame = results[0].plot()
            
            boxes = results[0].boxes
            total_count = len(boxes)
            detected_classes = [model.names[int(cls)] for cls in boxes.cls]
            
            st_frame.image(annotated_frame, channels="BGR", use_column_width=True)
            metric_total.metric("Total Detections", total_count)
            metric_alert.metric("Status / Items", f"{len(detected_classes)} detected")
            
        cap.release()

elif option == "Live Streaming (Webcam)":
    run = st.checkbox("Live Webcam Start Karein")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.warning("Webcam access nahi mil saka.")
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        FRAME_WINDOW.image(annotated_frame, channels="BGR")
    else:
        camera.release()