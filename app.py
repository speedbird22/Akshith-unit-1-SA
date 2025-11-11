import streamlit as st
from PIL import Image
import torch
import os
import sys
import subprocess

# FORCE HEADLESS OPENCV BEFORE ANYTHING
subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python-headless", "--force-reinstall", "--no-cache-dir"])
import cv2  # now safe

# Clone YOLOv5 if not there
if not os.path.exists("yolov5"):
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
    subprocess.run(["pip", "install", "-r", "yolov5/requirements.txt"])

sys.path.append("yolov5")

# ------------------- Indian Bin Colors -------------------
bin_colors = {
    'clothes': 'Yellow', 'paper': 'Blue', 'glass': 'Blue', 'battery': 'Red',
    'plastic': 'Blue', 'shoes': 'Yellow', 'trash': 'Black', 'cardboard': 'Blue',
    'biological': 'Green', 'metal': 'Blue'
}

bin_descriptions = {
    'Green': 'Wet Waste', 'Blue': 'Dry Recyclables', 'Yellow': 'Clothes/Shoes',
    'Red': 'Hazardous (Batteries)', 'Black': 'Non-recyclable'
}

st.set_page_config(page_title="Trash India", layout="centered")
st.title("🗑️ Indian Trash Classifier")
st.markdown("### Upload trash → Get correct bin color")

# Debug
with st.sidebar:
    st.write("Files:", os.listdir("."))

# ------------------- Load Model -------------------
@st.cache_resource
def load_model():
    if not os.path.exists("best.pt"):
        st.error("best.pt not found! Upload via Git LFS.")
        return None
    try:
        from models.common import DetectMultiBackend
        from utils.general import check_img_size
        from utils.torch_utils import select_device
        
        device = select_device('')
        model = DetectMultiBackend("best.pt", device=device)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(640, s=stride)
        model.warmup(imgsz=(1, 3, imgsz, imgsz))
        return model, names, stride, imgsz
    except Exception as e:
        st.error(f"Error: {e}")
        return None

model_data = load_model()
if model_data:
    model, names, stride, imgsz = model_data
    st.success("✅ Model loaded (NO libGL!)")

# ------------------- Predict -------------------
uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file and model_data:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded", use_column_width=True)

    with st.spinner("Detecting..."):
        import numpy as np
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression, scale_boxes

        img = np.array(image)
        img_resized = letterbox(img, imgsz, stride=stride)[0]
        img_resized = img_resized.transpose((2, 0, 1))[::-1]
        img_resized = np.ascontiguousarray(img_resized)
        img_tensor = torch.from_numpy(img_resized).to(model.device).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.45)[0]

        if len(pred):
            pred[:, :4] = scale_boxes(img_resized.shape[1:], pred[:, :4], img.shape).round()
            for *xyxy, conf, cls in pred:
                label = f"{names[int(cls)]} {conf:.2f}"
                cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        st.image(img, caption="Result", use_column_width=True)

        if len(pred):
            top_cls = names[int(pred[0][5])]
            top_conf = pred[0][4].item()
            color = bin_colors.get(top_cls, "Unknown")
            st.success(f"**{top_cls.capitalize()}** ({top_conf:.1%}) → **{color} Bin**")
            st.markdown(f"_{bin_descriptions[color]}_")
            st.balloons()
        else:
            st.warning("No trash detected.")

st.markdown("---")
st.markdown("### Indian Bins\n- Green → Wet\n- Blue → Dry\n- Yellow → Clothes\n- Red → Batteries\n- Black → Trash")
