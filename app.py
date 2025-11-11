import streamlit as st
from PIL import Image, ImageDraw
import torch
import os
import sys

# Add YOLOv5 repo to path
sys.path.insert(0, os.path.abspath("yolov5"))
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.torch_utils import select_device

# ------------------- Indian Bin Colors -------------------
bin_colors = {
    'clothes': 'Yellow', 'paper': 'Blue', 'glass': 'Blue', 'battery': 'Red',
    'plastic': 'Blue', 'shoes': 'Yellow', 'trash': 'Black', 'cardboard': 'Blue',
    'biological': 'Green', 'metal': 'Blue'
}

bin_descriptions = {
    'Green': 'Wet Waste (food, peels)',
    'Blue': 'Dry Recyclables (paper, plastic, metal, glass)',
    'Yellow': 'Clothes / Shoes',
    'Red': 'Hazardous (batteries)',
    'Black': 'Non-recyclable'
}

st.set_page_config(page_title="Trash India", layout="centered")
st.title("🗑️ Indian Trash Classifier")
st.markdown("### Upload trash → Get Swachh Bharat bin color")

with st.sidebar:
    st.write("Files:", os.listdir("."))

# ------------------- Load YOLOv5 Model -------------------
@st.cache_resource
def load_model():
    if not os.path.exists("best.pt"):
        st.error("best.pt not found! Upload via Git LFS.")
        return None
    try:
        device = select_device('cpu')
        model = attempt_load("best.pt", map_location=device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
if model:
    st.success("✅ YOLOv5 Model loaded successfully!")

# ------------------- Predict -------------------
uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded", use_column_width=True)

    with st.spinner("Detecting..."):
        # Preprocess
        img = letterbox(image, new_shape=640)[0]
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = torch.from_numpy(img).float() / 255.0
        img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.45)[0]

        draw = ImageDraw.Draw(image)
        if pred is None or len(pred) == 0:
            st.warning("No trash detected.")
        else:
            names = model.names
            for *box, conf, cls in pred:
                box = [int(x.item()) for x in box]
                label = names[int(cls)]
                draw.rectangle(box, outline="red", width=2)
                draw.text((box[0], box[1] - 10), f"{label} {conf:.1%}", fill="red")

            st.image(image, caption="Detected Trash", use_column_width=True)

            top = pred[torch.argmax(pred[:, 4])]
            cls_id = int(top[5])
            conf = float(top[4])
            cls = names[cls_id]
            color = bin_colors.get(cls, "Unknown")

            st.success(f"**{cls.capitalize()}** ({conf:.1%}) → **{color} Bin**")
            st.markdown(f"_{bin_descriptions[color]}_")
            st.balloons()

st.markdown("---")
st.markdown("""
### Swachh Bharat Bin Colors
- **Green** → Wet waste  
- **Blue** → Dry recyclables  
- **Yellow** → Clothes, shoes  
- **Red** → Batteries  
- **Black** → Non-recyclable
""")
st.info("This version uses YOLOv5 directly and avoids compatibility issues with YOLOv8.")
