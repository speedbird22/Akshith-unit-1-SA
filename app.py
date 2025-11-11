import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO
import os

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

# ------------------- Load YOLOv5-A Model -------------------
@st.cache_resource
def load_model():
    if not os.path.exists("best.pt"):
        st.error("best.pt not found! Upload via Git LFS.")
        return None
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
if model:
    st.success("✅ YOLOv5-A Model loaded successfully!")

# ------------------- Predict -------------------
uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded", use_column_width=True)

    with st.spinner("Detecting..."):
        results = model.predict(image, imgsz=640, conf=0.4, iou=0.45)
        boxes = results[0].boxes
        names = model.names

        if boxes is None or len(boxes) == 0:
            st.warning("No trash detected.")
        else:
            draw = ImageDraw.Draw(image)
            for box in boxes:
                b = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = names[cls_id]
                draw.rectangle(b, outline="red", width=2)
                draw.text((b[0], b[1] - 10), f"{label} {conf:.1%}", fill="red")

            st.image(image, caption="Detected Trash", use_column_width=True)

            top = max(boxes, key=lambda b: b.conf[0])
            cls_id = int(top.cls[0])
            conf = float(top.conf[0])
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
st.info("This version uses `ultralytics` and avoids OpenGL dependencies like `libGL.so.1`.")
