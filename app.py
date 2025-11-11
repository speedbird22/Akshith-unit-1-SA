import streamlit as st
from PIL import Image, ImageDraw
import torch
import numpy as np
import os

# ------------------- Bin Colors -------------------
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

# ------------------- Load YOLOv5 Model -------------------
@st.cache_resource
def load_model():
    try:
        model = torch.load("best.pt", map_location=torch.device("cpu"))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
if model:
    st.success("✅ YOLOv5 model loaded successfully!")

# ------------------- Predict -------------------
uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded", use_column_width=True)

    with st.spinner("Detecting..."):
        # Preprocess
        img = np.array(image)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0)

        # Inference
        results = model(img)[0]

        # Postprocess
        boxes = results.boxes
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
