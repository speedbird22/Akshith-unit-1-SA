import streamlit as st
from PIL import Image, ImageDraw
import torch
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

# Debug
with st.sidebar:
    st.write("Files:", os.listdir("."))

# ------------------- Load YOLOv5 Model -------------------
@st.cache_resource
def load_model():
    if not os.path.exists("best.pt"):
        st.error("best.pt not found! Upload via Git LFS.")
        return None
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)
        model.conf = 0.4
        model.iou = 0.45
        return model
    except Exception as e:
        st.error(f"Error: {e}")
        return None

model = load_model()
if model:
    st.success("✅ YOLOv5 Model loaded perfectly!")

# ------------------- Predict -------------------
uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded", use_column_width=True)

    with st.spinner("Detecting..."):
        results = model(image, size=640)
        preds = results.pandas().xyxy[0]

        if len(preds) == 0:
            st.warning("No trash detected.")
        else:
            # Draw bounding boxes manually using PIL
            draw = ImageDraw.Draw(image)
            for _, row in preds.iterrows():
                box = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
                label = row['name']
                draw.rectangle(box, outline="red", width=2)
                draw.text((box[0], box[1] - 10), label, fill="red")

            st.image(image, caption="Detected Trash", use_column_width=True)

            top = preds.loc[preds['confidence'].idxmax()]
            cls = top['name']
            conf = top['confidence']
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
st.info("This version avoids OpenGL dependencies like `libGL.so.1` by using PIL for rendering.")
