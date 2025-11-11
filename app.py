import streamlit as st
from PIL import Image
import torch
import os

# FORCE HEADLESS MODE — THIS IS THE ONLY LINE THAT MATTERS
import os
os.environ["YOLO_VERBOSE"] = "False"          # Silence logs
os.environ["OPENCV_IO_ENABLE_OPENCL"] = "0"   # Disable OpenCL
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

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

st.set_page_config(page_title="Trash Classifier India", layout="centered")
st.title("Indian Trash Classifier")
st.markdown("### Upload trash → Get correct bin color (Swachh Bharat)")

# Debug
with st.sidebar:
    st.write("Files in folder:", os.listdir("."))

# ------------------- Load Model -------------------
@st.cache_resource
def load_model():
    if not os.path.exists("best.pt"):
        st.error("best.pt not found! Upload using Git LFS.")
        return None
    try:
        # This forces pure PyTorch backend — no OpenCV at all
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', 
                               force_reload=False, _verbose=False)
        model.conf = 0.40
        model.iou = 0.45
        return model
    except Exception as e:
        st.error(f"Model failed: {e}")
        return None

model = load_model()
if model:
    st.success("Model loaded successfully!")

# ------------------- Predict -------------------
uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded", use_column_width=True)

    with st.spinner("Detecting..."):
        # Direct PIL → YOLOv5 (no cv2, no libGL)
        results = model(image, size=640)
        results.render()  # Draws boxes
        st.image(results.ims[0], caption="Result", use_column_width=True)

        preds = results.pandas().xyxy[0]
        if len(preds) == 0:
            st.warning("No trash detected.")
        else:
            top = preds.loc[preds['confidence'].idxmax()]
            cls = top['name']
            conf = top['confidence']
            bin_color = bin_colors.get(cls, "Unknown")

            st.success(f"**{cls.capitalize()}** ({conf:.1%})")
            st.markdown(f"### Throw in **{bin_color} Bin**")
            st.markdown(f"_{bin_descriptions[bin_color]}_")
            st.balloons()

st.markdown("---")
st.markdown("""
### Indian Bin Colors
- **Green** → Wet waste  
- **Blue** → Dry recyclables  
- **Yellow** → Clothes, shoes  
- **Red** → Batteries  
- **Black** → Non-recyclable
""")
