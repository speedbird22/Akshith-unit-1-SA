import streamlit as st
from PIL import Image
import torch
import numpy as np
import os

# ------------------- Indian Bin Colors -------------------
bin_colors = {
    'clothes': 'Yellow', 'paper': 'Blue', 'glass': 'Blue', 'battery': 'Red',
    'plastic': 'Blue', 'shoes': 'Yellow', 'trash': 'Black', 'cardboard': 'Blue',
    'biological': 'Green', 'metal': 'Blue'
}

bin_descriptions = {
    'Green': 'Wet Waste (Food, peels, etc.)',
    'Blue': 'Dry Recyclables (Paper, plastic, metal, glass, cardboard)',
    'Yellow': 'Clothes / Shoes / Reusables',
    'Red': 'Hazardous (Batteries, chemicals)',
    'Black': 'Non-recyclable'
}

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Trash Bin India", layout="centered")
st.title("Indian Trash Classifier")
st.markdown("### Upload trash → Know which **colored bin** to use!")

# Debug: Show files
with st.sidebar:
    st.write("Files in folder:")
    st.write(os.listdir("."))

# ------------------- Load Model -------------------
@st.cache_resource
def load_model():
    if not os.path.exists("best.pt"):
        st.error("best.pt not found! Upload it via Git LFS.")
        return None
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)
        model.conf = 0.4
        model.iou = 0.45
        return model
    except Exception as e:
        st.error(f"Model error: {e}")
        return None

model = load_model()
if model:
    st.success("Model loaded!")

# ------------------- Upload & Predict -------------------
uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded", use_column_width=True)

    with st.spinner("Detecting..."):
        # Direct PIL → YOLOv5 (no cv2 needed!)
        results = model(image, size=640)

        # Render results on image
        results.render()  # modifies results.ims in-place
        annotated_image = results.ims[0]  # PIL Image with boxes
        st.image(annotated_image, caption="Detected", use_column_width=True)

        # Get predictions
        preds = results.pandas().xyxy[0]
        if len(preds) == 0:
            st.warning("No trash detected.")
        else:
            # Top prediction
            top = preds.loc[preds['confidence'].idxmax()]
            cls = top['name']
            conf = top['confidence']
            bin_color = bin_colors[cls]

            st.success(f"**{cls.capitalize()}** ({conf:.1%}) → **{bin_color} Bin**")
            st.markdown(f"_{bin_descriptions[bin_color]}_")
            st.balloons()

# ------------------- Guide -------------------
st.markdown("---")
st.markdown("""
### India Bin Colors (Swachh Bharat)
- **Green** → Wet waste  
- **Blue** → Dry recyclables  
- **Yellow** → Clothes, shoes  
- **Red** → Batteries, e-waste  
- **Black** → Non-recyclable
""")
st.info("Make sure `best.pt` is uploaded via Git LFS (large files).")
