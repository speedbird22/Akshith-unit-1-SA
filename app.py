import streamlit as st
from PIL import Image
import torch
import os

# CRITICAL FIX: Disable GUI backends BEFORE anything else
os.environ["OPENCV_SHOTSHOW_WINDOW"] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# ------------------- Indian Bin Colors -------------------
bin_colors = {
    'clothes': 'Yellow', 'paper': 'Blue', 'glass': 'Blue', 'battery': 'Red',
    'plastic': 'Blue', 'shoes': 'Yellow', 'trash': 'Black', 'cardboard': 'Blue',
    'biological': 'Green', 'metal': 'Blue'
}

bin_descriptions = {
    'Green': 'Wet Waste (Food, peels)',
    'Blue': 'Dry Recyclables (Paper, plastic, metal, glass, cardboard)',
    'Yellow': 'Clothes / Shoes',
    'Red': 'Hazardous (Batteries)',
    'Black': 'Non-recyclable'
}

# ------------------- UI -------------------
st.set_page_config(page_title="Trash Bin India", layout="centered")
st.title("Indian Trash Classifier")
st.markdown("### Upload trash → Know which **colored bin** to use in India!")

# Debug sidebar
with st.sidebar:
    st.write("Files in folder:")
    st.write(os.listdir("."))

# ------------------- Load Model -------------------
@st.cache_resource
def load_model():
    if not os.path.exists("best.pt"):
        st.error("best.pt not found! Use Git LFS to upload it.")
        return None
    try:
        # This line forces headless mode
        torch.hub.set_dir("/tmp")  # Avoids permission issues
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)
        model.conf = 0.40
        model.iou = 0.45
        return model
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None

model = load_model()
if model:
    st.success("Model loaded successfully!")

# ------------------- Prediction -------------------
uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting trash..."):
        results = model(image, size=640)
        results.render()  # Draws boxes on results.ims[0]

        annotated = results.ims[0]
        st.image(annotated, caption="Detected Trash", use_column_width=True)

        preds = results.pandas().xyxy[0]
        if len(preds) == 0:
            st.warning("No trash detected. Try a clearer photo.")
        else:
            top = preds.loc[preds['confidence'].idxmax()]
            cls = top['name']
            conf = top['confidence']
            bin_color = bin_colors.get(cls, "Unknown")

            st.success(f"**{cls.capitalize()}** ({conf:.1%} confidence)")
            st.markdown(f"### Throw in **{bin_color} Bin**")
            st.markdown(f"_{bin_descriptions[bin_color]}_")
            st.balloons()

# ------------------- Guide -------------------
st.markdown("---")
st.markdown("""
### Indian Bin Colors (Swachh Bharat Mission)
- **Green** → Wet waste  
- **Blue** → Dry recyclables  
- **Yellow** → Clothes, shoes  
- **Red** → Batteries, chemicals  
- **Black** → Non-recyclable trash
""")

st.info("Ensure `best.pt` is uploaded using **Git LFS** for large files.")
