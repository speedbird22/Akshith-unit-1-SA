import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np

# ------------------- Indian Bin Colors -------------------
# As per Swachh Bharat Mission (India) - Standard Color Coding
bin_colors = {
    'clothes':      'Yellow',      # Reusable/Donatable - Yellow (for dry waste, often clothes go here)
    'paper':        'Blue',        # Dry waste - Blue bin
    'glass':        'Blue',        # Dry waste - Blue bin
    'battery':      'Red',         # Hazardous waste - Red bin (special collection)
    'plastic':      'Blue',        # Dry waste - Blue bin
    'shoes':        'Yellow',      # Reusable or dry waste - Yellow
    'trash':        'Black',       # Non-recyclable - Black bin
    'cardboard':    'Blue',        # Dry waste - Blue bin
    'biological':   'Green',       # Wet waste - Green bin
    'metal':        'Blue'         # Dry waste - Blue bin
}

bin_descriptions = {
    'Green':  'Wet Waste (Kitchen waste, food, etc.)',
    'Blue':   'Dry Waste (Paper, plastic, cardboard, metal, glass)',
    'Yellow': 'Reusable/Donatable (Clothes, shoes, toys)',
    'Red':    'Hazardous Waste (Batteries, chemicals, medicines)',
    'Black':  'Non-recyclable / Reject Waste'
}

# Class names (must match your training exactly)
class_names = ['clothes', 'paper', 'glass', 'battery', 'plastic', 
               'shoes', 'trash', 'cardboard', 'biological', 'metal']

# ------------------- Streamlit App -------------------
st.set_page_config(page_title="Trash Classifier - India", layout="centered")
st.title("🗑️ Indian Trash Classifier")
st.markdown("### Upload an image of trash and know which **colored bin** to use in India!")

# Load YOLOv5 model
@st.cache_resource
def load_model():
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)
        model.conf = 0.4  # confidence threshold
        model.iou = 0.45  # NMS IoU threshold
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Make sure 'best.pt' is in the same folder as this script.")
        return None

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload trash image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if model is None:
        st.error("Model not loaded. Cannot make predictions.")
    else:
        with st.spinner("Analyzing..."):
            # Convert PIL image to OpenCV format
            img_cv = np.array(image)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
            
            # Run inference
            results = model(img_cv)
            
            # Get predictions
            predictions = results.pandas().xyxy[0]  # pandas dataframe
            
            if len(predictions) == 0:
                st.warning("No trash detected. Try a clearer image.")
            else:
                # Show results on image
                results.render()  # renders on img_cv
                annotated_img = results.ims[0]
                annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                st.image(annotated_img, caption="Detection Results", use_column_width=True)
                
                # Show table of detections
                st.write("### Detected Items:")
                for idx, row in predictions.iterrows():
                    cls_name = row['name']
                    confidence = row['confidence']
                    bin_color = bin_colors.get(cls_name, 'Unknown')
                    
                    st.markdown(f"""
                    - **{cls_name.capitalize()}** (Confidence: {confidence:.2f})  
                      → Throw in **{bin_color} bin**  
                      _{bin_descriptions[bin_color]}_
                    """)
                
                # Summary: Most confident detection
                top_pred = predictions.loc[predictions['confidence'].idxmax()]
                top_class = top_pred['name']
                top_conf = top_pred['confidence']
                top_bin = bin_colors[top_class]
                
                st.success(f"**Main Item:** {top_class.capitalize()} → Use **{top_bin} Bin**")
                st.balloons()

# ------------------- Instructions -------------------
st.markdown("---")
st.markdown("""
### Indian Bin Color Guide:
- **Green** → Wet waste (food, peels)
- **Blue** → Dry recyclables (paper, plastic, metal, glass, cardboard)
- **Yellow** → Clothes, shoes, reusables
- **Red** → Batteries, e-waste, hazardous
- **Black** → Non-recyclable trash
""")

st.info("Place `best.pt` in the same folder as this `app.py` file.")
