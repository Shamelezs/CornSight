import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image

# Constants for Two Models (Each with a Different API Key)
API_KEY_1 = "NAGugbciLHsoK9sRrffV"  # API Key for Model 1 (HandHeld)
MODEL_1_ID = "corn-grayleafspot2/22"

API_KEY_2 = "7btX2iOoSgHFPVC73CUJ"  # API Key for Model 2 (Drone Based)
MODEL_2_ID = "testing-cmpv3/1"

# Streamlit UI
st.set_page_config(page_title="Corn Sight System", page_icon=":corn:", layout="wide")

# Create Tabs for Two Models
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📱 HandHeld", "🚁 DroneBased", "ℹ️ About"])

# Function to Perform Object Detection with a Specific API Key
def detect_objects(uploaded_file, model_id, api_key, font_scale, font_type):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)

        with st.spinner("Processing... Please wait"):
            url = f"https://detect.roboflow.com/{model_id}?api_key={api_key}"
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(url, files=files)

            if response.status_code == 200:
                data = response.json()
                detected_image = image.copy()
                class_counts = {}

                # Assign specific colors for each class
                class_colors = {
                    "Gray Leaf Spot": (255, 0, 0),  # Red
                    "Healthy": (0, 255, 0),  # Green
                    "default": (255, 255, 0)  # Yellow for unknown classes
                }

                for prediction in data.get("predictions", []):
                    x, y, width, height = int(prediction["x"]), int(prediction["y"]), int(prediction["width"]), int(prediction["height"])
                    label = prediction["class"]
                    confidence = float(prediction["confidence"]) * 100

                    if label in class_counts:
                        class_counts[label] += 1
                    else:
                        class_counts[label] = 1

                    x1, y1, x2, y2 = x - width // 2, y - height // 2, x + width // 2, y + height // 2
                    color = class_colors.get(label, class_colors["default"])

                    cv2.rectangle(detected_image, (x1, y1), (x2, y2), color, 4)

                    text = f"{label}: {confidence:.2f}%"
                    thickness = 2
                    text_size = cv2.getTextSize(text, font_type, font_scale, thickness)[0]
                    text_x, text_y = x1, y1 - 10
                    cv2.rectangle(detected_image, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), color, -1)
                    cv2.putText(detected_image, text, (text_x, text_y), font_type, font_scale, (0, 0, 0), thickness)

                return image, detected_image, class_counts

            else:
                st.error("Detection failed. Please try again.")
                return None, None, None

# Tab 1 >> Home
with tab1:
    st.subheader("Home")
    st.markdown(
    """
    <div style="text-align: center;">
        <p style ="font-size: 24px;">Welcome to Corn Sight System</p>
    </div>
    """,
    unsafe_allow_html=True
    )
    st.markdown("##### Abstract")
    st.markdown(
    """
    <div style="text-align: justify;">
        This study aims to enhance white corn disease detection 
        in the Philippines, given the crop's role as a staple food. 
        The production of white corn in Laguna experienced a slight increase in early 2023, 
        although the country's total corn inventory saw a significant decrease by mid-2024. 
        With plant diseases causing substantial global crop losses, there is a pressing need for 
        efficient and precise diagnosis methods. Traditional visual inspection by farmers is not 
        always feasible, especially in challenging terrains or with tall crops. Therefore, this 
        research will evaluate the effectiveness of drone-based disease detection using Red Green Near-infrared 
        (RGNIR) imaging at different corn maturity stages. The study will be conducted in Barangay Concepcion 
        Lumban, Laguna, where corn cultivation is a primary income source. The results of this study 
        will provide valuable insights into improving crop health monitoring and potentially mitigating 
        yield losses.
        <br><br><!-- Add line breaks for space -->
    </div>
    """,
    unsafe_allow_html=True
    )

# Tab 2 >> Handheld Detection (Uses API Key 1)
with tab2:
    st.subheader("Handheld")
    
    # Custom CSS for Handheld Tab
    st.markdown(
    """
    <style>
        .handheld-title {
            font-size: 30px;
            font-weight: bold;
            color: #A855F7;
        }
        .handheld-metric {
            font-size: 20px;
            color: #3B82F6;
        }
    </style>
    """,
    unsafe_allow_html=True
    )
    
    st.markdown('<p class="handheld-title">Handheld Detection Metrics</p>', unsafe_allow_html=True)

    # Data
    metrics = [
         {"name": "mAP", "value": 83.9, "color": "#A855F7"},
        {"name": "Precision", "value": 95.6, "color": "#3B82F6"},
        {"name": "Recall", "value": 76, "color": "#F59E0B"}
    ]

    # Display metrics with values properly aligned next to the graph
    for metric in metrics:
        cols = st.columns([1, 1, 3])
        with cols[0]:
            st.write(metric['name'])
        with cols[1]:
            st.write(f"<span class='handheld-metric'>**{metric['value']}%**</span>", unsafe_allow_html=True)
        with cols[2]:
            st.markdown(
                f"""
                <div style="width: 
                {metric['value']}%; height: 20px; background-color: {metric['color']}; border-radius: 10px;">
                </div>
                """,
            unsafe_allow_html=True
        )

    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True, key="Phone")
    if uploaded_files:
        for uploaded_file in uploaded_files:
            image, detected_image, class_counts = detect_objects(uploaded_file, MODEL_1_ID, API_KEY_1, font_scale=2.5, font_type=cv2.FONT_ITALIC)
            if image is not None and detected_image is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Uploaded Image")
                    st.image(image, caption="Uploaded Image", use_container_width=True)

                with col2:
                    st.subheader("Processed Image")
                    st.image(detected_image, caption="Detected Objects", use_container_width=True)

                st.subheader("Detected Objects Count:")
                for class_name, count in class_counts.items():
                    st.write(f"{class_name}: {count}")

# Tab 3 >> DroneBased Detection (Uses API Key 2)
with tab3:
    st.subheader("DroneBased")
    
    # Custom CSS for Drone-Based Tab
    st.markdown(
    """
    <style>
        .drone-title {
            font-size: 30px;
            font-weight: bold;
            color: #F59E0B;
        }
        .drone-metric {
            font-size: 20px;
            color: #3B82F6;
        }
    </style>
    """,
    unsafe_allow_html=True
    )
    
    st.markdown('<p class="drone-title">Drone-Based Detection Metrics</p>', unsafe_allow_html=True)

    # Data
    metrics = [
         {"name": "mAP", "value": 80, "color": "#A855F7"},
        {"name": "Precision", "value": 86.3, "color": "#3B82F6"},
        {"name": "Recall", "value": 77.5, "color": "#F59E0B"}
    ]

    # Display metrics with values properly aligned next to the graph
    for metric in metrics:
        cols = st.columns([1, 1, 3])
        with cols[0]:
            st.write(metric['name'])
        with cols[1]:
            st.write(f"<span class='drone-metric'>**{metric['value']}%**</span>", unsafe_allow_html=True)
        with cols[2]:
            st.markdown(
                f"""
                <div style="width: 
                {metric['value']}%; height: 20px; background-color: {metric['color']}; border-radius: 10px;">
                </div>
                """,
            unsafe_allow_html=True
        )

    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True, key="Drone")
    if uploaded_files:
        for uploaded_file in uploaded_files:
            image, detected_image, class_counts = detect_objects(uploaded_file, MODEL_2_ID, API_KEY_2, font_scale=0.5, font_type=cv2.FONT_ITALIC)
            if image is not None and detected_image is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Uploaded Image")
                    st.image(image, caption="Uploaded Image", use_container_width=True)

                with col2:
                    st.subheader("Processed Image")
                    st.image(detected_image, caption="Detected Objects", use_container_width=True)

                st.subheader("Detected Objects Count:")
                for class_name, count in class_counts.items():
                    st.write(f"{class_name}: {count}")

# Tab 4 >> About 
with tab4:
    st.subheader("About")
    st.image(image="tarp.jpg", use_container_width=True)
    st.markdown(
    """
    <div style="text-align: center; font-size: 20px; font-weight: normal;">
        Poster
    </div>
    """,
    unsafe_allow_html=True
    )
