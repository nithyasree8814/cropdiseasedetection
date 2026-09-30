```python
import os
import json
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Crop Disease AI",
    page_icon="🌱",
    layout="wide"
)

MODEL_DIR = "models"
IMG_SIZE = (224, 224)


# ==========================================
# 2. PROFESSIONAL UI
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #f4f8f4;
}

h1, h2, h3 {
    color: #14532d;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #d5e5d5;
}

.stButton > button {
    background-color: #15803d;
    color: white;
    border-radius: 8px;
    border: none;
    height: 45px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #166534;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD CLASS NAMES
# ==========================================

@st.cache_data
def load_class_names():

    path = os.path.join(MODEL_DIR, "class_names.json")

    if not os.path.exists(path):
        return []

    with open(path, "r") as file:
        return json.load(file)


class_names = load_class_names()


# ==========================================
# 4. LOAD TRAINED MODELS
# ==========================================

MODEL_FILES = {
    "MobileNetV2": "mobilenetv2.keras",
    "ResNet50": "resnet50.keras",
    "Custom CNN": "custom_cnn.keras"
}


@st.cache_resource
def load_model_file(model_name):

    path = os.path.join(
        MODEL_DIR,
        MODEL_FILES[model_name]
    )

    if not os.path.exists(path):
        return None

    return tf.keras.models.load_model(path)


# ==========================================
# 5. IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image, model_name):

    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    array = np.array(image, dtype=np.float32)

    # Model-specific preprocessing
    if model_name == "MobileNetV2":
        array = tf.keras.applications.mobilenet_v2.preprocess_input(
            array
        )

    elif model_name == "ResNet50":
        array = tf.keras.applications.resnet50.preprocess_input(
            array
        )

    else:
        # Custom CNN trained with rescaling 1/255
        array = array / 255.0

    return np.expand_dims(array, axis=0)


# ==========================================
# 6. DISEASE PREDICTION
# ==========================================

def predict_disease(model, image, model_name):

    input_image = preprocess_image(image, model_name)

    start_time = time.perf_counter()

    predictions = model.predict(
        input_image,
        verbose=0
    )[0]

    inference_time = time.perf_counter() - start_time

    top_indices = np.argsort(predictions)[::-1][:3]

    results = []

    for index in top_indices:

        results.append({
            "Disease": class_names[index],
            "Confidence": float(predictions[index])
        })

    return results, inference_time


# ==========================================
# 7. LEAF SEGMENTATION
# ==========================================

def segment_leaf(image):

    image = np.array(image.convert("RGB"))
    image = cv2.resize(image, IMG_SIZE)

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV
    )

    # Green, yellow and brown color range
    lower = np.array([5, 25, 20])
    upper = np.array([100, 255, 255])

    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Keep the largest connected component
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if count > 1:

        largest = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )

        mask = np.where(
            labels == largest,
            255,
            0
        ).astype(np.uint8)

    return image, mask


# ==========================================
# 8. SEVERITY ASSESSMENT
# ==========================================

def calculate_severity(image):

    image, leaf_mask = segment_leaf(image)

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV
    )

    # Approximate yellow and brown regions
    lower = np.array([10, 40, 40])
    upper = np.array([35, 255, 255])

    affected_mask = cv2.inRange(
        hsv,
        lower,
        upper
    )

    affected_mask = cv2.bitwise_and(
        affected_mask,
        leaf_mask
    )

    total_pixels = cv2.countNonZero(leaf_mask)
    affected_pixels = cv2.countNonZero(affected_mask)

    if total_pixels == 0:
        return 0, "Unable to estimate", leaf_mask, affected_mask

    percentage = (
        affected_pixels / total_pixels
    ) * 100

    if percentage < 5:
        severity = "Low"

    elif percentage < 15:
        severity = "Mild"

    elif percentage < 35:
        severity = "Moderate"

    else:
        severity = "High"

    return percentage, severity, leaf_mask, affected_mask


# ==========================================
# 9. FIND LAST CONVOLUTIONAL LAYER
# ==========================================

def find_last_conv_layer(model):

    for layer in reversed(model.layers):

        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

        if isinstance(layer, tf.keras.Model):

            try:
                nested_name = find_last_conv_layer(layer)

                if nested_name:
                    return layer.name + "/" + nested_name

            except Exception:
                pass

    return None


# ==========================================
# 10. GRAD-CAM
# ==========================================

def generate_gradcam(model, image, model_name, class_index):

    # Locate the final convolutional layer
    last_conv = None

    for layer in reversed(model.layers):

        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer
            break

        if isinstance(layer, tf.keras.Model):

            for nested_layer in reversed(layer.layers):

                if isinstance(nested_layer, tf.keras.layers.Conv2D):
                    last_conv = nested_layer
                    break

        if last_conv is not None:
            break

    if last_conv is None:
        raise ValueError(
            "No convolutional layer found for Grad-CAM."
        )

    # For models with a nested backbone, construct a
    # gradient model from the selected convolutional layer.
    if last_conv in model.layers:

        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[
                last_conv.output,
                model.output
            ]
        )

    else:
        raise ValueError(
            "Grad-CAM requires a directly accessible "
            "convolutional layer. This model has a nested "
            "backbone; expose its feature maps when building "
            "the model."
        )

    input_image = preprocess_image(image, model_name)

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            input_image,
            training=False
        )

        loss = predictions[:, class_index]

    gradients = tape.gradient(
        loss,
        conv_outputs
    )

    if gradients is None:
        raise ValueError("Gradients could not be calculated.")

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    heatmap = heatmap.numpy()

    heatmap = cv2.resize(
        heatmap,
        IMG_SIZE
    )

    return heatmap


def create_heatmap_overlay(image, heatmap):

    original = np.array(
        image.convert("RGB").resize(IMG_SIZE)
    )

    heatmap = np.uint8(255 * heatmap)

    colored = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    colored = cv2.cvtColor(
        colored,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original,
        0.6,
        colored,
        0.4,
        0
    )

    return colored, overlay


# ==========================================
# 11. SIDEBAR
# ==========================================

st.sidebar.title("🌱 Crop Disease AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Disease Detection",
        "Explainable AI",
        "Model Evaluation"
    ]
)

selected_model = st.sidebar.selectbox(
    "Select Deep Learning Model",
    list(MODEL_FILES.keys())
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Deep Learning | Computer Vision | Explainable AI"
)


# ==========================================
# 12. HOME PAGE
# ==========================================

if page == "Home":

    st.title("🌱 AI Crop Disease Detection")

    st.subheader(
        "Deep Learning-Based Disease Classification "
        "and Severity Assessment"
    )

    st.write(
        "An AI-powered plant disease analysis platform "
        "combining CNN classification, computer vision, "
        "and Explainable AI."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric("Deep Learning Models", "3")
    col2.metric("Image Processing", "OpenCV")
    col3.metric("Explainable AI", "Grad-CAM")

    st.markdown("### System Architecture")

    st.markdown("""
    **1. Image Acquisition**

    Upload a crop leaf image.

    **2. Image Preprocessing**

    Resize and normalize the image according to the
    selected deep learning model.

    **3. Disease Classification**

    Predict the disease using a trained CNN.

    **4. Severity Assessment**

    Segment the leaf and estimate the affected area.

    **5. Explainable AI**

    Generate a Grad-CAM visualization to show
    image regions influencing the prediction.

    **6. Results**

    Display disease predictions, confidence,
    severity, and visual explanations.
    """)

    st.info(
        "Navigate to Disease Detection to analyze an image."
    )


# ==========================================
# 13. DISEASE DETECTION PAGE
# ==========================================

elif page == "Disease Detection":

    st.title("🔬 Disease Detection")

    uploaded_file = st.file_uploader(
        "Upload a crop leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded Leaf")

        st.image(
            image,
            width=400
        )

        if st.button("Analyze Image"):

            model = load_model_file(selected_model)

            if model is None:
                st.error(
                    f"{selected_model} model not found. "
                    "Train the model and save it in the models folder."
                )
                st.stop()

            if not class_names:
                st.error(
                    "Class names not found. "
                    "Please add models/class_names.json."
                )
                st.stop()

            with st.spinner("Analyzing leaf image..."):

                predictions, inference_time = predict_disease(
                    model,
                    image,
                    selected_model
                )

                affected, severity, leaf_mask, affected_mask = (
                    calculate_severity(image)
                )

            st.success("Analysis completed!")

            st.markdown("---")
            st.subheader("Disease Prediction")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Predicted Disease",
                predictions[0]["Disease"]
            )

            col2.metric(
                "Confidence",
                f"{predictions[0]['Confidence'] * 100:.2f}%"
            )

            col3.metric(
                "Inference Time",
                f"{inference_time * 1000:.2f} ms"
            )

            st.markdown("### Top 3 Predictions")

            prediction_df = pd.DataFrame({
                "Disease": [
                    p["Disease"] for p in predictions
                ],
                "Confidence (%)": [
                    p["Confidence"] * 100 for p in predictions
                ]
            })

            st.bar_chart(
                prediction_df.set_index("Disease")
            )

            st.dataframe(
                prediction_df,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("---")
            st.subheader("Severity Assessment")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Estimated Affected Area",
                    f"{affected:.2f}%"
                )

                st.metric(
                    "Severity",
                    severity
                )

                st.progress(
                    min(int(affected), 100)
                )

            with col2:

                st.image(
                    leaf_mask,
                    caption="Segmented Leaf",
                    use_container_width=True
                )

            st.image(
                affected_mask,
                caption="Potentially Affected Regions",
                use_container_width=True
            )

            st.warning(
                "Severity is an approximate color-based estimate, "
                "not a validated disease measurement."
            )

            st.session_state["last_image"] = image
            st.session_state["last_model"] = selected_model
            st.session_state["last_class"] = int(
                np.argmax(
                    model.predict(
                        preprocess_image(image, selected_model),
                        verbose=0
                    )[0]
                )
            )


# ==========================================
# 14. EXPLAINABLE AI PAGE
# ==========================================

elif page == "Explainable AI":

    st.title("🧠 Explainable AI")

    st.subheader("Grad-CAM Visualization")

    if "last_image" not in st.session_state:

        st.info(
            "Analyze an image on the Disease Detection page first."
        )

    else:

        image = st.session_state["last_image"]
        model_name = st.session_state["last_model"]
        class_index = st.session_state["last_class"]

        model = load_model_file(model_name)

        if model is None:
            st.error("Model not found.")

        else:

            try:

                heatmap = generate_gradcam(
                    model,
                    image,
                    model_name,
                    class_index
                )

                colored, overlay = create_heatmap_overlay(
                    image,
                    heatmap
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.image(
                        image,
                        caption="Original Image",
                        use_container_width=True
                    )

                with col2:
                    st.image(
                        colored,
                        caption="Grad-CAM Heatmap",
                        use_container_width=True
                    )

                with col3:
                    st.image(
                        overlay,
                        caption="Heatmap Overlay",
                        use_container_width=True
                    )

                st.markdown("### Interpretation")

                st.write(
                    "Grad-CAM uses gradients flowing into a "
                    "convolutional layer to highlight image "
                    "regions relevant to the selected prediction. "
                    "The highlighted regions show model attention, "
                    "not necessarily the exact diseased tissue."
                )

            except Exception as error:

                st.error(
                    f"Grad-CAM could not be generated: {error}"
                )


# ==========================================
# 15. MODEL EVALUATION PAGE
# ==========================================

elif page == "Model Evaluation":

    st.title("📊 Model Evaluation")

    evaluation_path = os.path.join(
        MODEL_DIR,
        "evaluation.csv"
    )

    if os.path.exists(evaluation_path):

        df = pd.read_csv(evaluation_path)

        st.subheader("Model Performance Comparison")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        metric = st.selectbox(
            "Select Evaluation Metric",
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ]
        )

        if metric in df.columns:

            st.bar_chart(
                df.set_index("Model")[metric]
            )

    else:

        st.warning(
            "Evaluation results are unavailable. "
            "Train and evaluate your models first."
        )

        st.write(
            "The evaluation dashboard will display accuracy, "
            "precision, recall, and F1-score when the results "
            "are available."
        )


# ==========================================
# 16. FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "Crop Disease AI | Deep Learning | Computer Vision "
    "| Explainable AI"
)
```
