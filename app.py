
import os
import json
import requests
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Crop Disease AI",page_icon="🌿",layout="wide")

MODEL_PATH="best_mnv2_v2.keras"
MODEL_URL="https://huggingface.co/ashikaasriarun/MobileNet-V2-NewPlantDisease/resolve/main/best_mnv2_v2.keras"
IMAGE_SIZE=160
CLASS_NAMES=[
"Apple___Apple_scab",
"Apple___Black_rot",
"Apple___Cedar_apple_rust",
"Apple___healthy",
"Blueberry___healthy",
"Cherry_(including_sour)___Powdery_mildew",
"Cherry_(including_sour)___healthy",
"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
"Corn_(maize)___Common_rust_",
"Corn_(maize)___Northern_Leaf_Blight",
"Corn_(maize)___healthy",
"Grape___Black_rot",
"Grape___Esca_(Black_Measles)",
"Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
"Grape___healthy",
"Orange___Haunglongbing_(Citrus_greening)",
"Peach___Bacterial_spot",
"Peach___healthy",
"Pepper,_bell___Bacterial_spot",
"Pepper,_bell___healthy",
"Potato___Early_blight",
"Potato___Late_blight",
"Potato___healthy",
"Raspberry___healthy",
"Soybean___healthy",
"Squash___Powdery_mildew",
"Strawberry___Leaf_scorch",
"Strawberry___healthy",
"Tomato___Bacterial_spot",
"Tomato___Early_blight",
"Tomato___Late_blight",
"Tomato___Leaf_Mold",
"Tomato___Septoria_leaf_spot",
"Tomato___Spider_mites Two-spotted_spider_mite",
"Tomato___Target_Spot",
"Tomato___Tomato_Yellow_Leaf_Curl_Virus",
"Tomato___Tomato_mosaic_virus",
"Tomato___healthy"
]

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading the pretrained model. Please wait..."):
            response=requests.get(MODEL_URL,stream=True,timeout=300)
            response.raise_for_status()
            with open(MODEL_PATH,"wb") as file:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        file.write(chunk)
    return tf.keras.models.load_model(MODEL_PATH,compile=False)

def preprocess_image(image):
    image=image.convert("RGB").resize((IMAGE_SIZE,IMAGE_SIZE))
    array=np.array(image,dtype=np.float32)
    return np.expand_dims(array,axis=0)

def predict_disease(model,image):
    array=preprocess_image(image)
    predictions=np.asarray(model.predict(array,verbose=0)[0]).flatten()
    if len(predictions)!=len(CLASS_NAMES):
        raise ValueError(f"Model returned {len(predictions)} predictions, but {len(CLASS_NAMES)} class names are configured.")
    if np.any(predictions<0) or not np.isclose(np.sum(predictions),1,atol=0.05):
        predictions=tf.nn.softmax(predictions).numpy()
    indices=np.argsort(predictions)[::-1][:3]
    return [(CLASS_NAMES[i].replace("___"," - ").replace("_"," ").strip(),float(predictions[i])*100) for i in indices]

def estimate_severity(image):
    array=np.array(image.convert("RGB").resize((224,224)))
    hsv=cv2.cvtColor(array,cv2.COLOR_RGB2HSV)
    green=cv2.inRange(hsv,np.array([25,35,25]),np.array([95,255,255]))
    yellow=cv2.inRange(hsv,np.array([10,40,40]),np.array([35,255,255]))
    brown=cv2.inRange(hsv,np.array([0,40,20]),np.array([25,255,200]))
    leaf_pixels=np.count_nonzero(green|yellow|brown)
    if leaf_pixels==0:
        return 0,None
    affected=cv2.bitwise_or(yellow,brown)
    affected=cv2.bitwise_and(affected,cv2.bitwise_not(green))
    affected_pixels=np.count_nonzero(affected)
    severity=min(100,affected_pixels/leaf_pixels*100)
    return severity,affected

def find_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer,tf.keras.layers.Conv2D):
            return layer
        if isinstance(layer,tf.keras.Model):
            found=find_conv_layer(layer)
            if found:
                return found
    return None

def make_gradcam(model,image):
    conv_layer=find_conv_layer(model)
    if conv_layer is None:
        raise ValueError("No convolutional layer was found.")
    try:
        grad_model=tf.keras.Model(model.inputs,[conv_layer.output,model.output])
        input_array=preprocess_image(image)
        with tf.GradientTape() as tape:
            conv_outputs,predictions=grad_model(input_array,training=False)
            class_index=tf.argmax(predictions[0])
            loss=predictions[:,class_index]
        gradients=tape.gradient(loss,conv_outputs)
        if gradients is None:
            raise ValueError("Gradients could not be calculated for this model.")
        weights=tf.reduce_mean(gradients,axis=(0,1,2))
        heatmap=tf.reduce_sum(conv_outputs[0]*weights,axis=-1)
        heatmap=tf.maximum(heatmap,0)
        heatmap=heatmap/(tf.reduce_max(heatmap)+tf.keras.backend.epsilon())
        heatmap=cv2.resize(heatmap.numpy(),(224,224))
        heatmap=np.uint8(255*heatmap)
        colored=cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
        colored=cv2.cvtColor(colored,cv2.COLOR_BGR2RGB)
        original=np.array(image.convert("RGB").resize((224,224)))
        return cv2.addWeighted(original,0.6,colored,0.4,0)
    except Exception as e:
        raise ValueError(f"Grad-CAM is not supported for this model structure: {e}")

def display_results(results):
    st.subheader("Prediction Results")
    st.metric("Predicted Disease",results[0][0])
    st.metric("Confidence",f"{results[0][1]:.2f}%")
    df=pd.DataFrame(results,columns=["Disease","Confidence (%)"])
    st.subheader("Top 3 Predictions")
    st.dataframe(df,use_container_width=True,hide_index=True)
    st.bar_chart(df.set_index("Disease"))

st.sidebar.title("🌿 Crop Disease AI")
page=st.sidebar.radio("Navigation",["Home","Disease Detection","Explainable AI","Model Evaluation"])

if page=="Home":
    st.title("🌿 AI-Based Crop Disease Detection")
    st.write("Detect plant diseases from leaf images using a pretrained MobileNetV2 deep learning model.")
    st.subheader("Project Features")
    col1,col2=st.columns(2)
    with col1:
        st.info("📷 Leaf image upload")
        st.info("🧠 Deep learning disease prediction")
        st.info("📊 Top-3 prediction results")
    with col2:
        st.info("🔥 Grad-CAM visualization when supported")
        st.info("🌱 Estimated leaf damage")
        st.info("📈 Model evaluation dashboard")
    st.warning("The model is trained on the PlantVillage dataset. Predictions on real-world images may be less accurate.")

elif page=="Disease Detection":
    st.title("🔬 Crop Disease Detection")
    uploaded=st.file_uploader("Upload a plant leaf image",type=["jpg","jpeg","png"])
    if uploaded:
        image=Image.open(uploaded).convert("RGB")
        st.image(image,caption="Uploaded Leaf Image",use_container_width=True)
        if st.button("Detect Disease",type="primary"):
            try:
                model=load_model()
                with st.spinner("Analyzing leaf image..."):
                    results=predict_disease(model,image)
                    severity,mask=estimate_severity(image)
                st.session_state["results"]=results
                st.session_state["severity"]=severity
                st.session_state["image"]=image
                st.success("Prediction completed!")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    if "results" in st.session_state:
        display_results(st.session_state["results"])
        severity=st.session_state["severity"]
        st.subheader("Estimated Leaf Damage")
        st.metric("Estimated Affected Area",f"{severity:.2f}%")
        st.progress(int(severity))
        st.caption("This is a rough color-based estimate, not a validated disease-severity measurement.")

elif page=="Explainable AI":
    st.title("🔥 Explainable AI - Grad-CAM")
    st.write("Grad-CAM highlights image regions that may have influenced the model's prediction.")
    uploaded=st.file_uploader("Upload a leaf image",type=["jpg","jpeg","png"],key="gradcam_image")
    if uploaded:
        image=Image.open(uploaded).convert("RGB")
        st.image(image,caption="Input Image",use_container_width=True)
        if st.button("Generate Grad-CAM",type="primary"):
            try:
                model=load_model()
                with st.spinner("Generating visualization..."):
                    overlay=make_gradcam(model,image)
                st.image(overlay,caption="Grad-CAM Heatmap",use_container_width=True)
            except Exception as e:
                st.error(str(e))

elif page=="Model Evaluation":
    st.title("📊 Model Evaluation")
    st.write("Upload a CSV file containing evaluation metrics calculated on a test dataset.")
    uploaded=st.file_uploader("Upload evaluation.csv",type=["csv"])
    if uploaded:
        try:
            df=pd.read_csv(uploaded)
            st.subheader("Evaluation Metrics")
            st.dataframe(df,use_container_width=True)
            numeric=df.select_dtypes(include=np.number)
            if not numeric.empty:
                st.subheader("Metric Visualization")
                st.bar_chart(numeric)
        except Exception as e:
            st.error(f"Could not read the CSV file: {e}")
    else:
        st.info("No evaluation file uploaded. This page will display metrics only when you provide test results.")

