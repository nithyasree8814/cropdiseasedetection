
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Crop Disease AI",page_icon="🌿",layout="wide")

MODEL_PATHS={
    "MobileNetV2":"models/mobilenetv2.keras",
    "ResNet50":"models/resnet50.keras",
    "Custom CNN":"models/custom_cnn.keras"
}

CLASS_NAMES_PATH="models/class_names.json"

@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

@st.cache_data
def load_class_names():
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH,"r") as f:
            names=json.load(f)
        if isinstance(names,dict):
            return [names[str(i)] for i in sorted(map(int,names.keys()))]
        return names
    return []

def preprocess_image(image):
    image=image.convert("RGB").resize((224,224))
    image_array=np.array(image,dtype=np.float32)
    return np.expand_dims(image_array,axis=0)

def predict_disease(model,image,class_names):
    processed=preprocess_image(image)
    predictions=model.predict(processed,verbose=0)[0]
    predictions=np.asarray(predictions).flatten()
    if np.any(predictions<0) or not np.isclose(np.sum(predictions),1,atol=0.05):
        predictions=tf.nn.softmax(predictions).numpy()
    top_indices=np.argsort(predictions)[::-1][:3]
    results=[]
    for index in top_indices:
        name=class_names[index] if index<len(class_names) else f"Class {index}"
        results.append((name,float(predictions[index])*100))
    return results

def estimate_severity(image):
    image_array=np.array(image.convert("RGB").resize((224,224)))
    hsv=cv2.cvtColor(image_array,cv2.COLOR_RGB2HSV)
    green_mask=cv2.inRange(hsv,np.array([25,35,25]),np.array([95,255,255]))
    leaf_pixels=np.count_nonzero(green_mask)
    if leaf_pixels==0:
        return 0.0,None
    yellow_mask=cv2.inRange(hsv,np.array([10,40,40]),np.array([35,255,255]))
    brown_mask=cv2.inRange(hsv,np.array([0,40,20]),np.array([25,255,200]))
    affected=cv2.bitwise_or(yellow_mask,brown_mask)
    affected=cv2.bitwise_and(affected,cv2.bitwise_not(green_mask))
    affected_pixels=np.count_nonzero(affected)
    severity=min(100,(affected_pixels/leaf_pixels)*100)
    return severity,affected

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer,tf.keras.layers.Conv2D):
            return layer.name
    return None

def make_gradcam(model,image):
    layer_name=find_last_conv_layer(model)
    if layer_name is None:
        raise ValueError("No directly accessible Conv2D layer was found.")
    grad_model=tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output,model.output]
    )
    image_array=preprocess_image(image)
    with tf.GradientTape() as tape:
        conv_outputs,predictions=grad_model(image_array,training=False)
        class_index=tf.argmax(predictions[0])
        loss=predictions[:,class_index]
    gradients=tape.gradient(loss,conv_outputs)
    if gradients is None:
        raise ValueError("Grad-CAM could not calculate gradients for this model.")
    pooled_gradients=tf.reduce_mean(gradients,axis=(0,1,2))
    conv_outputs=conv_outputs[0]
    heatmap=tf.reduce_sum(conv_outputs*pooled_gradients,axis=-1)
    heatmap=tf.maximum(heatmap,0)
    maximum=tf.reduce_max(heatmap)
    heatmap=heatmap/(maximum+tf.keras.backend.epsilon())
    heatmap=cv2.resize(heatmap.numpy(),(224,224))
    original=np.array(image.convert("RGB").resize((224,224)))
    heatmap=np.uint8(255*heatmap)
    colored=cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
    colored=cv2.cvtColor(colored,cv2.COLOR_BGR2RGB)
    overlay=cv2.addWeighted(original,0.6,colored,0.4,0)
    return overlay

st.sidebar.title("🌿 Crop Disease AI")
page=st.sidebar.radio("Navigation",["Home","Disease Detection","Explainable AI","Model Evaluation"])
st.sidebar.markdown("---")
model_name=st.sidebar.selectbox("Select Model",list(MODEL_PATHS.keys()))
model_path=MODEL_PATHS[model_name]

if page=="Home":
    st.title("🌿 AI-Based Crop Disease Detection")
    st.write("An image-based crop disease detection system using deep learning.")
    st.subheader("Project Features")
    col1,col2=st.columns(2)
    with col1:
        st.info("📷 Upload a crop leaf image")
        st.info("🧠 Predict disease using a trained model")
    with col2:
        st.info("🔍 View model predictions")
        st.info("🔥 Visualize model attention using Grad-CAM")
    st.warning("The application requires trained model files and a class_names.json file in the models folder.")

elif page=="Disease Detection":
    st.title("🔬 Crop Disease Detection")
    st.write(f"Selected model: **{model_name}**")
    uploaded_file=st.file_uploader("Upload a leaf image",type=["jpg","jpeg","png"])
    if uploaded_file:
        image=Image.open(uploaded_file).convert("RGB")
        st.image(image,caption="Uploaded Leaf Image",use_container_width=True)
        if st.button("Detect Disease",type="primary"):
            if not os.path.exists(model_path):
                st.error(f"Model file not found: {model_path}")
            else:
                try:
                    with st.spinner("Analyzing image..."):
                        model=load_model(model_path)
                        class_names=load_class_names()
                        results=predict_disease(model,image,class_names)
                        severity,mask=estimate_severity(image)
                    st.session_state["image"]=image
                    st.session_state["results"]=results
                    st.session_state["severity"]=severity
                    st.session_state["model_name"]=model_name
                    st.success("Prediction completed!")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
    if "results" in st.session_state:
        st.subheader("Prediction Results")
        results=st.session_state["results"]
        st.metric("Predicted Disease",results[0][0])
        st.metric("Prediction Confidence",f"{results[0][1]:.2f}%")
        st.subheader("Top 3 Predictions")
        result_df=pd.DataFrame(results,columns=["Disease","Confidence (%)"])
        st.dataframe(result_df,use_container_width=True,hide_index=True)
        st.bar_chart(result_df.set_index("Disease"))
        severity=st.session_state["severity"]
        st.subheader("Estimated Leaf Damage")
        st.metric("Estimated Affected Area",f"{severity:.2f}%")
        st.progress(int(severity))
        st.caption("This is a color-based estimate, not a validated disease-severity measurement.")

elif page=="Explainable AI":
    st.title("🔥 Explainable AI - Grad-CAM")
    st.write("Grad-CAM highlights image regions that influence the model's prediction.")
    uploaded_file=st.file_uploader("Upload a leaf image for Grad-CAM",type=["jpg","jpeg","png"],key="gradcam_upload")
    if uploaded_file:
        image=Image.open(uploaded_file).convert("RGB")
        st.image(image,caption="Input Image",use_container_width=True)
        if st.button("Generate Grad-CAM",type="primary"):
            if not os.path.exists(model_path):
                st.error(f"Model file not found: {model_path}")
            else:
                try:
                    with st.spinner("Generating Grad-CAM..."):
                        model=load_model(model_path)
                        overlay=make_gradcam(model,image)
                    st.image(overlay,caption="Grad-CAM Visualization",use_container_width=True)
                except Exception as e:
                    st.error(f"Grad-CAM generation failed: {e}")
                    st.info("This implementation requires a model with a directly accessible Conv2D layer connected to its output.")

elif page=="Model Evaluation":
    st.title("📊 Model Evaluation")
    st.write(f"Selected model: **{model_name}**")
    evaluation_path="evaluation.csv"
    if os.path.exists(evaluation_path):
        try:
            evaluation_df=pd.read_csv(evaluation_path)
            st.subheader("Evaluation Results")
            st.dataframe(evaluation_df,use_container_width=True)
            numeric_columns=evaluation_df.select_dtypes(include=np.number).columns.tolist()
            if numeric_columns:
                st.subheader("Metric Visualization")
                st.bar_chart(evaluation_df.set_index(evaluation_df.columns[0])[numeric_columns])
        except Exception as e:
            st.error(f"Could not read evaluation.csv: {e}")
    else:
        st.warning("evaluation.csv was not found.")
        st.write("Add an evaluation.csv file containing the metrics calculated from your test dataset.")

