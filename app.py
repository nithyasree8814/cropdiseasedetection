import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image

st.set_page_config(page_title="AI Crop Disease Detection",page_icon="🌿",layout="wide")

MODEL_PATH="models/resnet50.keras"
IMG_SIZE=224

CLASS_NAMES=[
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca",
    "Grape___Leaf_blight",
    "Grape___healthy",
    "Orange___Haunglongbing",
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
        return None
    return tf.keras.models.load_model(MODEL_PATH,compile=False)

def preprocess_image(image):
    image=image.convert("RGB").resize((IMG_SIZE,IMG_SIZE))
    array=np.array(image,dtype=np.float32)
    array=np.expand_dims(array,axis=0)
    return array/255.0

def predict_disease(model,image):
    processed=preprocess_image(image)
    predictions=model.predict(processed,verbose=0)[0]
    index=int(np.argmax(predictions))
    confidence=float(predictions[index])*100
    if len(predictions)!=len(CLASS_NAMES):
        raise ValueError(
            f"Model has {len(predictions)} output classes, "
            f"but {len(CLASS_NAMES)} class names are configured."
        )
    return CLASS_NAMES[index],confidence,predictions

def estimate_severity(image):
    img=np.array(image.convert("RGB"))
    hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    lower=np.array([5,40,40])
    upper=np.array([35,255,255])
    mask=cv2.inRange(hsv,lower,upper)
    affected=np.count_nonzero(mask)
    total=img.shape[0]*img.shape[1]
    percentage=(affected/total)*100
    if percentage<5:
        level="Healthy or very mild"
    elif percentage<15:
        level="Mild"
    elif percentage<35:
        level="Moderate"
    else:
        level="Severe"
    return level,percentage

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer,tf.keras.layers.Conv2D):
            return layer
        if isinstance(layer,tf.keras.Model):
            found=find_last_conv_layer(layer)
            if found is not None:
                return found
    return None

def make_gradcam(model,image):
    try:
        conv_layer=find_last_conv_layer(model)
        if conv_layer is None:
            raise ValueError("No convolutional layer was found.")

        grad_model=tf.keras.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output,model.output]
        )

        processed=preprocess_image(image)

        with tf.GradientTape() as tape:
            conv_outputs,predictions=grad_model(processed,training=False)
            class_index=tf.argmax(predictions[0])
            loss=predictions[:,class_index]

        gradients=tape.gradient(loss,conv_outputs)

        if gradients is None:
            raise ValueError("Gradients could not be calculated for this model.")

        weights=tf.reduce_mean(gradients,axis=(0,1,2))
        conv_outputs=conv_outputs[0]
        heatmap=tf.reduce_sum(conv_outputs*weights,axis=-1)
        heatmap=tf.maximum(heatmap,0)

        maximum=tf.reduce_max(heatmap)
        if float(maximum)==0:
            raise ValueError("Grad-CAM generated an empty heatmap.")

        heatmap=heatmap/maximum
        heatmap=heatmap.numpy()
        heatmap=cv2.resize(heatmap,(IMG_SIZE,IMG_SIZE))
        heatmap=np.uint8(255*heatmap)
        heatmap=cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
        heatmap=cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB)

        original=np.array(image.convert("RGB").resize((IMG_SIZE,IMG_SIZE)))
        overlay=cv2.addWeighted(original,0.6,heatmap,0.4,0)
        return overlay

    except Exception as e:
        raise ValueError(str(e))

st.title("🌿 AI Crop Disease Detection")
st.write("Upload a crop leaf image to predict its disease and estimate visible damage.")

page=st.sidebar.radio(
    "Navigation",
    ["Home","Disease Detection","Explainable AI","About"]
)

model=load_model()

if page=="Home":
    st.header("Welcome")
    st.write(
        "This application demonstrates crop disease detection using "
        "a deep learning model, with an image-based severity estimate "
        "and an optional Grad-CAM visualization."
    )
    st.info("Upload a leaf image in the Disease Detection section to begin.")

elif page=="Disease Detection":
    st.header("Disease Detection")
    uploaded_file=st.file_uploader(
        "Upload a leaf image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file:
        image=Image.open(uploaded_file).convert("RGB")
        st.image(image,caption="Uploaded Leaf Image",width=350)

        if model is None:
            st.error(
                f"Model file not found: {MODEL_PATH}. "
                "Place your trained model at this location."
            )
        else:
            if st.button("Detect Disease"):
                with st.spinner("Analyzing image..."):
                    try:
                        disease,confidence,predictions=predict_disease(model,image)
                        severity,percentage=estimate_severity(image)

                        st.success("Prediction completed")
                        st.subheader("Prediction Results")
                        st.write("**Predicted class:**",disease.replace("___"," - "))
                        st.metric("Model confidence",f"{confidence:.2f}%")
                        st.write("**Estimated visible severity:**",severity)
                        st.metric("Estimated affected area",f"{percentage:.2f}%")

                        st.subheader("Top 3 Predictions")
                        top_indices=np.argsort(predictions)[-3:][::-1]
                        for index in top_indices:
                            st.write(
                                f"{CLASS_NAMES[index].replace('___',' - ')}: "
                                f"{predictions[index]*100:.2f}%"
                            )

                        st.caption(
                            "Severity is a simple color-based estimate, "
                            "not a medically or agriculturally validated measurement."
                        )

                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

elif page=="Explainable AI":
    st.header("Explainable AI - Grad-CAM")
    uploaded_file=st.file_uploader(
        "Upload a leaf image for Grad-CAM",
        type=["jpg","jpeg","png"],
        key="gradcam_upload"
    )

    if uploaded_file:
        image=Image.open(uploaded_file).convert("RGB")
        st.image(image,caption="Uploaded Image",width=350)

        if model is None:
            st.error(
                f"Model file not found: {MODEL_PATH}. "
                "Place your trained model at this location."
            )
        elif st.button("Generate Grad-CAM"):
            with st.spinner("Generating visualization..."):
                try:
                    result=make_gradcam(model,image)
                    st.image(
                        result,
                        caption="Grad-CAM Visualization",
                        width=450
                    )
                    st.caption(
                        "Highlighted areas indicate image regions that "
                        "contributed to the model's prediction."
                    )
                except Exception as e:
                    st.error(f"Grad-CAM is not supported for this model structure: {e}")

elif page=="About":
    st.header("About the Project")
    st.write("**Project:** AI Crop Disease Detection and Severity Assessment")
    st.write("**Framework:** Streamlit")
    st.write("**Deep Learning:** TensorFlow and Keras")
    st.write("**Image Processing:** OpenCV")
    st.write("**Visualization:** Grad-CAM")
    st.warning(
        "The application requires a compatible trained model. "
        "Severity estimation and Grad-CAM results should be validated "
        "before being used for real agricultural decisions."
    )
