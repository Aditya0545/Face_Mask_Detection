import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('face_mask_detection_model.h5')

# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Home", "Face Mask Detection", "About"])

# Home page
if option == "Home":
    st.title("Welcome to the Face Mask Detection App")
    # Display an image
    st.image('background.png', caption='Your Image Caption', use_column_width=True)
    st.write("""
        This app allows you to upload an image and detect whether the person in the image is wearing a face mask or not.
        Navigate to the 'Face Mask Detection' section to start using the app.
    """)

# Face Mask Detection page
elif option == "Face Mask Detection":
    st.title('Face Mask Detection')
    
    # Function to predict mask
    def predict_mask(image):
        image = image.resize((128, 128))
        image = np.array(image)
        image = image / 255.0  # Normalize the image
        image = np.reshape(image, (1, 128, 128, 3))

        prediction = model.predict(image)
        return np.argmax(prediction)

    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the uploaded image
        image = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Predict mask or no mask
        if st.button('Predict'):
            label = predict_mask(image)
            if label == 1:
                st.write("The person in the image is **wearing a mask**.")
            else:
                st.write("The person in the image is **not wearing a mask**.")

# About page
elif option == "About":
    st.title("About")
    st.write("""
        This Face Mask Detection app is built using TensorFlow and Streamlit. 
        It uses a pre-trained MobileNetV2 model to classify images as 'with mask' or 'without mask'. 
        You can use this app to test the model with your own images.
    """)
