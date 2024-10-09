import tensorflow as tf
import streamlit as st
import numpy as np  # Corrected typo from nps to np
from tensorflow import keras
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model(r'C:\Users\Om Chaudhari\OneDrive\Desktop\Project SEM V\Image_classify.keras')

# Define categories
data_cat = ['O', 'R']

# Define image dimensions
img_height = 180
img_width = 180

# Set header
st.header("EcoSort AI")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpeg", "jpg", "png"])

if uploaded_file is not None:
    # Load the image
    image_load = tf.keras.utils.load_img(uploaded_file, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_load)  # Corrected function from array_to_img to img_to_array
    img_bat = tf.expand_dims(img_arr, axis=0)

    # Make predictions
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)

    # Display the image at a smaller size (e.g., width = 300)
    st.image(image_load, caption='Uploaded Image', width=300)

    # Show prediction results
    st.write('Waste in image is **{}** with accuracy of **{:0.2f}%**'.format(data_cat[np.argmax(score)], np.max(score) * 100))
