import tensorflow as tf
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('model/Image_classify.keras')

# Define categories
data_cat = ['Organic', 'Recyclable']

# Define image dimensions
img_height = 180
img_width = 180

# Set header and description
st.title("EcoSort AI: Waste Classification")
st.write("Upload an image of waste, and this AI will classify it as either Organic or Recyclable!")

# Upload image
uploaded_file = st.file_uploader("Choose an image file (JPEG, JPG, PNG)...", type=["jpeg", "jpg", "png"])

if uploaded_file is not None:
    # Load the image
    image_load = tf.keras.utils.load_img(uploaded_file, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = tf.expand_dims(img_arr, axis=0)

    # Make predictions
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)

    # Get prediction and confidence
    category = data_cat[np.argmax(score)]
    confidence = np.max(score) * 100

    # Display the image at a smaller size and center it
    st.markdown("<p style='text-align: center;'>Uploaded Image Preview:</p>", unsafe_allow_html=True)
    st.image(image_load, caption='Uploaded Image', width=200, use_column_width=False)

    # Show prediction results with larger text size
    st.markdown(f"<h2 style='text-align: center;'>Prediction: {category}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;'>Confidence: <b>{confidence:.2f}%</b></p>", unsafe_allow_html=True)

    # Confidence Meter
    st.write("### Confidence Meter:")
    fig, ax = plt.subplots(figsize=(4, 0.5))  # Slimmer and smaller meter

    # Create the meter bar with custom aesthetics
    ax.barh([0], confidence, color="green" if confidence > 50 else "red", height=0.2)
    ax.set_xlim([0, 100])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([])
    ax.set_xlabel('Confidence (%)', fontsize=8)

    # Remove frame and make it cleaner
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')

    # Display the meter
    st.pyplot(fig)

    # Confidence feedback
    if confidence < 50:
        st.warning("Confidence is low. The model is unsure about this classification.")
    else:
        st.success(f"The model is {confidence:.2f}% confident in this prediction.")

else:
    st.info("Please upload an image to classify.")
