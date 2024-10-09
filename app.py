import tensorflow as tf
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model/Image_classify.keras')

# Define categories
data_cat = ['O', 'R']  # Assuming 'O' = Organic and 'R' = Recyclable, adjust accordingly

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

    # Display the image at a smaller size
    st.image(image_load, caption='Uploaded Image', use_column_width=True)

    # Show prediction results
    category = data_cat[np.argmax(score)]  # 'O' for Organic, 'R' for Recyclable
    confidence = np.max(score) * 100
    st.subheader(f"Prediction: {category}")
    st.write(f"The waste in the image is predicted to be **{category}** with a confidence of **{confidence:.2f}%**.")

    # Feedback for confidence levels
    if confidence < 50:
        st.warning("Confidence is low. The model is unsure about this classification.")
    else:
        st.success(f"The model is {confidence:.2f}% confident in this prediction.")

else:
    st.info("Please upload an image to classify.")
