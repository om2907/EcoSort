import tensorflow as tf
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import time

# Load the trained model
model = load_model(r"C:\Users\Om Chaudhari\OneDrive\Desktop\Project SEM V\Image_classify.keras")

# Define categories
data_cat = ['Organic', 'Recyclable']

# Define image dimensions based on the model's expected input shape
img_height = 180  # Height defined in the model
img_width = 180   # Width defined in the model

# Set header and description
st.title("EcoSort AI: Waste Classification")
st.write("Using your webcam or uploading an image, we will classify waste as either Organic or Recyclable!")

# Start video capture
def start_video():
    # Set up OpenCV video capture
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera
    return cap

# Function to make predictions on a frame
def predict_frame(image):
    # Resize image to model's expected input shape
    image = image.resize((img_width, img_height))
    img_arr = tf.keras.utils.img_to_array(image)
    img_bat = tf.expand_dims(img_arr, axis=0)

    # Make predictions
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)

    # Get prediction and confidence
    category = data_cat[np.argmax(score)]
    confidence = np.max(score) * 100  # Convert to percentage
    return category, confidence

def run_app():
    col3, col4 = st.columns(2)
    with col3:
        cap = start_video()
        stframe = st.empty()  # Create a placeholder for the video frame
    with col4:
        prediction_display = st.empty()  # Placeholder for prediction display

    running = True  # Flag to control the webcam loop
    
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to Image for prediction
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Make predictions on the current frame
        category, confidence = predict_frame(image)

        # Display the live webcam feed
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels='RGB', use_column_width=True)

       
        # Update the prediction and confidence display
        prediction_display.markdown(f"<p style='text-align: center;'>Confidence: <b>{confidence:.2f}%</b></p>", unsafe_allow_html=True)
        prediction_display.markdown(f"<h2 style='text-align: center;'><strong>Prediction: {category}<strong></h2>", unsafe_allow_html=True)

        # Delay to control frame rate
        time.sleep(0.1)

    cap.release()




# Upload image functionality
uploaded_file = st.file_uploader("Upload an image...", type=["jpeg", "jpg", "png"])

if uploaded_file is not None:
    # Load and display the uploaded image
    uploaded_image = Image.open(uploaded_file)

    # Create columns for layout
    col1, col2 = st.columns(2)

    # Display uploaded image in the first column
    with col1:
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    # Make predictions on the uploaded image and display in the second column
    with col2:
        category, confidence = predict_frame(uploaded_image)
        st.markdown(f"<h2 style='text-align: center;'><strong>Prediction: {category}<strong></h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>Confidence: <b>{confidence:.2f}%</b></p>", unsafe_allow_html=True)


# Run the app
if st.button("Start Webcam"):
    run_app()
else:
    st.info("Press the button above to start webcam predictions.")
