import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Function to load the model from the saved file
def load_model(model_path='my_model.h5'):
    """Loads a pre-trained model from the specified path."""
    model = tf.keras.models.load_model(model_path, 
                                       custom_objects={
                                           'Precision': tf.keras.metrics.Precision, 
                                           'Recall': tf.keras.metrics.Recall
                                       })
    st.success("Model loaded successfully.")
    return model

# Function to preprocess the input image
def preprocess_image(img_path, target_size=(299, 299)):
    """Preprocesses an image to the required format for prediction."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict the class probabilities for the input image
def predict_image(model, img_array):
    """Predicts class probabilities for the input image using the loaded model."""
    predictions = model.predict(img_array)
    return predictions[0]  # Return the first (and only) set of predictions

# Function to plot a histogram of the class probabilities
def plot_histogram(probabilities, class_names=None):
    """Plots a histogram of the class probabilities."""
    if class_names is None:
        class_names = [f"Class {i+1}" for i in range(len(probabilities))]  # Default class names
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(class_names, probabilities, color='skyblue')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Probability')
    ax.set_title('Class Probability Distribution')
    ax.set_ylim(0, 1)  # Since probabilities are in the range [0, 1]
    st.pyplot(fig)

# Streamlit application
st.title("Image Classification with Deep Learning")
st.write("Upload an image to classify it into one of the pre-defined categories.")

# Step 1: Load the model
model_path = 'Brain_tumour.h5'  # Path to the saved model
model = load_model(model_path)

# Step 2: Upload an image file
uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    # Step 3: Preprocess the input image
    img_array = preprocess_image(uploaded_file)
    
    # Step 4: Predict the class probabilities
    probabilities = predict_image(model, img_array)
    
    # Step 5: Display predictions
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']  # Replace with actual class names if known
    
    st.subheader("Class Probabilities")
    for i, prob in enumerate(probabilities):
        st.write(f"{class_names[i]}: {prob * 100:.2f}%")
    
    # Step 6: Plot the histogram of probabilities
    plot_histogram(probabilities, class_names)
