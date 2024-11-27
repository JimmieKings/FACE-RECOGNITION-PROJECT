# key libbraries
import streamlit as st
import numpy as np
import os
import pickle
from keras_facenet import FaceNet
from PIL import Image
import cv2

# Loading the saved SVM model and embeddings dataset
with open('svm_classifier.pkl', 'rb') as model_file:
    svm_classifier = pickle.load(model_file)

data = np.load('celebrity_faces_embeddings.npz')
train_labels = data['train_labels']

# Loading FaceNet model for embeddings
embedder = FaceNet()

# Defining helper functions
def extract_face(image, size=(160, 160)):
    """Detect and extract a face from the given image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = image[y:y+h, x:x+w]
    face = cv2.resize(face, size)
    return face

def display_celeb_list():
    """for displaying the unique list of celebrities available in the dataset."""
    st.sidebar.markdown("### 📜 List of Available Celebrities")
    celeb_list = sorted(set(train_labels))
    for celeb in celeb_list:
        st.sidebar.write(f"- {celeb}")

def get_sample_image(label):
    """Get a sample image of the correctly identified celebrity."""
    celeb_dir = os.path.join('celeb_images', label)
    if os.path.exists(celeb_dir):
        files = os.listdir(celeb_dir)
        if files:
            return os.path.join(celeb_dir, files[0])
    return None

# Streamlit App
st.title("🎭Face Recognition App")
st.write("Welcome to the **Celebrity Face Recognition App**! Upload a photo to see if we can recognize the celebrity.")

# Sidebar
st.sidebar.header("How to Use")
st.sidebar.write("1. Upload an image of a celebrity.\n2. See if we can predict the name!\n3. Check the list of celebrities for hints. 😉")
display_celeb_list()

# Upload image
uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Detect and process the face
    face = extract_face(image_np)
    if face is not None:
        # Generate embedding and predict
        face_embedding = embedder.embeddings(np.expand_dims(face, axis=0))[0]
        prediction = svm_classifier.predict([face_embedding])[0]
        probability = svm_classifier.predict_proba([face_embedding]).max()

        # Display prediction and confidence
        st.success(f"🎉 Prediction: {prediction}")
        st.write(f"Confidence: {probability:.2f}")

        # Display sample image of the predicted celebrity
        sample_image_path = get_sample_image(prediction)
        if sample_image_path:
            sample_image = Image.open(sample_image_path)
            st.image(sample_image, caption=f"Identified as: {prediction}", use_column_width=True)
        else:
            st.warning("No sample image available for this celebrity.")

    else:
        st.error("No face detected in the uploaded image. Please try again with a clearer image!")
