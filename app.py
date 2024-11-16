import os
import gdown
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import streamlit as st

# Function to download the model from Google Drive
def download_model(model_url, model_filename):
    if not os.path.exists(model_filename):  # Check if the file exists
        gdown.download(model_url, model_filename, quiet=False)
    else:
        print(f"Model file {model_filename} already exists.")

# Custom CNN model definition
# Load trained model
def load_model():
    model = CustomCNN()  # Replace with your model class
    model.load_state_dict(torch.load("final_trained_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# Predict character from uploaded image
def predict_image(image, model, transform, device):
    image = transform(image).unsqueeze(0).to(device)  # Apply transformation and add batch dimension
    output = model(image)
    _, predicted = output.max(1)
    return predicted.item()

# Streamlit UI
st.title("Character Recognition with Drag-and-Drop")

# Upload image file using Streamlit's drag-and-drop file uploader
uploaded_file = st.file_uploader("Drag and drop an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image from the uploaded file
    image = Image.open(uploaded_file).convert('RGB')

    # Define image transformation (resize to match model input size)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to match the model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Convert image to tensor
    ])

    # Load the pre-trained model
    model_url = "https://drive.google.com/uc?id=1WKaRuJzaHfaybAM6ggr11q1RBYk2z8ef"  # Direct link for downloading
    model_filename = "final_trained_model.pth"
    download_model(model_url, model_filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model()
    model.to(device)

    # Show the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict the character in the image
    with st.spinner('Classifying image...'):
        predicted_class = predict_image(image, model, transform, device)
        
        # Map predicted class index to corresponding character
        class_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # Mapping of indices to characters
        predicted_char = class_names[predicted_class]
        st.write(f"Predicted character: {predicted_char}")
