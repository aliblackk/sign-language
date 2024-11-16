import streamlit as st
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from PIL import Image
import torch.nn as nn
import gdown
import os

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 28 * 28, 512)  # Adjust the size based on your image input
        self.fc2 = nn.Linear(512, 26)  # Output layer for 26 classes (alphabet)
        
        # New layers added between fc1 and fc2
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.batch_norm = nn.BatchNorm1d(512)
    
    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = x.view(-1, 256 * 28 * 28)  # Flatten the output for the fully connected layer
        x = self.relu(self.fc1(x))  # Apply ReLU activation on fc1 output
        x = self.batch_norm(x)  # Apply BatchNorm after fc1
        x = self.dropout(x)  # Apply Dropout
        x = self.fc2(x)  # Final output layer
        return x

import streamlit as st
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from PIL import Image
import torch.nn as nn
import gdown
import os

# Function to download the model from Google Drive
def download_model(model_url, model_filename):
    if not os.path.exists(model_filename):
        # Download the model from Google Drive
        gdown.download(model_url, model_filename, quiet=False)
    else:
        st.write(f"Model file {model_filename} already exists.")

# Load trained model
def load_model():
    model = CustomCNN()  # Replace with your model class
    model.load_state_dict(torch.load("final_trained_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# Generate confusion matrix plot
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    st.pyplot(plt)

# Evaluate the model on a dataset
def evaluate_model(model, loader):
    all_labels = []
    all_preds = []

    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            _, predicted = output.max(1)
            correct_preds += predicted.eq(target).sum().item()
            total_preds += target.size(0)

            all_labels.extend(target.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = correct_preds / total_preds
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "loss": total_loss / len(loader),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }

# Streamlit app
st.title("Sign Language Model Evaluation")

# Google Drive model URL
model_url = "https://drive.google.com/uc?id=1WKaRuJzaHfaybAM6ggr11q1RBYk2z8ef"  # Direct link for downloading
model_filename = "final_trained_model.pth"

# Download the model from Google Drive
download_model(model_url, model_filename)

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload Test Dataset", type=["pt"])

if uploaded_file is not None:
    # Load uploaded dataset
    test_dataset = torch.load(uploaded_file)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # Adjust batch size as needed

    # Load model
    model = load_model()

    # Evaluate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    results = evaluate_model(model, test_loader)

    # Display metrics
    st.subheader("Model Metrics")
    st.write(f"Loss: {results['loss']:.4f}")
    st.write(f"Accuracy: {results['accuracy']:.2%}")
    st.write(f"Precision: {results['precision']:.2%}")
    st.write(f"Recall: {results['recall']:.2%}")
    st.write(f"F1 Score: {results['f1']:.2%}")

    # Display confusion matrix
    st.subheader("Confusion Matrix")
    class_names = [str(i) for i in range(results["confusion_matrix"].shape[0])]
    plot_confusion_matrix(results["confusion_matrix"], class_names)

    # Option to test individual images
    st.subheader("Test an Image")
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Adjust to your input size
            transforms.ToTensor(),
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = output.max(1)
            st.write(f"Predicted Class: {predicted.item()}")
