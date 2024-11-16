import gdown
import zipfile
import os
import shutil
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import streamlit as st

# Function to download the test dataset from Google Drive
def download_test_dataset(test_url, test_filename):
    if not os.path.exists(test_filename):
        # Download the test dataset from Google Drive
        gdown.download(test_url, test_filename, quiet=False)
    else:
        print(f"Test dataset file {test_filename} already exists.")

def extract_zip(uploaded_file):
    # Create a temporary directory to extract the contents of the zip file
    with tempfile.TemporaryDirectory() as tmpdirname:
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(tmpdirname)
        
        # Use a local directory to store the extracted contents
        persistent_dir = "./extracted_images2"  # Directory in the current working directory
        shutil.move(tmpdirname, persistent_dir)  # Move the extracted content to a permanent location

        # Check if files are present in the extracted directory
        extracted_files = os.listdir(persistent_dir)
        
        # Return the correct directory
        return persistent_dir  

# Custom dataset to load images from extracted folder structure
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Iterate through the folders and collect images and their labels
        for label, folder in enumerate(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                print(f"Found {len(image_files)} images in folder {folder}")
                for image_name in image_files:
                    image_path = os.path.join(folder_path, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Open and transform the image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label


# Function to download the model from Google Drive
def download_model(model_url, model_filename):
    if not os.path.exists(model_filename):
        # Download the model from Google Drive
        gdown.download(model_url, model_filename, quiet=False)
    else:
        print(f"Model file {model_filename} already exists.")

# Custom CNN model definition
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
def evaluate_model(model, loader, device):
    all_labels = []
    all_preds = []

    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)  # Ensure data is on correct device
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            _, predicted = output.max(1)
            correct_preds += predicted.eq(target).sum().item()
            total_preds += target.size(0)

            all_labels.extend(target.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Avoid division by zero
    accuracy = correct_preds / total_preds if total_preds > 0 else 0.0
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

# Streamlit UI
st.title("Image Classification Model Evaluation")

# Upload zip file containing the test dataset
uploaded_file = st.file_uploader("Upload Test Dataset", type=["zip"])

if uploaded_file is not None:
    # Extract the dataset
    temp_dir = extract_zip(uploaded_file)
    st.write(f"Files extracted to: {temp_dir}")

    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images if necessary
        transforms.ToTensor(),
    ])

    dataset_dir = os.path.join(temp_dir, 'test')  # Assuming 'test' is the subfolder name
    dataset = ImageFolderDataset(root_dir=dataset_dir, transform=transform)
    st.write(f"Test dataset contains {len(dataset)} images.")

    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Load the model
    model_url = "https://drive.google.com/uc?id=1WKaRuJzaHfaybAM6ggr11q1RBYk2z8ef"  # Direct link for downloading
    model_filename = "final_trained_model.pth"
    download_model(model_url, model_filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model()
    model.to(device)

    # Evaluate the model
    evaluation_results = evaluate_model(model, test_loader, device)

    # Display evaluation results
    st.write(f"Loss: {evaluation_results['loss']:.4f}")
    st.write(f"Accuracy: {evaluation_results['accuracy']:.4f}")
    st.write(f"Precision: {evaluation_results['precision']:.4f}")
    st.write(f"Recall: {evaluation_results['recall']:.4f}")
    st.write(f"F1 Score: {evaluation_results['f1']:.4f}")

    # Display confusion matrix
    plot_confusion_matrix(evaluation_results['confusion_matrix'], class_names=[str(i) for i in range(26)])
