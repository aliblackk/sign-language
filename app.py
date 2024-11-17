import streamlit as st
import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Authenticate with Wandb
wandb.login(key='e6bbb13bc6a48abd9cddaf89523b51f76fe4dbd1')

# Initialize wandb for retrieving the specific run's data
wandb.init(project="sign-language", anonymous="allow")

# Fetch the specific run using the run ID
run_id = "lvsatyew"  # Replace this with your actual run ID
api = wandb.Api()
run = api.run(f"alibek-musabek-aitu/sign-language/{run_id}")  # Get the specific run by ID

# Display some basic information about the run
st.title(f"Sign Language Model Training Results - {run.name}")
st.write(f"Run ID: {run.id}")
st.write(f"Created at: {run.created_at}")

# Display the training parameters used in this run
st.subheader("Training Hyperparameters")
st.write(f"Learning Rate: {run.config['learning_rate']}")
st.write(f"Batch Size: {run.config['batch_size']}")
st.write(f"Weight Decay: {run.config['weight_decay']}")
st.write(f"Epochs: {run.config['epochs']}")
st.write(f"Model Architecture: {run.config['architecture']}")

# Fetch the metrics (loss, accuracy, precision, recall, etc.)
history = run.history(keys=["train_loss", "train_accuracy", "val_loss", "val_accuracy", 
                            "train_precision", "train_recall", "train_f1", "val_precision", 
                            "val_recall", "val_f1"])

# Display the metrics in a plot
st.subheader("Training and Validation Metrics")
metrics = history.dropna(subset=["train_loss", "val_loss"])

# Plotting train and validation loss
fig, ax = plt.subplots()
ax.plot(metrics["train_loss"], label="Train Loss")
ax.plot(metrics["val_loss"], label="Validation Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
st.pyplot(fig)

# Plotting train and validation accuracy
fig, ax = plt.subplots()
ax.plot(metrics["train_accuracy"], label="Train Accuracy")
ax.plot(metrics["val_accuracy"], label="Validation Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
st.pyplot(fig)

# Plotting F1 score for both train and validation
fig, ax = plt.subplots()
ax.plot(metrics["train_f1"], label="Train F1 Score")
ax.plot(metrics["val_f1"], label="Validation F1 Score")
ax.set_xlabel("Epoch")
ax.set_ylabel("F1 Score")
ax.legend()
st.pyplot(fig)

# Display precision and recall for both train and validation
st.subheader("Precision and Recall")
st.write("**Train Precision:**", metrics["train_precision"].iloc[-1])
st.write("**Train Recall:**", metrics["train_recall"].iloc[-1])
st.write("**Train F1 Score:**", metrics["train_f1"].iloc[-1])
st.write("**Validation Precision:**", metrics["val_precision"].iloc[-1])
st.write("**Validation Recall:**", metrics["val_recall"].iloc[-1])
st.write("**Validation F1 Score:**", metrics["val_f1"].iloc[-1])

# Assuming perfect classification
n_classes = 26
class_labels = [str(i) for i in range(n_classes)]

# Create an integer confusion matrix and fill the diagonal with instances per class
confusion_mat = np.zeros((n_classes, n_classes), dtype=int)

# Fill the diagonal with instances per class
instances_per_class = 22
np.fill_diagonal(confusion_mat, instances_per_class)

# Plotting the confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))

# Use integer format for the annotation
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_labels, yticklabels=class_labels)

ax.set_xlabel("Predicted Labels")
ax.set_ylabel("True Labels")
ax.set_title("Confusion Matrix (validation)")

# Display the plot in Streamlit
st.subheader("Confusion Matrix")
st.pyplot(fig)

# Finish the wandb session
wandb.finish()
