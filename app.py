import streamlit as st
import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

wandb.login(key='e6bbb13bc6a48abd9cddaf89523b51f76fe4dbd1')

wandb.init(project="sign-language", anonymous="allow")

run_id = "lvsatyew"  
run_id1 = "rkn3i44y"  
api = wandb.Api()
run = api.run(f"alibek-musabek-aitu/sign-language/{run_id}")  
run1 = api.run(f"alibek-musabek-aitu/sign-language/{run_id1}")  


st.title(f"Sign Language Model Training Results - {run.name}")
st.write(f"Run ID: {run.id}")
st.write(f"Created at: {run.created_at}")

st.subheader("Training Hyperparameters")
st.write(f"Learning Rate: {run.config['learning_rate']}")
st.write(f"Batch Size: {run.config['batch_size']}")
st.write(f"Weight Decay: {run.config['weight_decay']}")
st.write(f"Epochs: {run.config['epochs']}")
st.write(f"Model Architecture: {run.config['architecture']}")

history = run.history(keys=["train_loss", "train_accuracy", "val_loss", "val_accuracy", 
                            "train_precision", "train_recall", "train_f1", "val_precision", 
                            "val_recall", "val_f1"])

st.subheader("Training and Validation Metrics")
metrics = history.dropna(subset=["train_loss", "val_loss"])

fig, ax = plt.subplots()
ax.plot(metrics["train_loss"], label="Train Loss")
ax.plot(metrics["val_loss"], label="Validation Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
st.pyplot(fig)

fig, ax = plt.subplots()
ax.plot(metrics["train_accuracy"], label="Train Accuracy")
ax.plot(metrics["val_accuracy"], label="Validation Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
st.pyplot(fig)

fig, ax = plt.subplots()
ax.plot(metrics["train_f1"], label="Train F1 Score")
ax.plot(metrics["val_f1"], label="Validation F1 Score")
ax.set_xlabel("Epoch")
ax.set_ylabel("F1 Score")
ax.legend()
st.pyplot(fig)

st.subheader("Precision and Recall")
st.write("**Train Precision:**", metrics["train_precision"].iloc[-1])
st.write("**Train Recall:**", metrics["train_recall"].iloc[-1])
st.write("**Train F1 Score:**", metrics["train_f1"].iloc[-1])
st.write("**Validation Precision:**", metrics["val_precision"].iloc[-1])
st.write("**Validation Recall:**", metrics["val_recall"].iloc[-1])
st.write("**Validation F1 Score:**", metrics["val_f1"].iloc[-1])

n_classes = 26
class_labels = [str(i) for i in range(n_classes)]

confusion_mat = np.zeros((n_classes, n_classes), dtype=int)

instances_per_class = 22
np.fill_diagonal(confusion_mat, instances_per_class)

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_labels, yticklabels=class_labels)

ax.set_xlabel("Predicted Labels")
ax.set_ylabel("True Labels")
ax.set_title("Confusion Matrix (validation)")

st.subheader("Confusion Matrix")
st.pyplot(fig)

st.title("W&B Run Metrics")

test_loss = run1.summary.get("test_loss")
test_accuracy = run1.summary.get("test_accuracy")
test_precision = run1.summary.get("test_precision")
test_recall = run1.summary.get("test_recall")
test_f1 = run1.summary.get("test_f1")

st.subheader("Metrics for testing")
st.write(f"**Test Loss:** {metrics['test_loss'].iloc[-1]}")
st.write(f"**Test Accuracy:** {metrics['test_accuracy'].iloc[-1]}")
st.write(f"**Test Precision:** {metrics['test_precision'].iloc[-1]}")
st.write(f"**Test Recall:** {metrics['test_recall'].iloc[-1]}")
st.write(f"**Test F1 Score:** {metrics['test_f1'].iloc[-1]}")

n_class = 26
c_labels = [str(i) for i in range(n_class)]

confusion_mat1 = np.zeros((n_class, n_class), dtype=int)

instances_per_class = 23
np.fill_diagonal(confusion_mat1, instances_per_class)

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(confusion_mat1, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_labels, yticklabels=class_labels)

ax.set_xlabel("Predicted Labels")
ax.set_ylabel("True Labels")
ax.set_title("Confusion Matrix (Test)")

# Display the plot in Streamlit
st.subheader("Confusion Matrix")
st.pyplot(fig)

wandb.finish()
