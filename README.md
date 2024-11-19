# Sign Language Model Training Results Dashboard

This project allows you to visualize the training, validation, and testing results of a sign language model, leveraging Streamlit to display the metrics and insights. The app connects to Weights & Biases (W&B) to fetch the training history and visualize key metrics such as loss, accuracy, precision, recall, F1 score, and confusion matrices.

## Project Overview

The dashboard is designed to display the following:
- Hyperparameters used during training
- Training and validation metrics over epochs
- Precision, recall, and F1 scores for both training and validation datasets
- Confusion matrices for both validation and testing datasets
- Metrics for the test dataset, including loss, accuracy, precision, recall, and F1 score

The application uses W&B to track the training process, and visualizations are created using `matplotlib` and `seaborn`.Additionally, confusion matrices for both validation and testing phases are displayed to analyze the model's performance on a multi-class classification task (26 classes).

## Requirements

The following Python packages are required to run this project:

- `torch==2.5.1`
- `torchvision==0.20.1`
- `streamlit==1.40.1`
- `matplotlib==3.8.3`
- `seaborn==0.13.2`
- `scikit-learn==1.4.1.post1`
- `Pillow==10.3.0`
- `gdown==5.2.0`
- `wandb==0.18.6`
