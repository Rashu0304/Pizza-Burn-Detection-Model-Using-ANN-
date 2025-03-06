# 🍕 Pizza Image Classification: Burnt vs Fresh Pizza

📌 Objective:

This project aims to build a machine learning model capable of classifying pizza images into two categories: "Burnt Pizza" and "Fresh Pizza". The model is trained on labeled image data, leveraging a neural network to learn visual features and accurately predict the state of a pizza from unseen images.

🛠️ Tools & Libraries Used:

Python

TensorFlow/Keras

scikit-learn

OpenCV

Matplotlib & Seaborn

Google Colab (for execution)

tqdm (for progress tracking)

📂 Dataset Setup:

The training and testing datasets are stored in RAR files on Google Drive. The project includes steps to mount Google Drive, install necessary packages, and extract the dataset for model training.

🧠 Model Architecture:

A simple yet powerful feedforward neural network was used:

Input Layer: Flattened image data (128x128x3)

Hidden Layers:

Dense (1024 units, ReLU activation)

Dropout (40%)

Dense (512 units, ReLU activation)

Output Layer: Dense (2 units, Softmax activation)

The model is compiled with the Adam optimizer and trained using sparse categorical cross-entropy loss.

📊 Model Training:

Train-Test Split: 80% training, 20% testing

Epochs: 50

Batch Size: 32

Training accuracy and loss, as well as validation metrics, are tracked across epochs to monitor performance.

🧩 Evaluation:

Accuracy: Achieved around 88% test accuracy

Loss: Low test loss indicating a well-generalized model

🟢 Classification Report & Confusion Matrix:

Comprehensive evaluation using scikit-learn metrics to analyze precision, recall, and F1 scores for both classes.

📈 Visualization:

Training vs Validation Accuracy Plot

Training vs Validation Loss Plot

Confusion Matrix Heatmap

🔮 Prediction:

The model makes predictions on test images, with outputs converted to class labels.

💾 Model Saving:

The trained model is saved as an H5 file for future inference.

🚀 Key Takeaways:

Image Preprocessing: Resizing, normalization, and flattening improved model performance.

Overfitting Prevention: Dropout layers helped prevent overfitting.

Real-World Applicability: The project demonstrates how deep learning can solve practical image classification problems, with potential extensions to broader food quality detection systems.

📘 How to Run the Project:

Clone the repository and upload the notebook to Google Colab.

Mount Google Drive and upload the dataset.

Install required libraries (rarfile, TensorFlow, etc.).

Train the model or load the saved model for inference.

Visualize results and analyze the model's performance.

📂 Project Structure:

├── Dataset
│   ├── train (Burnt & Fresh Pizza images)
│   └── test (Burnt & Fresh Pizza images)
├── pizza_classifier.ipynb
└── pizza_classifier_model.h5

