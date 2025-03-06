# 🍕 Pizza Burn Detection Model (Using ANN)

🟩 Project Overview

A famous pizza franchise with outlets in over 90 countries started home delivery services, but some customers began exploiting the refund system by falsely claiming they received burnt pizzas. To address this, the franchise decided to integrate a pizza detection model into their app. Customers can upload pizza images, and the model classifies them as either burnt or good, automating the refund decision process.

🚀 Goal: Build an image classification model using an Artificial Neural Network (ANN) to detect whether a pizza is burnt or not, without using CNNs or rule-based models.

📂 Dataset Description

Training Set: Contains images of both burnt and good pizzas, organized into labeled folders.

Burnt pizza → 0

Good pizza → 1

Test Set: Contains mixed images of burnt and good pizzas for model evaluation.

📈 Approach

Data Preprocessing:

Resized images to a fixed dimension (e.g., 128x128).

Flattened images into 1D vectors.

Normalized pixel values (0-255 scaled to 0-1).

Encoded labels (burnt: 0, good: 1).

Model Architecture:

Input Layer: Number of neurons = flattened image size.

Hidden Layers: Multiple dense layers with ReLU activation.

Output Layer: Single neuron with sigmoid activation.

Training Details:

Loss Function: Binary Crossentropy

Optimizer: Adam

Evaluation Metric: Accuracy

🚀 How to Run the Project

Clone the Repository:

git clone https://github.com/Rashu0304/Pizza-Burn-Detection-Model-Using-ANN-.git
cd pizza-detection-ann

Install Required Libraries:

pip install -r requirements.txt

Train the Model:

python train.py

Test the Model:

python test.py

Evaluate Accuracy:

python evaluate.py

🧠 Results

Training Accuracy: XX%

Test Accuracy: XX%

The model successfully classifies pizza images as either burnt or good, helping automate the refund process and reduce fraudulent claims.

📚 Technologies Used

Python

TensorFlow / Keras

NumPy

OpenCV / PIL

📄 Future Improvements

Experiment with data augmentation to improve generalization.

Try transfer learning with pre-trained models (if constraints allow future iterations).

Fine-tune hyperparameters for better accuracy.

🙌 Acknowledgements

Data collected from various online sources.

Guided by project constraints to stick to ANN-based models.
