#🌿 Deep-Learning-Based-CNN-Approach-for-Agricultural-Plant-Disease-Classification
Deep Learning-based Convolutional Neural Network (CNN) for classifying plant leaf images 

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📌 Overview
This project implements a **Convolutional Neural Network (CNN)** to classify plant leaves into three categories:
- 🌱 **Healthy**
- 🍂 **Powdery**
- 🌾 **Rust**

The model is trained from scratch on a balanced dataset, achieving **95% accuracy** on the test set.  
It is lightweight, easy to extend to more classes, and suitable for potential **edge deployment**.

---

## 📊 Key Features
- 🖼 **Three-class classification**: Healthy, Powdery, Rust.
- ⚡ **High accuracy**: 95% test accuracy.
- 📦 **Simple & reproducible pipeline** with TensorFlow/Keras.
- 🛠 **Extensible**: Easily adapt to more plant species/diseases.
- 🚀 **Portable model** (`.keras` format) for deployment.

---

## 📂 Project Structure

├── Training & Validation.ipynb # Model training and validation
├── Testing.ipynb # Model loading and testing
├── trained_plant_disease_model.keras # Saved trained model
├── training_hist.json # Training history
└── Dataset/
├── Train/Train/{Healthy, Powdery, Rust}
├── Validation/Validation/{Healthy, Powdery, Rust}
└── Test/Test/{Healthy, Powdery, Rust}


---

## 🧠 Model Architecture
A **custom CNN** with stacked convolutional blocks:

[Conv2D → Conv2D → MaxPool2D] × 5 → Dropout → Flatten → Dense(1500) → Dropout → Dense(3, Softmax)

- **Optimizer:** Adam (lr = 0.0001)  
- **Loss:** Categorical Crossentropy  
- **Metrics:** Accuracy  
- **Epochs:** 10  
- **Input Size:** 128×128 RGB images

---

## 📦 Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/plant-disease-classification.git
cd plant-disease-classification

### 2️⃣ Install Dependencies
It's recommended to use a virtual environment or Anaconda:

pip install -r requirements.txt

requirements.txt should include:

tensorflow>=2.8
numpy
matplotlib
pandas
scikit-learn
Pillow

### 3️⃣ Prepare Dataset
Ensure your dataset follows this structure:

Dataset/
  ├─ Train/Train/{Healthy, Powdery, Rust}
  ├─ Validation/Validation/{Healthy, Powdery, Rust}
  └─ Test/Test/{Healthy, Powdery, Rust}

📌 Image size will be automatically resized to 128×128 during training.

###🚀 Running the Project
🔹 Training & Validation
1. Open Training & Validation.ipynb in Jupyter Notebook / VS Code.
2. Update dataset path in:

training_set = tf.keras.utils.image_dataset_from_directory(
    "path/to/Dataset/Train/Train",
    image_size=(128, 128),
    color_mode="rgb",
    label_mode="categorical",
    batch_size=32
)

Run all cells to:

Load dataset

Build and compile CNN

Train for 10 epochs

Save the model: trained_plant_disease_model.keras

Save training history: training_hist.json

🔹 Testing
1. Open Testing.ipynb.
2. Load the trained model:

cnn = tf.keras.models.load_model('trained_plant_disease_model.keras')

3. For batch testing:
- Evaluate on test set to get precision, recall, F1-score.

4. For single image prediction:

img = tf.keras.preprocessing.image.load_img('path/to/image.jpg', target_size=(128,128))

Get predictions using cnn.predict().

| Class                | Precision | Recall | F1-score |
| -------------------- | --------- | ------ | -------- |
| Healthy              | 0.91      | 0.96   | 0.93     |
| Powdery              | 0.94      | 0.96   | 0.95     |
| Rust                 | 1.00      | 0.92   | 0.96     |
| **Overall Accuracy** | **—**     | **—**  | **0.95** |

🔮 Future Improvements
📸 Add data augmentation for better generalization.

📱 Convert to TFLite for mobile deployment.

🌍 Expand dataset to multiple plant species and real-field conditions.
