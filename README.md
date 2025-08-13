# Deep Learning-Based CNN for Agricultural Plant Disease Classification

This repository contains the implementation of a **Convolutional Neural Network (CNN)** for the classification of plant leaf diseases. The model is trained to identify **Healthy**, **Powdery mildew**, and **Rust** leaves from RGB images, aiming to assist in early disease detection and yield protection in agriculture.

## 📜 Project Overview

Plant diseases cause significant crop losses worldwide. Manual diagnosis is time-consuming and error-prone, while automated image-based methods can provide faster and more accurate results.  
This project implements a **custom sequential CNN** trained from scratch to classify leaf images into three categories:

- **Healthy**
- **Powdery mildew**
- **Rust**

The trained model achieved **95% accuracy** on a balanced test set and is designed to be extendable to other plant species and diseases.

---

## 📂 Dataset

**Dataset**: https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset 

The dataset is organized into **Train**, **Validation**, and **Test** sets, with class-specific subfolders.

| Subset     | Healthy | Powdery | Rust | Total |
|------------|---------|---------|------|-------|
| Training   | 458     | 430     | 434  | 1322  |
| Validation | 20      | 20      | 20   | 60    |
| Test       | 50      | 50      | 50   | 150   |

- **Image size:** 128×128 pixels (resized on load)  
- **Color mode:** RGB  
- **Labels:** One-hot encoded  

Example structure:
```plaintext
Dataset/
  ├─ Train/Train/Healthy
  ├─ Train/Train/Powdery
  ├─ Train/Train/Rust
  ├─ Validation/Validation/Healthy
  ├─ Validation/Validation/Powdery
  ├─ Validation/Validation/Rust
  ├─ Test/Test/Healthy
  ├─ Test/Test/Powdery
  └─ Test/Test/Rust
  ```

## 🏗 Model Architecture
The model is a 5-block CNN with the following structure:

```cnn = tf.keras.models.Sequential([
    # Block 1
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Block 2
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Block 3
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Block 4
    tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(256, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Block 5
    tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(512, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1500, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(3, activation='softmax')
])
```

## ⚙️ Training Configuration

```cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = cnn.fit(
    training_set,
    validation_data=validation_set,
    epochs=10
)
cnn.save('trained_plant_disease_model.keras')
```

- Loss function: Categorical Cross-Entropy
- Optimizer: Adam (lr=0.0001)
- Batch size: 32
- Epochs: 10

## 📊 Results
Test set performance:

| Class       | Precision | Recall | F1-score |
| ----------- | --------- | ------ | -------- |
| Healthy     | 0.91      | 0.96   | 0.93     |
| Powdery     | 0.94      | 0.96   | 0.95     |
| Rust        | 1.00      | 0.92   | 0.96     |
| **Overall** | -         | -      | **0.95** |

Confusion Matrix:
-Most errors occurred between Powdery and Healthy leaves with subtle symptom overlap.
-Rust predictions were precise but missed a few cases with mild symptoms.

## 🚀 How to Run

1. Clone the repository

```git clone https://github.com/yourusername/plant-disease-cnn.git
cd plant-disease-cnn
```

2. Prepare dataset in the directory structure described above.

3. Install dependencies

```pip install -r requirements.tx
t```

4. Train the model

```python Training & Validation.py
```

5. Test the model

```python Testing.py
```
## 📈 Future Improvements

- Add data augmentation for better generalization.

- Increase dataset diversity (lighting, background, species).

- Experiment with transfer learning (MobileNetV3, EfficientNet).

- Deploy lightweight version to mobile or edge devices.





