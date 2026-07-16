# 🌍 Country Classifier from Latitude & Longitude using PyTorch

A deep learning project that predicts the **country or ocean region** from a pair of **latitude and longitude coordinates** using a PyTorch neural network.

---

## 📌 Project Overview

Most location-based systems determine a country using GIS (Geographic Information System) algorithms such as Point-in-Polygon searches.

The objective of this project was to explore an alternative approach by training a neural network to learn the relationship between geographic coordinates and countries/oceans.

Instead of performing geometric polygon searches during prediction, the trained model predicts the country directly from latitude and longitude.

---

# 🚀 Features

* Predicts **226 classes** (Countries + Ocean Regions)
* Built completely using **PyTorch**
* Automatic preprocessing pipeline
* GPU support (CUDA)
* Checkpoint saving and loading
* Resume training from checkpoints
* Model evaluation
* Precision, Recall and F1 Score
* Standalone prediction script
* Exportable model for deployment

---

# 🛠 Technologies Used

* Python
* PyTorch
* Pandas
* NumPy
* Scikit-Learn
* GeoPandas
* Shapely
* Joblib

---

# 📂 Dataset Generation

The dataset was generated manually instead of using an existing dataset.

### Country Coordinates

Country boundaries were obtained from **Natural Earth**.

Random latitude and longitude points were generated inside each country's polygon using rejection sampling.

Each generated point contains:

* Latitude
* Longitude
* Country Name

---

### Ocean Coordinates

Ocean polygons were obtained from GOAS (Global Oceans and Seas).

Random points were generated inside the following ocean regions:

* Arctic Ocean
* Atlantic Ocean
* Pacific Ocean
* Indian Ocean
* Southern Ocean
* Mediterranean Region
* Baltic Sea
* South China and Easter Archipelagic Seas
* North Atlantic Ocean
* North Pacific Ocean

---

### Final Dataset

* Dataset Size : **1,380,000 samples**
* Features : **Latitude, Longitude**
* Output Classes : **226**

---

# 📊 Data Preprocessing Pipeline

The preprocessing pipeline consists of:

1. Load dataset
2. Separate features and labels
3. Label Encoding
4. Train-Test Split
5. Standard Scaling
6. Convert to PyTorch Tensors
7. TensorDataset Creation
8. DataLoader Creation

The same StandardScaler and LabelEncoder are exported for inference.

---

# 🧠 Neural Network Architecture

```
Input (Latitude, Longitude)

        │

Linear (2 → 128)

        │

ReLU

        │

Linear (128 → 128)

        │

ReLU

        │

Linear (128 → 226)

        │

Predicted Country/Ocean
```

---

# ⚙ Training Configuration

Optimizer

* Adam

Learning Rate

* 0.001

Loss Function

* CrossEntropyLoss

Batch Size

* 128

Device

* CUDA (if available)
* CPU (fallback)

---

# 📈 Training Pipeline

The training script performs:

* Forward Pass
* Loss Calculation
* Backpropagation
* Weight Update
* Average Loss Calculation
* Checkpoint Saving
* Training Log Generation

Training can be resumed automatically from the saved checkpoint.

---

# 🧪 Evaluation

The evaluation script reports:

* Test Loss
* Test Accuracy
* Precision
* Recall
* F1 Score
* Classification Report

Evaluation is performed using:

* `model.eval()`
* `torch.no_grad()`

---

# 📊 Results

## Training

Average Training Loss after training 150 epohs :

**0.1633**

---

## Testing

Test Loss

**0.1611**

Test Accuracy

**93.98%**

The classification report also includes:

* Precision
* Recall
* F1 Score

for every class along with Macro Average and Weighted Average.

---

# 📦 Model Export

After training, the model can be exported using:

```
python save_model.py
```

Generated Files

* country_classifier_model.pth
* scaler.pkl
* label_encoder.pkl

These files are sufficient for deployment and inference.

---

# 🔍 Prediction

A standalone prediction script allows inference without requiring the training dataset.

Run

```
python predict.py
```

Example

```
Enter Latitude : 12.9716
Enter Longitude: 77.5946

Prediction

Country    : India
Confidence : 99.34%
```

---

# 📁 Project Structure

```
CountryClassifierUsingCoordinates/

│
├── CountryClassifierTrain.py
├── train.py
├── test.py
├── save_model.py
├── predict.py
│
├── country_coordinate_data.csv
├── country_classifier_checkpoint.pth
├── country_classifier_model.pth
├── scaler.pkl
├── label_encoder.pkl
├── training_log.txt
│
└── README.md
```

---

# ▶ How to Run

Train

```
python CountryClassifierTrain.py
```

Evaluate

```
python test.py
```

Export Model

```
python save_model.py
```

Predict

```
python predict.py
```

---

# 💡 Future Improvements

* Learning Rate Scheduler
* Early Stopping
* Confusion Matrix Visualization
* ONNX Export
* TensorRT Optimization
* REST API using FastAPI
* Batch Prediction Support
* Docker Deployment

---

# 📚 What I Learned

This project helped me gain practical experience with:

* Object-Oriented Programming
* Data Preprocessing
* Neural Networks
* PyTorch
* GPU Training
* Model Checkpointing
* Saving and Loading Models
* Classification Metrics
* Model Deployment
* End-to-End Machine Learning Pipeline

---

# 📄 License

This project is open-source and intended for learning and educational purposes.
