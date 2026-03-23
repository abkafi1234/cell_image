# 🔬 Robust Malaria Parasite Detection Pipeline

A complete, end-to-end Machine Learning and Computer Vision pipeline for detecting malaria parasites in Giemsa-stained thin blood smear images.

This project tackles the most difficult challenges in digital pathology (stain variance, illumination differences, and cell artifacts) using dynamically adaptive computer vision techniques. It includes a rigorous statistically-validated training pipeline and an interactive Streamlit inference dashboard for clinical visualization.

---

## 🌟 Key Features

* **Stain-Invariant Feature Extraction:** Utilizes the Gray World algorithm to mathematically neutralize color differences caused by varying Giemsa stain concentrations.
* **Adaptive Local Thresholding:** Replaces failure-prone global thresholds (like Otsu) with local adaptive thresholding on the Green Channel to find microscopic parasites even in heavily shadowed cells.
* **Bayesian Hyperparameter Optimization:** Uses probabilistic searching (`skopt`) to dynamically find the optimal parameters for SVM, Random Forest, and HistGradientBoosting models without data leakage.
* **Clinical-Grade Statistical Validation:** Evaluates models using 20-Fold Stratified Cross-Validation, reporting 95% Confidence Intervals and One-Sample T-Tests (p-values) to mathematically prove stability over random chance.
* **Interactive Inference App:** A Streamlit frontend that not only predicts infection but visually charts the algorithm's step-by-step mathematical reasoning.

---

## 🛠️ Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/malaria-detection.git
cd malaria-detection
pip install -r requirements.txt
```

**Required Libraries:**
`numpy`, `opencv-python`, `scikit-image`, `scikit-learn`, `scikit-optimize`, `scipy`, `matplotlib`, `streamlit`, `joblib`

---

## 🧠 Methodology: The Computer Vision Pipeline

Because parasites vary wildly in size, color, and location, raw deep learning often fails or overfits on stain color. This project uses a handcrafted, interpretable feature extraction pipeline:

1. **Color Normalization:** Scales the B, G, and R channels based on the image's global mean. This forces pale pink cells and dark purple cells into the same mathematical color space.
2. **Cell Masking:** Isolates the cell from the black background.
3. **Adaptive Spot Detection:** Scans the high-contrast Green channel using a 21x21 pixel neighborhood (`cv2.adaptiveThreshold`). It flags pixels only if they are significantly darker than their immediate surroundings, ignoring soft shadows.
4. **Morphological Cleaning & Saturation Check:** Removes 1-pixel dust artifacts and calculates the HSV saturation of the largest spot to differentiate between a dark shadow and a dense clump of parasitic DNA.

---

## 🚀 Usage 1: Training & Statistical Validation

The training script automatically extracts features, runs Bayesian optimization, and evaluates the models using 20-Fold CV.

### Running the Training Script

```python
# train.py
from model_pipeline import run_evaluation

# Point this to your dataset directory containing 'Uninfected' and 'Parasitized' folders
DATASET_PATH = "./cell_images/" 

if __name__ == "__main__":
    run_evaluation(DATASET_PATH)
```

### Under the Hood: The Evaluation Loop

The pipeline uses nested cross-validation to prevent data leakage during scaling and hyperparameter tuning:

```python
# Inside run_evaluation()
bayes_search = BayesSearchCV(
    estimator=Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier())]),
    search_spaces={'classifier__n_estimators': Integer(50, 300), 'classifier__max_depth': Integer(3, 20)},
    n_iter=15, 
    cv=3, # Inner CV for tuning
    n_jobs=-1
)
bayes_search.fit(X, y)

# Outer CV for Clinical Stability
cv_results = cross_validate(bayes_search.best_estimator_, X, y, cv=20, scoring='accuracy')
```

**Output Example:**

```text
Random Forest:
  Accuracy : 0.9412 ± 0.0124
  95% CI   : [0.9354, 0.9470]
  p-value  : 1.2e-15 (Stable & Significant)
```

---

## 🔍 Usage 2: Streamlit Inference App

To make the model accessible, the project includes an interactive web app. It takes a raw image, runs it through the saved optimized model, and plots the exact steps the computer vision algorithm took to reach its conclusion.

### Running the App

```bash
streamlit run app.py
```

### How the App Works (`app.py` snippet)

When an image is uploaded, the app extracts the mathematical features and feeds them to the `.pkl` model saved during training:

```python
import streamlit as st
import joblib

# Load the Bayesian-Optimized model
model = joblib.load('best_malaria_model.pkl')

# Process Uploaded Image
features = extract_features_from_array(img_bgr)
features_2d = features.reshape(1, -1)

# Predict
prediction = model.predict(features_2d)[0]

if prediction == 1:
    st.error("🚨 PREDICTION: PARASITIZED")
else:
    st.success("✅ PREDICTION: UNINFECTED (HEALTHY)")
```

*The app also features a built-in Matplotlib visualizer showing the Original Image -> Normalized Image -> Green Channel -> Raw Threshold -> Cleaned Spots -> Final Contours.*

---

## 📁 Project Structure

```text
├── cell_images/               # Dataset (Not included in repo)
├── app.py                     # Streamlit web application
├── test.ipynb                 # Normal Training 20-Fold CV logic
├── bayesian.ipynb             # Bayesian Opt & 20-Fold CV logic
├── best_malaria_model.pkl     # Saved weights from the best model
└── README.md                  # Project documentation
```
