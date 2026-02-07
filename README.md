# 🌱 Plant Disease Detection Pipeline (Generic Multi-Plant System)

## 📌 Project Overview

This project is a **flexible and reusable pipeline** for detecting plant diseases using both **traditional machine learning models** (RandomForest, XGBoost) and **segmentation-based preprocessing**.

The system is **generic**:

* You can input *any plant species* (e.g., apple, tomato, wheat).
* The pipeline automatically retrieves or processes its disease dataset.
* It trains a **specialized model** for that plant.
* You can compare **training with segmentation vs. without segmentation** to evaluate performance improvements.

The aim is to combine **automation**, **flexibility**, and **scalability** for agricultural AI applications.

---

## 🚀 Key Features

* **Multi-Plant Support:** Works with datasets from different plants and diseases without rewriting code.
* **Segmentation Option:** Train models on raw images or segmented leaf/disease areas.
* **Ensemble Learning:** RandomForest & XGBoost for fast, interpretable, and efficient classification.
* **Hyperparameter Tuning:** Automated search via `RandomizedSearchCV`.
* **Comprehensive Evaluation:** Accuracy, precision, recall, F1-score, and confusion matrix.
* **Feature Importance:** Understand which features drive predictions.
* **Modular Structure:** Clear separation between data loading, preprocessing, training, and evaluation.

This pipeline detects plant diseases using two approaches:
1. Baseline: Direct ML classification (RandomForest/XGBoost)
2. Segmentation-enhanced: Preprocess images to isolate plant regions before classification

The system is plant-agnostic – works with apple, tomato, wheat, etc. by loading appropriate datasets.

---

## 📂 Project Structure

```
plant-disease-pipeline/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── sampled-plants-diseases/             # Original datasets
│
├── notebooks/
│   ├── EDA.ipynb
│   └── Evaluation.ipynb
│
├── models/
│   ├── without_segmentation/
│   └── with_segmentation/
│
└── deployment/
    ├── app.py
    ├── model_loader.py
    └── demo_streamlit.py
```

---

## 🏃 Running the Pipeline

1. **Set Dataset Path** in notebooks.
2. **Run Training Without Segmentation**:

   ```bash
   python notebooks/plant_disease_pipeline.py --plant_name apple --model_type random_forest
   ```
3. **Run Training With Segmentation**:

   ```bash
   python  notebooks/plant_disease_pipeline.py --plant_name apple --model_type xgboost --with_segmentation --segmentation_method otsu

   ```
4. **Evaluate and Compare Results**:

   ```bash
   python notebooks/Evaluation.ipynb
   ```

---

## 📈 Evaluation Metrics

The pipeline reports:

* Accuracy
* Precision, Recall, F1-score
* Confusion Matrix
* Feature Importance Plots
* Segmentation vs Non-Segmentation Performance Comparison


