# Hospital Readmission Risk Predictor

![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

An end-to-end machine learning pipeline that predicts 30-day hospital readmission risk for diabetic patients, helping hospitals reduce penalties, allocate resources efficiently, and improve patient outcomes.

---

## Problem Statement

Hospital readmissions within 30 days cost the U.S. healthcare system billions of dollars annually and are penalized by Medicare/Medicaid. Early identification of high-risk patients allows hospitals to intervene proactively, reducing readmissions and improving care quality.

---

## Key Features

- End-to-end ML pipeline from raw data to live dashboard
- Trained on 71,515 real patient records from 130 U.S. hospitals
- XGBoost model with ROC-AUC of 0.6254
- SMOTE for handling severe class imbalance (10:1 ratio)
- SHAP explainability to understand why each prediction is made
- Out-of-distribution detection for responsible AI practices
- Interactive Streamlit dashboard with clinical recommendations

---

## Dataset

**Diabetes 130-US Hospitals Dataset** (UCI ML Repository)

- 101,766 patient encounters cleaned to 71,515 unique patients
- 50 original features engineered to 36 meaningful features
- Target: 30-day readmission (less than 30 days = 1, otherwise = 0)
- Class imbalance: only 8.8% are critical readmissions

[Dataset Link](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

---

## Project Structure
```
healthcare-readmission-pipeline/
│
├── data/
│   ├── diabetic_data.csv                  
│   ├── IDS_mapping.csv                    
│   ├── diabetic_data_cleaned.csv          
│   └── diabetic_data_features.csv         
│
├── models/
│   ├── xgboost_model.pkl                  
│   └── readmission_pipeline.pkl           
│
├── plots/
│   ├── target_distribution.png
│   ├── age_distribution.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   ├── roc_curves.png
│   ├── shap_feature_importance.png
│   └── shap_summary.png
│
├── module1_data_acquisition.ipynb
├── module2_data_cleaning.ipynb
├── module3_eda.ipynb
├── module4_feature_engineering.ipynb
├── module5_model_building.ipynb
├── module6_evaluation_explainability.ipynb
├── module7_pipeline.ipynb
├── app.py                                 
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Methodology

### 1. Data Cleaning
- Replaced hidden missing values represented as "?" with NaN
- Removed duplicate patient encounters and kept first visit only
- Dropped columns with more than 80% missing values
- Decoded ICD-9 diagnosis codes and admission type mappings

### 2. Feature Engineering
Seven new clinically meaningful features were created:

| Feature | Description |
|---------|-------------|
| hospital_utilization_score | Total prior hospital visits |
| num_med_changed | Number of medications changed |
| num_med_prescribed | Active medications count |
| treatment_intensity | Combined clinical complexity score |
| patient_complexity | Diagnoses multiplied by hospital days |
| primary_diag_diabetes | Diabetes as primary diagnosis flag |
| age_risk_group | Clinical age risk categories |

### 3. Model Building
- Applied SMOTE to handle 10:1 class imbalance
- Trained four models: Logistic Regression, Decision Tree, Random Forest, XGBoost
- XGBoost achieved the best ROC-AUC of 0.6254
- Decision threshold tuned to 0.2 for higher recall in clinical setting

### 4. Model Explainability
SHAP values were used to identify the top readmission drivers:

| Rank | Feature | Insight |
|------|---------|---------|
| 1 | change | Medication changes during visit |
| 2 | patient_complexity | Overall patient complexity |
| 3 | time_in_hospital | Length of stay |
| 4 | num_med_prescribed | Number of active medications |
| 5 | num_med_changed | Medication adjustments |

---

## Model Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.6254 |
| Recall (Readmitted) | 29.2% |
| Precision (Readmitted) | 16.4% |
| F1 Score | 0.21 |
| True Positives Caught | 367 out of 1,259 |

Recall is prioritized over precision in this healthcare context. Missing a readmission is far more costly than a false alarm.

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/sohailalij/healthcare-readmission-pipeline.git
cd healthcare-readmission-pipeline
```

### 2. Create conda environment
```bash
conda env create -f environment.yml
conda activate healthcare-pipeline
```

### 3. Install additional dependencies
```bash
pip install xgboost imbalanced-learn shap streamlit
```

### 4. Run the Streamlit dashboard
```bash
streamlit run app.py
```

---

## Responsible AI Practices

This project implements the following responsible AI practices:

- Out-of-distribution detection warns when a patient profile is outside the training data range
- Confidence scoring flags predictions with low reliability
- Human review flag recommends clinician review for uncertain cases
- Model limitations are clearly documented
- Decision threshold is explicitly shown in the dashboard

---

## Known Limitations

- Model was trained predominantly on patients aged 40 to 90
- Performance degrades for younger patients or those not on diabetes medication
- Dataset is from 1999 to 2008 and may not reflect current clinical practices
- Race and socioeconomic factors are underrepresented in the training data

---

## Future Improvements

- Collect more diverse training data including younger patients
- Implement model calibration using CalibratedClassifierCV
- Add separate models for different age groups
- Deploy on cloud platforms such as AWS or Azure
- Add real-time monitoring for model drift
- Integrate with hospital EHR systems

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Core language |
| Pandas and NumPy | Data manipulation |
| Matplotlib and Seaborn | Visualization |
| Scikit-learn | ML pipeline and preprocessing |
| XGBoost | Primary ML model |
| SHAP | Model explainability |
| Imbalanced-learn | SMOTE for class imbalance |
| Streamlit | Interactive dashboard |
| Joblib | Model serialization |
| Git and GitHub | Version control |

---

## Author

**Sohail Ali J**
- GitHub: [@sohailalij](https://github.com/sohailalij)

---

## License

This project is licensed under the MIT License.

---

Built as a portfolio project to demonstrate end-to-end ML pipeline development in the healthcare domain.