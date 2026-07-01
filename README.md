# Patient Readmission Prediction

## Project Overview
This project builds a machine learning model to predict whether a diabetic 
patient will be readmitted to hospital within 30 days of discharge. Using 
10 years of clinical data from 130 US hospitals, the project follows a 
structured 10-step data science workflow, from data cleaning through 
model interpretation and business recommendations.

## Problem Statement
Hospital readmissions within 30 days are costly for both healthcare systems 
and patients. For diabetic patients specifically, readmission rates remain 
high despite preventive efforts. This project asks: can we identify, at the 
point of discharge, which patients are most at risk of being readmitted, 
and what clinical factors drive that risk?

## Dataset
- **Source:** [UCI Machine Learning Repository - Diabetes 130-US Hospitals 
for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- **Size:** 101,766 patient encounters, 50 features
- **Target variable:** Readmitted within 30 days (binary: yes/no)
- **Class imbalance:** ~89% not readmitted, ~11% readmitted

## Tools & Libraries
| Tool | Purpose |
|---|---|
| Python | Core programming language |
| pandas | Data manipulation and cleaning |
| scikit-learn | Model building, evaluation, preprocessing |
| XGBoost | Final model (gradient boosting) |
| SHAP | Model interpretation and feature direction |
| imbalanced-learn | SMOTENC resampling experiment |
| Matplotlib / Seaborn | Visualization |
| Google Colab | Development environment |

## Project Workflow

The project follows a structured 10 step data science framework:

| Step | Description |
|---|---|
| 1. Business Understanding | Define the clinical problem and success metrics |
| 2. Data Collection | Load dataset via UCI ML Repository API |
| 3. Data Cleaning | Handle missing values, remove duplicates, fix inconsistencies |
| 4. Exploratory Data Analysis | Distributions, class imbalance, feature relationships |
| 5. Feature Engineering | Encode categoricals, engineer utilization features |
| 6. Model Building | Logistic Regression baseline and Random Forest |
| 7. Model Evaluation | Compare models on ROC-AUC, recall, precision |
| 8. Hyperparameter Tuning | RandomizedSearchCV + narrowed GridSearchCV on XGBoost |
| 9. Feature Importance & SHAP | Identify and interpret top predictive features |
| 10. Business Insights | Translate findings into clinical recommendations |

## Model Comparison

Five modeling approaches were evaluated systematically:

| Model | ROC-AUC | Recall (Readmitted) | Precision (Readmitted) | Accuracy |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.645 | 0.51 | 0.17 | 0.67 |
| RF - Threshold 0.2 | 0.652 | 0.21 | 0.26 | 0.84 |
| RF - balanced_subsample (0.2) | 0.653 | 0.21 | 0.26 | 0.85 |
| RF - SMOTENC (0.2, stacked) | 0.600 | 0.69 | 0.14 | 0.48 |
| RF - SMOTENC (0.5, natural) | 0.600 | 0.15 | 0.17 | 0.82 |
| **XGBoost - scale_pos_weight** | **0.684** | **0.61** | **0.18** | **0.65** |
| **XGBoost - GridSearchCV Tuned** | **0.687** | **0.61** | **0.18** | **0.65** |

**Why XGBoost won:** threshold tuning, balanced_subsample, and SMOTENC all 
converged on the same ~0.65 ROC-AUC ceiling for Random Forest, regardless 
of how class imbalance was handled. XGBoost's sequential, error correcting 
architecture extracted more signal from the same features and was the only 
approach to meaningfully break that ceiling.

## Key Findings

**1. Prior utilization history is the dominant risk signal.**
Number of prior inpatient visits (`number_inpatient`) is the strongest 
single predictor, roughly double the importance of the next ranked feature. 
SHAP analysis confirms the direction: more prior hospitalizations 
consistently push predicted readmission risk higher. `total_prior_visits` 
and `number_diagnoses` follow the same pattern.

**2. Discharge destination is a strong, independent second signal.**
Readmission rates vary sharply by where a patient goes after discharge:

| Discharge destination | Readmission rate | Patients |
|---|---|---|
| Home (baseline) | 9.3% | 60,234 |
| Skilled nursing facility | 14.7% | 13,954 |
| Another short-term hospital | 16.1% | 2,128 |
| Another inpatient care institution | 20.9% | 1,184 |
| Another rehab facility | 27.7% | 1,993 |

Patients transferred to continued institutional care face 1.5x to 3x the 
baseline readmission rate of patients discharged home.

**3. Imbalance handling had diminishing returns; model architecture mattered more.**
Three separate imbalance correction techniques (threshold adjustment, 
balanced_subsample, SMOTENC) all plateaued at the same ROC-AUC ceiling 
for Random Forest. The performance breakthrough came from switching model 
class, not from further resampling, a finding that informs future 
modeling decisions on similar healthcare datasets.

## Recommendations

Two actionable, independent risk signals emerged that do not require the 
full model to act on:

1. **At the point of discharge (immediately available):** flag patients 
being discharged to a skilled nursing facility, rehab facility, another 
hospital, or another inpatient institution for enhanced discharge planning 
and coordinated care handoff.

2. **From patient history (requires record pull):** prioritize patients 
with 2 or more prior inpatient admissions for follow up within 7 days of 
discharge, including a phone call, medication reconciliation review, or 
scheduled outpatient appointment.

Combining both signals gives care teams a practical, two factor targeting 
rule that works even without model scoring infrastructure in place.

## Limitations

- All modeling approaches converged within a 0.60 to 0.69 ROC-AUC band, 
suggesting the performance ceiling is driven by available features rather 
than modeling choices.
- Key clinical signals likely missing from this dataset include: lab result 
trends over time, social determinants of health (housing stability, 
transportation, caregiver support), and medication adherence data.
- Model precision (18%) means most flagged patients will not actually be 
readmitted. The model is best used as a triage prioritization tool, not 
a standalone diagnostic.

## What I Learned

- **Imbalance handling has limits:** when ROC-AUC stays flat across 
multiple resampling techniques, the bottleneck is feature signal, not 
class distribution. Recognizing this earlier saves significant 
experimentation time.
- **Threshold and model choice interact:** lowering a decision threshold 
on an already rebalanced model (SMOTENC + threshold 0.2) double corrects 
for imbalance and produces misleadingly high recall at the cost of 
precision and accuracy, a subtle trap that requires checking both 
threshold dependent and threshold independent metrics.
- **SHAP adds direction, not just magnitude:** feature importance rankings 
alone do not tell you whether high values increase or decrease risk. SHAP 
summary plots confirmed the directionality of the top predictors, which 
is what makes feature findings actionable in a clinical context.
- **Sequential boosting outperforms bagging on narrow signals:** 
`number_inpatient` dominates this dataset. XGBoost's ability to 
iteratively correct errors around that signal explains its performance 
advantage over Random Forest, where the same signal gets diluted across 
independently-built trees.

## How to Run

1. Clone this repository
2. Open `Diabetes.ipynb` in Google Colab or Jupyter Notebook
3. Run all cells in order (Runtime > Run all)
4. All required libraries are installed within the notebook

**Required libraries:**
pip install ucimlrepo xgboost shap imbalanced learn

## Author
**Edwin Kwao-Asemnor**
[GitHub: github.com/eddiek2](https://github.com/eddiek2)
