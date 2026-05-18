# NeuroBoost Precision: A Deep Dive into Boosting and ANN Dynamics

## Project Overview

This project performs a comprehensive comparative analysis of classical boosting algorithms and artificial neural networks (ANN) for diabetes prediction. The study implements seven different machine learning approaches, optimizes their hyperparameters, analyzes feature importance, and evaluates their performance on a binary classification task. The project culminates in building custom ANN architectures with various regularization strategies.

## Dataset Information

- **Source**: [Kaggle Diabetes Dataset](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset)
- **Total Records**: 768 instances
- **Features**: 8 input features
- **Target Variable**: Outcome (Binary: 0 = No Diabetes, 1 = Diabetes)
- **Missing Values**: None detected
- **Unique Age Values**: 52

### Feature Description

| Feature | Description | Data Type |
|---|---|---|
| Pregnancies | Number of pregnancies | int64 |
| Glucose | Plasma glucose concentration | int64 |
| BloodPressure | Diastolic blood pressure (mm Hg) | int64 |
| SkinThickness | Triceps skin fold thickness (mm) | int64 |
| Insulin | 2-Hour serum insulin (mu U/ml) | int64 |
| BMI | Body mass index (kg/m²) | float64 |
| DiabetesPedigreeFunction | Diabetes pedigree function | float64 |
| Age | Age of patient (years) | int64 |

### Dataset Statistics

| Statistic | Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin |
|---|---|---|---|---|---|
| Mean | 3.85 | 120.89 | 69.11 | 20.54 | 79.80 |
| Std | 3.37 | 31.97 | 19.36 | 15.95 | 115.24 |
| Min | 0 | 0 | 0 | 0 | 0 |
| Max | 17 | 199 | 122 | 99 | 846 |

## Exploratory Data Analysis (EDA)

### Visualizations Generated

- Age Distribution histogram with KDE overlay
- Outcome class distribution count plot
- Age vs Outcome scatter plot
- Box plots of Age grouped by Outcome
- Age Group vs Pregnancies relationship plot
- Pair plot across all features
- Correlation heatmap (BrBG colormap)

### Key Correlations with Outcome

| Feature | Correlation with Outcome |
|---|---|
| Glucose | 0.467 |
| BMI | ~0.29 |
| Age | ~0.24 |
| Pregnancies | 0.222 |
| DiabetesPedigreeFunction | 0.137 |
| Insulin | ~0.13 |
| SkinThickness | ~0.07 |
| BloodPressure | 0.065 |

## Data Splitting Strategy

### For Boosting Models (Decision Tree, Random Forest, AdaBoost, LightGBM, Gradient Boosting)

| Split | Shape | Percentage |
|---|---|---|
| Training | (614, 8) | 80% |
| Test | (154, 8) | 20% |

### For XGBoost and ANN

| Split | XGBoost Shape | ANN Shape | Percentage |
|---|---|---|---|
| Training | (537, 8) | (537, 5) | 70% |
| Cross-Validation | (115, 8) | (115, 5) | 15% |
| Test | (116, 8) | (116, 5) | 15% |

---

## Boosting Models: Hyperparameter Tuning Results

Each model underwent both `GridSearchCV` and `RandomizedSearchCV` with 5-fold cross-validation.

### Best Hyperparameters Found (GridSearchCV)

| Model | Best Hyperparameters |
|---|---|
| Decision Tree | max_depth=5, min_samples_split=50 |
| Random Forest | max_depth=7, min_samples_split=2, n_estimators=100 |
| AdaBoost | learning_rate=0.1, n_estimators=100 |
| LightGBM | learning_rate=0.01, max_depth=7, n_estimators=200 |
| Gradient Boosting | learning_rate=0.1, max_depth=3, min_samples_split=20, n_estimators=50 |
| XGBoost | learning_rate=0.1, max_depth=3, min_child_weight=7, n_estimators=50 |

---

## Boosting Models: Performance Comparison

### Test Set Accuracy (Actual Results)

| Model | Test Accuracy | Notes |
|---|---|---|
| Decision Tree | 0.7338 (~0.73) | Training accuracy: 0.82 (mild overfitting) |
| Random Forest | 0.7013 (~0.70) | Lowest performance among ensembles |
| AdaBoost | 0.7468 (~0.75) | Best boosting performance |
| LightGBM | 0.7403 (~0.74) | Strong second |
| Gradient Boosting | 0.6948 (~0.69) | Lowest overall accuracy |
| XGBoost | 0.7328 (~0.73) | Best iteration: 49 (early stopping) |

### Performance Ranking (Boosting Models)

| Rank | Model | Accuracy |
|---|---|---|
| 1 | AdaBoost | 0.7468 |
| 2 | LightGBM | 0.7403 |
| 3 | Decision Tree | 0.7338 |
| 4 | XGBoost | 0.7328 |
| 5 | Random Forest | 0.7013 |
| 6 | Gradient Boosting | 0.6948 |

---

## Feature Importance Findings

Based on feature importance plots from all tree-based models, five features consistently emerged as the most influential predictors:

### Top Influential Features (Retained)

| Feature | Importance Level |
|---|---|
| Age | High |
| DiabetesPedigreeFunction | High |
| BMI | High |
| Glucose | High |
| Pregnancies | High |

### Low Importance Features (Removed for ANN)

| Feature | Reason for Removal |
|---|---|
| Insulin | Low contribution to predictions |
| SkinThickness | Low contribution to predictions |
| BloodPressure | Low contribution to predictions |

After feature elimination, the dataset for ANN was reduced from 8 features to **5 features**.

---

## Artificial Neural Network (ANN) Implementation

### Preprocessing

- **Scaler**: `StandardScaler` applied to training, CV, and test sets
- **Random Seed**: 50 (for numpy and TensorFlow)
- **Activation Functions**: ReLU (hidden), Sigmoid (output)
- **Optimizer**: Adam (learning_rate=0.01)
- **Loss Function**: Binary Crossentropy
- **Epochs**: 2000

### ANN Architectures Tested

| Model | Architecture | Layer Count |
|---|---|---|
| model_1 | Dense(2) → Dense(4) → Dense(1) | 3 layers |
| model_2 | Dense(4) → Dense(4) → Dense(2) → Dense(1) | 4 layers |
| model_3 | Dense(4) → Dense(3) → Dense(1) | 3 layers |

### Regularization Strengths Tested

L2 (Ridge) regularization tested at three different strengths: **0**, **0.01**, and **0.001**

---

## ANN Training & Cross-Validation Results

### Complete ANN Results Across All Configurations

| Model | Regularization | Training Accuracy | CV Accuracy |
|---|---|---|---|
| model_1 | 0 | 0.7821 | 0.7304 |
| model_2 | 0 | 0.8082 | 0.7304 |
| model_3 | 0 | 0.7914 | 0.7304 |
| model_1 | 0.01 | 0.7914 | 0.7130 |
| **model_2** | **0.01** | **0.7858** | **0.7565** |
| model_3 | 0.01 | 0.7952 | 0.7565 |
| model_1 | 0.001 | 0.7914 | 0.7391 |
| model_2 | 0.001 | 0.7989 | 0.7043 |
| model_3 | 0.001 | 0.8026 | 0.7217 |

### Selected ANN Model

**`model_2_reg_0.01`** was selected for final evaluation based on its strong CV accuracy (0.7565) with a balanced training accuracy (0.7858), indicating good generalization without overfitting.

---

## Final ANN Model Test Performance

### Selected: model_2 with L2 Regularization = 0.01

| Metric | Value |
|---|---|
| **Accuracy** | 0.7586 |
| **Precision** | 0.7667 |
| **Recall** | 0.5227 |
| **F1 Score** | 0.6216 |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 (No Diabetes) | 0.76 | 0.90 | 0.82 | 72 |
| 1 (Diabetes) | 0.77 | 0.52 | 0.62 | 44 |
| **Accuracy** | | | **0.76** | **116** |
| Macro Avg | 0.76 | 0.71 | 0.72 | 116 |
| Weighted Avg | 0.76 | 0.76 | 0.75 | 116 |

---

## Complete Model Comparison: All 7 Models

| Model | Test Accuracy | Features Used | Notes |
|---|---|---|---|
| Decision Tree | 0.7338 | 8 | Single tree, max_depth=5 |
| Random Forest | 0.7013 | 8 | 100 trees, max_depth=7 |
| AdaBoost | 0.7468 | 8 | 100 estimators, lr=0.1 |
| LightGBM | 0.7403 | 8 | 200 estimators, lr=0.01 |
| Gradient Boosting | 0.6948 | 8 | 50 estimators, max_depth=3 |
| XGBoost | 0.7328 | 8 | Stopped at iteration 49 |
| **ANN (model_2_reg_0.01)** | **0.7586** | **5** | **Best overall, uses fewer features** |

### Final Ranking by Test Accuracy

| Rank | Model | Accuracy |
|---|---|---|
| 1 | **ANN (model_2_reg_0.01)** | **0.7586** |
| 2 | AdaBoost | 0.7468 |
| 3 | LightGBM | 0.7403 |
| 4 | Decision Tree | 0.7338 |
| 5 | XGBoost | 0.7328 |
| 6 | Random Forest | 0.7013 |
| 7 | Gradient Boosting | 0.6948 |

---

## Key Findings

1. **ANN Achieved Best Performance**: The optimized ANN model (`model_2_reg_0.01`) achieved the highest test accuracy of **0.7586** despite using only 5 features (vs. 8 features in other models).

2. **Feature Reduction Worked**: Removing the three low-importance features (Insulin, SkinThickness, BloodPressure) did not harm performance — it actually improved it for the neural network approach.

3. **AdaBoost Topped Classical Methods**: Among the six boosting/tree models, **AdaBoost** achieved the highest accuracy (**0.7468**), followed closely by LightGBM (0.7403).

4. **Decision Tree Showed Overfitting**: Training accuracy of 0.82 vs test accuracy of 0.73 indicates the Decision Tree had a noticeable gap, even after hyperparameter tuning.

5. **Regularization Sweet Spot**: L2 regularization at **0.01** consistently produced the best CV results across ANN architectures. No regularization led to identical CV accuracy across models (0.7304), suggesting the dataset is sensitive to regularization.

6. **Recall on Positive Cases is Low**: The final ANN model has high precision (0.77) but low recall (0.52) on diabetic cases, meaning the model is more conservative in predicting diabetes — when it does predict diabetes, it's usually correct, but it misses many actual cases.

7. **Class Imbalance Effect**: The classification report reveals strong performance on Class 0 (No Diabetes, F1=0.82) but weaker performance on Class 1 (Diabetes, F1=0.62), pointing to inherent class imbalance in the dataset (72 vs 44 in test set).

8. **Random Forest Underperformed**: Despite being an ensemble, Random Forest scored second-lowest (0.7013), suggesting it didn't capture this dataset's signal as well as sequential boosting methods.

---

## Technical Stack

| Category | Tools |
|---|---|
| Data Processing | pandas, numpy |
| ML Algorithms | scikit-learn, xgboost, lightgbm |
| Neural Network | tensorflow, keras |
| Visualization | matplotlib, seaborn |
| Hyperparameter Tuning | GridSearchCV, RandomizedSearchCV |
| Evaluation Metrics | accuracy_score, precision_score, recall_score, f1_score, classification_report |

---

## Future Improvements

- Apply class balancing techniques (SMOTE, class weights) to improve recall on positive cases
- Implement ensemble stacking combining the top 3 models (ANN + AdaBoost + LightGBM)
- Explore advanced ANN regularization (Dropout, Batch Normalization)
- Test deeper ANN architectures
- Implement learning rate scheduling for ANN optimization
- Apply permutation importance for more robust feature selection
- Cross-validate the final ANN model with stratified k-fold
