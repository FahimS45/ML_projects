# Residential Energy Consumption Prediction in BTU

## Project Overview

This project builds a machine learning model to predict residential energy consumption in British Thermal Units (BTU) based on household characteristics, appliances, and energy usage patterns. The model utilizes data from the Residential Energy Consumption Survey (RECS) 2015 and applies advanced preprocessing techniques and machine learning algorithms.

## Dataset Information

- **Data Source**: RECS 2015 Public Dataset (v4) with accompanying codebook
- **Initial Features**: 265 columns containing household characteristics and energy data
- **Data Points**: Complete residential survey records
- **Target Variable**: Total BTU consumption across all fuel types

## Project Workflow

### 1. Data Exploration & Understanding

Explored the RECS dataset structure to understand variables and their meanings:

- Reviewed all 265 feature columns and their descriptions from codebook
- Identified different types of energy consumption measurements
- Analyzed dataset dimensions and basic statistics

### 2. Target Variable Creation

Created the target variable from four energy consumption types:

| Energy Source | Variable Name | Unit | Description |
|---|---|---|---|
| Electricity | BTUEL | BTU | Energy from electrical usage |
| Natural Gas | BTUNG | BTU | Energy from natural gas consumption |
| Liquid Propane | BTULP | BTU | Energy from propane consumption |
| Fuel Oil | BTUFO | BTU | Energy from fuel oil consumption |

Final target variable: **TOTALBTU_incode** = sum of all four energy sources

### 3. Feature Engineering & Data Cleaning

#### Feature Removal

Dropped unnecessary columns:
- Statistical weight columns (NWEIGHT, BRRWT1-BRRWT96) - 97 columns
- Detailed energy and dollar breakdowns by appliance usage - over 300 columns
- Duplicate or redundant measurements

Retained final dataset: **265 columns** containing meaningful household features

#### Feature Mapping & Encoding

Applied comprehensive value mapping to categorical variables with over 100 mappings including:

- **Geographic Data**: Region (Northeast, Midwest, South, West), Division (New England, Middle Atlantic, etc.)
- **Housing Characteristics**: Type (Mobile home, Detached house, Apartment), Stories, Garage size, Year built
- **Insulation & Structure**: Insulation quality (Well/Adequately/Poorly insulated), Drafty condition, Wall type, Roof type, Window type
- **Appliances**: Refrigerator size/age/type, HVAC equipment, Cooking appliances, Washer/Dryer specifications
- **Climate Control**: Heating/Cooling equipment types, Thermostat usage patterns, Temperature preferences
- **Household Demographics**: Education level, Income bracket, Employment status, Household size
- **Energy Assistance**: Various energy assistance program participation flags

#### Feature Separation

| Feature Type | Count | Purpose |
|---|---|---|
| Numerical Features | 17 | Direct continuous measurements |
| Ordinal Categorical | 27 | Ordered categories (age ranges, sizes, counts) |
| Nominal Categorical | 221 | Unordered categories (types, yes/no flags) |

### 4. Data Preprocessing Pipeline

Built a comprehensive preprocessing pipeline:

```
Step 1: Categorical Encoding
  ├─ Ordinal features → OrdinalEncoder (preserves order)
  ├─ Nominal features → OneHotEncoder (creates binary flags, drop_first=True)
  └─ Numerical features → Pass through unchanged

Step 2: Feature Scaling
  └─ StandardScaler applied to all features
```

**Output Shape**: 265 input features → ~500+ features after one-hot encoding

### 5. Dimensionality Reduction (PCA)

Addressed high-dimensionality problem using Principal Component Analysis:

#### PCA Analysis Results

- **Initial Features After Encoding**: 500+ features
- **Explained Variance by Components**: 
  - 90% variance explained by **~250 components**
  - 95% variance explained by **~300 components**
  - 99% variance explained by **~400 components**

#### Key Finding

Reduced dataset to **250 PCA components** that capture 90%+ of variance while eliminating noise and multicollinearity

## Machine Learning Models

### Model Development & Evaluation

Trained and compared three regression algorithms:

#### 1. XGBoost Regressor

| Metric | Value |
|---|---|
| R² Score (Test Set) | High performance |
| Cross-validation (10-fold) | Strong consistency |
| Model Type | Gradient boosting ensemble |

#### 2. Random Forest Regressor

| Metric | Value |
|---|---|
| R² Score (Test Set) | Competitive performance |
| Algorithm | Bagging ensemble |
| Feature Importance | Bagging provides natural importance scores |

#### 3. Decision Tree Regressor

| Metric | Value |
|---|---|
| R² Score (Test Set) | Baseline comparison |
| Interpretability | Highly interpretable |
| Purpose | Baseline model for comparison |

### Model Performance Visualization

Generated prediction vs. actual plots for all three models to assess:

- **Prediction accuracy**: How close predictions align with actual values
- **Residual patterns**: Any systematic biases in predictions
- **Value range coverage**: Model performance across different consumption levels

## Data Processing Summary

| Stage | Details |
|---|---|
| **Raw Dataset** | 265 columns, multiple categorical types |
| **Feature Removal** | Dropped redundant weight and detailed breakdowns |
| **Feature Mapping** | 100+ value mappings for categorical encoding |
| **After Encoding** | 500+ features (numerical + one-hot encoded) |
| **After Standardization** | Features normalized to mean=0, std=1 |
| **After PCA** | 250 components capturing 90%+ variance |
| **Train/Test Split** | 80% training, 20% testing |

## Key Findings

1. **Feature Richness**: The RECS 2015 dataset contains comprehensive household data across appliances, comfort settings, energy assistance programs, and demographic information

2. **Dimensionality Challenge**: Initial 500+ features (post-encoding) were reduced through PCA, demonstrating that approximately 250 components capture most meaningful variation in energy consumption patterns

3. **Multiple Energy Sources**: Energy consumption varies significantly based on primary heating fuel, appliance types, household demographics, and geographic location

4. **Model Comparison**: XGBoost demonstrated superior performance compared to Random Forest and Decision Tree baselines, suggesting non-linear relationships and interaction effects between features

5. **Prediction Visualization**: Scatter plots of predicted vs. actual consumption reveal model accuracy patterns and identify potential areas for improvement in specific consumption ranges

## Technical Stack

- **Data Processing**: pandas, numpy
- **Preprocessing**: scikit-learn (ColumnTransformer, StandardScaler, OrdinalEncoder, OneHotEncoder)
- **Dimensionality Reduction**: scikit-learn PCA
- **Machine Learning**: 
  - XGBoost (XGBRegressor)
  - scikit-learn (RandomForestRegressor, DecisionTreeRegressor)
- **Evaluation Metrics**: Mean Squared Error (MSE), R² Score
- **Cross-validation**: 10-fold cross-validation
- **Visualization**: matplotlib, seaborn

## Future Improvements

- Hyperparameter tuning with GridSearchCV or RandomizedSearchCV
- Ensemble methods combining multiple models
- Feature importance analysis to identify key consumption drivers
- Error analysis on specific appliances or household demographics
- Time-series analysis if temporal data becomes available
- Clustering households by consumption patterns for targeted recommendations
