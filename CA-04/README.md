## Classification Using Ensemble Methods (CA04)
### Project Overview

This project applies multiple ensemble machine learning methods to predict income classification using U.S. Census data. The primary objective is to analyze how model performance changes as the number of estimators increases across different ensemble algorithms.

The models evaluated include:

**Random Forest**

**AdaBoost**

**Gradient Boosting**

**Extreme Gradient Boosting (XGBoost)**

The project focuses on data preprocessing, feature encoding, model training, and performance evaluation using Accuracy and ROC-AUC metrics to determine the ideal number of estimators for each ensemble method on this dataset.

**Dataset**

Source:
https://github.com/ArinB/MSBA-CA-03-Decision-Trees/blob/master/census_data.csv?raw=true

The dataset contains demographic and employment-related features used to predict whether an individual earns above or below a specific income threshold.

**Key Characteristics**

Predefined train and test split using the flag column

Target variable: y

Mixed data types (ordinal, categorical, and binned numerical features)

No missing values

### Project Workflow
#### **1. Data Loading and Exploration**

Imported dataset using pandas

Checked dataset shape, column names, and null values

Reviewed summary statistics to understand feature distributions. 

#### **2. Data Preprocessing**
Train/Test Split

The dataset was split using the provided flag column:

Training set: rows labeled “train”

Testing set: rows labeled “test”

This ensured consistency with the assignment’s predefined data structure.

**Feature Encoding**
Ordinal Encoding (LabelEncoder)

Applied to ordered binned variables:

hours_per_week_bin

education_num_bin

age_bin

capital_gl_bin

One-Hot Encoding (OneHotEncoder)

Applied to nominal categorical variables:

occupation_bin

msr_bin

race_sex_bin

education_bin

workclass_bin

Final encoded datasets:

X_train

X_test

y_train

y_test

This encoding approach is appropriate for tree-based and ensemble learning models.

## Models Implemented
**1. Random Forest Classifier**

**2. AdaBoost Classifier**

**3. Gradient Boosting Classifier**

**4. Extreme Gradient Boosting (XGBoost)**



## Performance Evaluation and Key Findings
**Overall Model Behavior**

Model performance did not increase linearly as the number of estimators increased. Instead, each ensemble method showed different stability patterns including early peaks, plateaus, and slight performance declines at higher estimator values.

This demonstrates that increasing model complexity does not always result in improved predictive performance.

**Random Forest Findings**

Accuracy fluctuated slightly before stabilizing

AUC remained relatively consistent across estimator values

Increasing estimators beyond the mid-range provided diminishing performance improvements

The model showed strong stability but limited performance gains after convergence

**AdaBoost Findings**

Accuracy and AUC gradually improved as estimators increased

Performance eventually plateaued, indicating model convergence

The sequential boosting process allowed the model to steadily reduce errors before stabilizing

**Gradient Boosting Findings**

Achieved the highest overall Accuracy and AUC among all models

Performance peaked at a moderate number of estimators

Slight performance decline at higher estimator values suggests potential overfitting due to increased model complexity

Provided the best balance between bias and variance for this dataset

**XGBoost Findings**

Strong early performance with peak metrics at lower estimator values

Increasing estimators beyond the optimal range reduced generalization performance

Likely overfitting occurred as model complexity increased with a fixed learning rate

**Final Conclusion**

Among all ensemble methods, Gradient Boosting produced the best overall performance in terms of both Accuracy and ROC-AUC on the Census Income dataset. This indicates that it achieved the most effective balance between model complexity and generalization.

Additionally, ROC-AUC proved to be a more informative metric than Accuracy, especially in cases where Accuracy plateaued while AUC continued to reveal differences in model discrimination ability.

**Optimal Estimators Summary**

To determine the ideal number of estimators, each model was evaluated across values ranging from 50 to 500 estimators using both Accuracy and ROC-AUC on the test dataset. The optimal value was selected based on peak performance and model stability rather than simply choosing the largest estimator size.

Optimal n_estimators based on analysis of Accuracy and AUC
#### Random Forest:	300 n_estimators
Stable accuracy and AUC after mid-range	Additional trees produced diminishing returns

#### AdaBoost: 350 n_estimators
Gradual improvement then plateau	Sequential learning improved performance until convergence

#### Gradient Boosting: 200 n_estimators
Highest Accuracy and AUC before decline	Best balance of complexity and generalization

#### XGBoost: 50 n_estimators
Early peak followed by decreasing performance	Higher estimators likely caused overfitting

## Key Interpretation

The results clearly show that larger ensemble sizes do not automatically lead to better performance. While ensemble methods reduce variance and enhance predictive accuracy, excessively increasing the number of estimators can introduce overfitting and reduce model generalization on unseen data.

Furthermore:

Optimal estimators are model-specific

Maximum estimator size is not necessarily optimal

AUC is a more reliable metric than Accuracy when evaluating classification models with performance plateaus

#### Technologies Used

Python 3

pandas

numpy

matplotlib

seaborn

scikit-learn

XGBoost

AutoViz

graphviz

Jupyter Notebook (VS Code Environment)

#### How to Run the Project

Install required packages:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost autoviz graphviz

Open the notebook:

jupyter notebook CA4MarcelaBrandon Final.ipynb

Run all cells sequentially to reproduce preprocessing, model training, and evaluation results.

##### Authors

Marcela Lozano
Brandon Richard
