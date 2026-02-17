# CA03 -- Decision Tree Algorithm (Census Income Classification)

**Course:** BSAN 6070 -- Introduction to Machine Learning\
**Students:** Marcela Lozano and Brandon Richard

------------------------------------------------------------------------

## Project Overview

This project implements a Decision Tree Classifier to predict whether an individual’s income is ≤50K or >50K using a discretized Census Income dataset.
The analysis includes data quality assessment, systematic hyperparameter tuning across four runs, model evaluation, tree visualization, and probability-based prediction for a new individual.

The primary objective was to identify the best-performing decision tree based on accuracy while analyzing model generalization and overfitting.

------------------------------------------------------------------------

## Dataset Information

Source: U.S. Census Bureau (Census Income Dataset)

Total Observations: 48,842

Target Classes: ≤50K (0) and >50K (1)

Feature Type: Discretized (binned) categorical variables

### Model Features Used

hours_per_week_bin

occupation_bin

msr_bin (Marriage Status & Relationships)

capital_gl_bin

race_sex_bin

education_num_bin

education_bin

workclass_bin

age_bin

All numerical attributes were pre-binned into categorical groups (a., b., c., etc.), which improves interpretability and aligns well with decision tree splitting logic.

------------------------------------------------------------------------
### Libraries Used

pandas
numpy
scikit-learn
matplotlib
graphviz
------------------------------------------------------------------------
## Data Preprocessing

Performed Data Quality Analysis (DQA)

Checked descriptive statistics and value distributions

Verified missing values and data structure

Applied OneHotEncoding to categorical binned features

Split dataset into training and testing using the provided flag column

Minimal preprocessing was required because the dataset was already discretized and cleaned.

------------------------------------------------------------------------
## Model Workflow

Built a baseline Decision Tree model

Evaluated performance using Accuracy, Precision, Recall, F1 Score, and Confusion Matrix

Performed structured hyperparameter tuning across four runs

Selected the best hyperparameters based on accuracy

Built the final optimized model (Run 5)

Visualized the best decision tree using GraphViz

Predicted income for a new individual and calculated prediction probability

------------------------------------------------------------------------

## Hyperparameter Tuning (Analytical Summary)

### Run 1 -- Split Criterion

Tested: Gini vs Entropy\
**Best Found:** Entropy\
Entropy provided slightly higher accuracy, indicating stronger class
separation for this discretized dataset.

### Run 2 -- Minimum Samples Leaf

Tested: \[5, 10, 15, 20, 25, 30, 35, 40\]\
**Best Found:** Optimal mid-range leaf size-20 (from results table)\
Moderate leaf sizes improved generalization, while very small leaves
increased variance and very large leaves caused underfitting.

### Run 3 -- Maximum Features

Tested: None, 0.3, 0.4, 0.5, 0.6, 0.8\
**Best Found:** Value with highest accuracy in Run 3 results was **None**\
Higher feature availability improved split quality, while very low
fractions reduced model performance.

### Run 4 -- Maximum Depth

Tested: \[2, 4, 6, 8, 10, 12, 14, 16\]\
**Best Found:** Depth where accuracy plateaued-10 (from Run 4 results)\
Accuracy increased with depth until an optimal level, after which gains
diminished and overfitting risk increased.

------------------------------------------------------------------------

## Best Model (Run 5)

The final model combined the best hyperparameters from Runs 1--4: - Best
Criterion: Entropy\
- Best Min Samples Leaf: From Run 2\
- Best Max Features: From Run 3\
- Best Max Depth: From Run 4

This tuned model outperformed the baseline and showed stable performance
across all evaluation metrics.



The model was evaluated using:

Confusion Matrix (TP, TN, FP, FN)

Accuracy

Precision

Recall

F1 Score

Results showed balanced classification performance and strong generalization on the test dataset.

------------------------------------------------------------------------

## Runtime Analysis

Training time for the best model was measured using
`time.perf_counter()`.\
The model trained efficiently due to the discretized feature structure
and the computational simplicity of decision trees.

------------------------------------------------------------------------

## Tree Visualization and Interpretation

The GraphViz visualization revealed:

Clear hierarchical decision splits

Interpretable feature importance

Logical rule-based classification paths

The tree structure indicates controlled growth rather than a fully overfit model.

------------------------------------------------------------------------

## Overfitting Analysis

Train and test accuracy were compared to evaluate generalization.\
The small performance gap suggests the tuned model is not significantly
overfitting. Hyperparameters such as max_depth and min_samples_leaf
helped regularize the tree.

------------------------------------------------------------------------

## Prediction for New Individual

A single-record dataframe was created using the exact encoded structure of the training data with the specified demographic attributes.
The best model was used to:

Predict income class (≤50K or >50K)

Compute class probabilities using predict_proba()

Report model confidence score

Providing probability estimates improves interpretability and satisfies the assignment requirement for prediction confidence.
------------------------------------------------------------------------

## Key Findings

-  Hyperparameter tuning significantly improved model accuracy over the baseline

Optimal depth and leaf size balanced bias and variance

Decision Trees perform effectively on discretized categorical datasets

The final model is interpretable, stable, and not fully overfit

------------------------------------------------------------------------

## Reproducibility

To run this project: 1. Open the Jupyter Notebook (.ipynb) 2. Install
required libraries: - pandas - numpy - scikit-learn - matplotlib -
graphviz 3. Run all cells sequentially
