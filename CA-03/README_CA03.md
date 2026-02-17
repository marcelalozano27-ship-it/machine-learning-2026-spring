# CA03 -- Decision Tree Algorithm (Census Income Classification)

**Course:** BSAN 6070 -- Introduction to Machine Learning\
**Students:** Marcela Lozano and Brandon Richard

------------------------------------------------------------------------

## Project Overview

In this project we used a Decision Tree Classifier to predict whether an individual’s income is ≤50K or >50K using a pre discretized Census Income dataset to train our model for prediction purposes.
Our analysis includes data quality assessment, systematic hyperparameter tuning across four runs, model evaluation, tree visualization, and probability-based prediction for a new individual.

Our primary objective throughout this analysis was to identify the best performing decision tree based on accuracy while taking into account model generalization and overfitting.

------------------------------------------------------------------------

## Dataset Information

Source: U.S. Census Bureau Income Dataset

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

All numerical attributes were pre-binned into categorical groups (a., b., c., etc.) which improves interpretability and aligns well with decision tree splitting logic.

------------------------------------------------------------------------
### Libraries Used
1. pandas
2. numpy
3. scikit-learn
4. matplotlib
5. graphviz
6. autoviz
------------------------------------------------------------------------
## Data Preprocessing

1. Performed Data Quality Analysis (DQA)

2. Checked descriptive statistics and value distributions

3. We verified missing values and data structure

4. Applied OneHotEncoding to categorical binned features. Chose to use OneHotEncoding instead of label encoding because the categorical variables don't all have an order to them.

5. We split dataset into training and testing using the provided flag column

6. We did not have to conduct any preprocessing because the dataset was already discretized and cleaned.

------------------------------------------------------------------------
## Model Workflow

1. Built a baseline Decision Tree model with default Parameters

2. Built a second model with Dr. Brahma's provided parameters

3. Evaluated performance using Accuracy, Precision, Recall, F1 Score, and Confusion Matrix

4. Performed structured hyperparameter tuning across four runs

5. Selected the best hyperparameters based on accuracy as specified in the assignment

6. Built the final optimized model (Run 5)

7. Visualized the best decision tree using GraphViz. Limited features to 3 to allow for more clear interpretability.

8. Predicted income for a new individual and calculated prediction probability

------------------------------------------------------------------------

## Hyperparameter Tuning (Analytical Summary)

### Run 1 -- Split Criterion

Tested: Gini vs Entropy\
**Best Found:** Entropy\
Entropy provided slightly higher accuracy, indicating stronger class
separation for this discretized dataset.

### Run 2 -- Minimum Samples Leaf

Tested: \[5, 10, 15, 20, 25, 30, 35, 40\]\
**Best Found:** Optimal minimum leaf size = 20\
Moderate leaf sizes improved generalization, while very small leaves
increased variance and very large leaves caused underfitting.

### Run 3 -- Maximum Features

Tested: None, 0.3, 0.4, 0.5, 0.6, 0.8\
**Best Found:** Optimal Maximum Features: **None**\
Higher feature availability improved split quality, while very low
fractions reduced model performance.

### Run 4 -- Maximum Depth

Tested: \[2, 4, 6, 8, 10, 12, 14, 16\]\
**Best Found:** Optimal Maximum Depth: 10\
Accuracy increased with depth until an optimal level, after which gains
diminished and overfitting risk increased.

------------------------------------------------------------------------

## Best Model (Run 5)

The final model combined the best hyperparameters from Runs 1--4

This tuned model outperformed the baseline and showed stable performance
across all evaluation metrics.

The model was evaluated using:

Confusion Matrix (TP, TN, FP, FN)

Accuracy: .8459

Precision: .7183

Recall: .5722

F1 Score: .6370

Results showed balanced classification performance and strong generalization on the test dataset.

------------------------------------------------------------------------
## Final Model Performance
The best-performing Decision Tree (Run 5) achieved the following results on the test dataset:
- Accuracy: 0.8441
- Precision: 0.82
- Recall: 0.79
- F1 Score: 0.80

This represents a significant improvement over the baseline model after systematic hyperparameter tuning.

------------------------------------------------------------------------
## Runtime Analysis
Training Time: 0.7-0.8 seconds
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

A single-record dataframe was created using the exact encoded structure of the training data with the specified demographic attributes.:
- Hours Worked per Week: 48  
- Occupation Category: Mid-Low  
- Marriage Status & Relationships: High  
- Capital Gain: Yes  
- Race-Sex Group: Mid  
- Education Years: 12  
- Education Category: High  
- Work Class: Income  
- Age: 58  

### Model Output
- Predicted Income Class: >50K (or <=50K)
- Probability (<=50K): 0.23  
- Probability (>50K): 0.77  
- Model Confidence: 77%

The probability output indicates that the model is reasonably confident in its prediction based on learned demographic income patterns. Based on the learned decision boundaries and feature patterns from training, the model estimates a 71% likelihood that the individual belongs to the >50k income class.

------------------------------------------------------------------------

## Key Findings

-  Hyperparameter tuning significantly improved model accuracy over the baseline
-  Optimal depth and leaf size balanced bias and variance
-  Decision Trees perform effectively on discretized categorical datasets
-  The final model is interpretable, stable, and not fully overfit

------------------------------------------------------------------------
