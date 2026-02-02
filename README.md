# CA-01: Housing Price Prediction

##Assignment Overview
This project is for ***BSAN 6070- Introduction to Machine Learning** and focuses on EDA and data preprocessing for a dataset on housing price prediction. The goal of the assignment is to understand the data using Univariate and Multivariate visualizations as well as general data exploration. We aimed to identify patterns as well as relationships within the data to prepare it for modeling. We included a structured data quality report and a detailed feature selection process.

##Technologies and Packages Used
This analysis was conducted using **Python** in **Google Colab**.
The primary libraries used include:
**pandas** for data manipulation and cleaning
**numpy** for numerical computations
**matplotlib** for data vizualization
**seaborn** for statistical visualizations

##Steps taken in the analysis

1) An Exploratory Data Analysis was performed to understand the data
   - Looked at dataset structure using summary statistics
   -Visualized numerical and categorical feature distributions 
   - Analyzed missing value patterns
   - Identified features with outliers
  
2) Data Preprocessing
   - Handled Missing values
   - Removed outliers using a threshold
   - Used Feature encoding on ordinal and nominal categorical variables
  
3) Collinearity Analysis and Feature Selection
    - Used correlation matrices and threshold filtering to identify multicollinearity
    - Identified highly correlated paris with over 70% correlation
    - Reduced redundancy of features
  
##Key Insights
- The Overall Quality and Above Ground Living Area features seemed to be the strongest predictors of house price
- Several numerical variables had right skewed distributions and extreme outliers
- For categorical features there was high concentration in a small number of categories
- Certain features showed strong multicolinearity and were addressed accordingly

## Data Source
The dataset used in this analysis was provided by the course instructor on the following GitHub repository:
https://github.com/ArinB/MSBA-CA-Data/raw/refs/heads/main/CA01/house-price-train.csv

