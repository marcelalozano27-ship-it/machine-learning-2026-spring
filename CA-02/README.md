# Computer Assignment 2 – Naïve Bayes

**By:** Marcela and Brandon

## Spam Mail Detector Using Naïve Bayes Model

### Overview

To detect spam e-mails, we implemented a Multinominal Naïve Bayes Model to detect key words linked to spam or not spam.

The classifier is trained on a dataset of emails labeled as spam or not spam (train_data), so when given the test_data it will try to predict as accurately as possible whether new emails are spam or not.

### Model Step Process

1. Imported all necessary libraries to run the model (NumPy, pandas, sklearn, etc.)
2. Due to the size of the data and not wanting to lose any in transfer, linked our CoLab to a shared drive containing all needed data
3. Created a Dictionary function that will split and count every word from the various emails and only returned 3000 of the most common words for the model
4. Created an Extract Features function that will take those words, convert them into vectors, and document each appearance
5. Calling our training data and testing data, we plug them into our Dictionary and Extract Features functions
6. Plugged the trained variables into our NB model, then ran it with our testing data

### Model Accuracy

**96.15%**

### Conclusion

Model accuracy of 96% would be acceptable if this is a balanced dataset model. However, if it is unbalanced this can pose a problem classifying e-mails that are not spam as spam (false positive) or not classifying e-mails that are spam (false negative).

### Dataset Information

- **Training Data** – "spmsgc" is the classification of SPAM emails in the training dataset
- **Testing Data** – 

### Libraries Imported

- NumPy
- pandas
- sklearn

### Credit

- Professor Arin Brahma for his source code
- Marcela and Brandon for improving the source code