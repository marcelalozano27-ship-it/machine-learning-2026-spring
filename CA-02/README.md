# Computer Assignment 2 – Naïve Bayes

**By:** Marcela Lozano and Brandon Richard

## Spam Mail Detector Using Naïve Bayes Model

### Overview

This project is for BSAN 6070-Introduction to Machine Learning and focuses on building a text based classification model that detects spam emails using a Naïve Bayes Model. The goal of the assignment is to apply fundamental machine learning concepts to a real world text mining problem. We focus on transforming email data into numerical features suitable for using a model.

The classifier is trained on a dataset of emails labeled as spam or not spam (train_data). After training the model, the model predicts whether unseen emails in the test dataset(test_data) should be classified as spam or not. We aim for the model to identify patterns in word usage across emails that form the basis of the vocabulary used to distinguish spam from legitimate emails. From there we evaluate the effectiveness of Naïve Bayes for email classification on unseen data.

### Dataset Information

- **Training Data** – Used for training the model
      702 emails (351 spam, 351 non spam)
- **Testing Data** – Used for evaluating model performance on unseen emails.
      260 emails (130 spam, 130 non spam)
- Spam emails are classified through the file name starting with "spmsgc"
- The third line of each email file contains the email body used 

### Steps taken in the analysis

1. Imported all necessary libraries to run the model (NumPy, pandas, sklearn, etc.)
2. Due to the size of the data and not wanting to lose any in transfer, we linked our Colab to a shared drive containing all needed data
3. Text features were extracted from a folder full of text file emails
4. Created a Dictionary function that tokenizes and counts every word from the various emails and selected the 3,000 most frequent words to use as the vocabulary for the model.
5. Created an Extract Features function that converts each individual email into a numerical feature vector based on the vocabulary established in the dictionary. Each feature represents the frequency of a word appearing in that email.
6. Used the Dictionary and Extract Features functions to the training and testing datasets to generate feature matrices with their corresponding labels.
7. Trained a Gaussian Naive Bayes classifier using the training feature matrix with labels. We then evaluated the model by predicting spam and non spam emails for the test dataset.  

### Model Accuracy

**96.15%**

### Conclusion

A model accuracy of 96.15% initially suggests a strong overall performance. This however can be misleading as accuracy works best on a balanced dataset. In cases where the data is imbalanced, the model may classify e-mails legitimate emails as spam (false positives) or not classifying e-mails that are spam (false negatives). It is important to include additional evaluation metrics in further analysis of model performance. One metric to look into is the precision of the model. Another evaluation metric we could look at is recall. Finally it might be useful to include a confusion matrix for a more thorough assessment.

### Libraries Imported

**numpy** for numerical computations
**pandas** for data manipulation and cleaning
**sklearn** for splitting the data, training the Naive Bayes model, and testing the model accuracy

### Credit

- Professor Arin Brahma provided his original source code
- Marcela and Brandon for refining and implementing the final model
