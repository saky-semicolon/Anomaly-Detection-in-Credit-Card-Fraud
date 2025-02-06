# Anomaly Detection in Credit Card Fraud: A Comparative Study of Autoencoder, Isolation Forest, and One-Class SVM

## Overview

This repository contains the code, datasets, and documentation for the research paper
**"Anomaly Detection in Credit Card Fraud: A Comparative Study of Autoencoder, Isolation
Forest, and One-Class SVM."** The project explores various **unsupervised learning methods**
for detecting fraudulent credit card transactions, comparing the performance of **deep learning
and traditional machine learning models**.

## Dataset

```
● Source: Kaggle's Credit Card Fraud Detection Dataset.
● Size: 284,806 transactions.
● Class Distribution:
○ Class 0 (Non-Fraudulent): 284,314 transactions.
○ Class 1 (Fraudulent): 492 transactions (~0.17% of dataset).
● Features: 30 anonymized numerical features (V1-V28), along with Time and Amount.
```
## Models Implemented

The following **anomaly detection models** were used for fraud detection:

1. **Autoencoder (Deep Learning)**
    ○ Trained on only normal transactions.
    ○ Detects fraud by evaluating reconstruction errors.
2. **Isolation Forest (Traditional Machine Learning)**
    ○ Randomly partitions data to isolate anomalies.
3. **One-Class SVM (Traditional Machine Learning)**
    ○ Constructs a boundary around normal transactions to flag anomalies.

## Performance Metrics

The models were evaluated using:


```
● ROC AUC (Area Under Curve)
● Precision & Recall
● F1-Score
● Training Time Efficiency
```
## Future Work

```
● Improve precision of the Autoencoder model.
● Implement hybrid models combining deep learning and traditional ML.
● Optimize One-Class SVM for better computational efficiency.
```
## Contributors

```
● S M Asiful Islam Saky – Research, Implementation, and Documentation.
```

## Acknowledgments

```
● Special thanks to Kaggle for providing the dataset.
● Inspired by various studies on anomaly detection and credit card fraud prevention.
```

