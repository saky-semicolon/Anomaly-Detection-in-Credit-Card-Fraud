# Anomaly Detection in Credit Card Fraud: A Comparative Analysis of Autoencoder, Isolation Forest, and One-Class SVM

<h2>Abstract</h2>
Credit card fraud detection is an important research problem but is considered to be a difficult problem due to the
nature of the transaction data where the number of fraudulent transactions is small compared to normal transactions.
This project implements an anomaly detection system through the use of an autoencoder which is a kind of deep
learning model with unsupervised learning. The idea of the model is to discover those transactions that normally are
not like the rest as they may possibly be fraudulent. To compare the results, two other transfer learning models are
used alongside the proposed autoencoder: Isolation Forest and One-Class SVM. The data is taken from Kaggle and
the data covers 284806 transactions with 30 features obtained by applying PCA. The preprocessing part included
scaling of the data and applying feature reduction to 15 features. The normal transaction data was used to train the
autoencoder and performance metrics include accuracy, precision, recall and ROC AUC value. The results best ROC
AUC of 0.94 for the autoencoder ensures the models effectiveness for fraud detection over the transfer learning
models. This work illustrates the applicability of deep learning in the real-world financial abnormalities and
performs the comparative analysis of classical and deep learning-based approaches.

<h1>For the Complete Report See this <a href="https://github.com/saky-semicolon/Anomaly-Detection-in-Credit-Card-Fraud/blob/main/Anomaly%20Detection%20in%20Credit%20Card%20Fraud.pdf">File</a></h1>


