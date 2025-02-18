Text Classification Project

Overview

This project focuses on text classification using both statistical and deep learning approaches. The objective is to build and evaluate multiple models to classify textual data accurately. The project is divided into two main parts:

Statistical Method (Part 1A): Implements a machine learning-based approach using the GradientBoostingClassifier algorithm.

Deep Learning Method (Part 1B): Implements a neural network-based approach using Long Short-Term Memory (LSTM) networks.

Part 1A: Statistical Method

Algorithm Used: Gradient Boosting Classifier

The GradientBoostingClassifier is employed for text classification, which works by training multiple decision trees in sequence, where each tree corrects the errors made by the previous ones. The final prediction is made by combining all the trained trees.

Objectives

Build 24 different model combinations using Gradient Boosting.

Evaluate models using the following metrics:

Accuracy

Precision

Recall

F1-score

Precision-Recall (P-R) Curves

Data Preprocessing

Dataset Used: train.csv and test.csv

Class Distribution Analysis: Examines the proportion of each label in the dataset.

Text Processing Steps:

Tokenization

Lemmatization (using a custom LemmaTokenizer class)

Vectorization using TF-IDF

Model Training

Implemented GradientBoostingClassifier with hyperparameter tuning.

Trained the model on the preprocessed text data.

Evaluated using different performance metrics.

Part 1B: Deep Learning Method

Algorithm Used: Long Short-Term Memory (LSTM)

LSTM is a type of Recurrent Neural Network (RNN) designed to process sequential data efficiently. It helps in capturing long-term dependencies in textual data.

Objectives

Train an LSTM-based neural network for text classification.

Compare its performance with the Gradient Boosting approach.

Data Preprocessing

Tokenization and Padding:

Convert text into numerical sequences.

Apply padding to ensure uniform sequence length.

Embedding Layer:

Transform words into dense vector representations.

Model Architecture

Embedding Layer: Converts words into numerical embeddings.

LSTM Layer: Processes sequential data.

Dense Layer: Outputs predictions for classification.

Model Training and Evaluation

Trained using categorical cross-entropy loss.

Evaluated using the same metrics as Part 1A.

Results and Comparison

A comparison is drawn between the Gradient Boosting and LSTM-based models.

Performance is analyzed based on various evaluation metrics.

Insights into which approach works better for different types of text classification problems.

Outcome

The Gradient Boosting Model performed well for structured and smaller datasets, showing high precision and recall.

The LSTM Model excelled in handling large textual data, capturing contextual meaning better than traditional machine learning models.

Overall, LSTM-based deep learning models provided better generalization for complex text data, while Gradient Boosting worked efficiently for interpretable results in structured text classification.

The findings suggest that hybrid approaches combining both techniques could further enhance classification accuracy.

Technologies Used

Programming Language: Python

Libraries:

Pandas, NumPy (Data Handling)

Scikit-learn (Gradient Boosting and Evaluation Metrics)

TensorFlow/Keras (LSTM Implementation)

Future Scope

Implement additional NLP techniques like BERT or Transformers.

Perform hyperparameter tuning for LSTM models.

Explore other ensemble learning techniques.
