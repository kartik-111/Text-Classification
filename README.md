# Text Classification Project

## Overview
This project focuses on text classification using both statistical and deep learning approaches. The objective is to build and evaluate multiple models to classify textual data accurately. The project is divided into two main parts:

1. **Statistical Method (Part 1A)**: Implements a machine learning-based approach using the `GradientBoostingClassifier` algorithm.
2. **Deep Learning Method (Part 1B)**: Implements a neural network-based approach using Long Short-Term Memory (LSTM) networks.

## Part 1: Text Classification
### Dataset
The content has been gathered from the popular academic website **arXiv.org** for articles tagged as computer science content (though some are in mathematics or physics categories). The dataset consists of the following fields:

- **Title**: The full title of the article.
- **Abstract**: The full abstract of the article.
- **InformationTheory**: Binary label (`1` if classified as an Information Theory article, otherwise `0`).
- **ComputerVision**: Binary label (`1` if classified as a Computer Vision article, otherwise `0`).
- **ComputationalLinguistics**: Binary label (`1` if classified as a Computational Linguistics article, otherwise `0`).

These three classes (**Computational Linguistics, Information Theory, and Computer Vision**) can occur in any combination, meaning an article could belong to multiple categories simultaneously.

### Objective
The goal of this project is to build text classifiers that predict each of these three classes individually using the **Abstract** field. The same experiment is then repeated using only the **Titles**. Different text classifiers are trained using multiple configurations to evaluate their performance on the three binary classification tasks.

### Configurations Considered
This project explores various configurations for training text classification models:

1. **Task**: 3 binary classification tasks.
2. **Input Features**:
   - Use **Abstract** only.
   - Use **Title** only.
3. **Algorithms**:
   - One **RNN-based classifier** (LSTM).
   - One **statistical classifier** (e.g., logistic regression, SVM, or another readily available classifier in Python).
4. **Text Preprocessing Variations**:
   - Version 1: Stemming, removal of stopwords, conversion to lowercase, etc.
   - Version 2: Different text preprocessing choices.
5. **Training Data Size**:
   - **First 1000 cases** in the training set.
   - **Full training set** (excluding the last 10% as validation set).

### Model Training and Testing
Each model is trained separately under the above configurations, resulting in a **2 (Abstract vs. Title) × 3 (Binary classifiers) × 2 (Algorithms) × 2 (Preprocessing methods) × 2 (Training sizes) = 24 different configurations**.

For each configuration, the trained model is tested on the **test set** by ensuring:
- Models trained on **Abstracts** are tested on **Abstracts** of the test set.
- Models trained on **Titles** are tested on **Titles** of the test set.

### Evaluation Metrics
The performance of the models is evaluated using the following metrics:
- **F1-score**
- **Precision**
- **Recall**
- **Accuracy**
- **Precision-Recall curve**

### Discussion and Insights
The analysis of results includes:
- **Comparison of the two algorithms**: When and why did each perform better?
- **Abstract vs. Title-based models**: How did their performance differ?
- **Effectiveness of different text preprocessing techniques**: Which one worked better and why?
- **Insights from evaluation metrics and precision-recall curves**.

## Results and Comparison
- A comparison is drawn between the **Gradient Boosting** and **LSTM-based models**.
- Performance is analyzed based on various evaluation metrics.
- Insights into which approach works better for different types of text classification problems.

## Outcome
- The **Gradient Boosting Model** performed well for structured and smaller datasets, showing high precision and recall.
- The **LSTM Model** excelled in handling large textual data, capturing contextual meaning better than traditional machine learning models.
- Overall, LSTM-based deep learning models provided better generalization for complex text data, while Gradient Boosting worked efficiently for interpretable results in structured text classification.
- The findings suggest that hybrid approaches combining both techniques could further enhance classification accuracy.

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - Pandas, NumPy (Data Handling)
  - Scikit-learn (Gradient Boosting and Evaluation Metrics)
  - TensorFlow/Keras (LSTM Implementation)

## Future Scope
- Implement additional NLP techniques like **BERT** or **Transformers**.
- Perform hyperparameter tuning for LSTM models.
- Explore other ensemble learning techniques.

## How to Run the Project
1. Install the required dependencies:
   ```sh
   pip install pandas numpy scikit-learn tensorflow
   ```
2. Clone this repository and navigate to the project folder.
   ```sh
   git clone <repository-url>
   cd text_classification_project
   ```
3. Run the Jupyter Notebook or Python script to train and evaluate models.

## Author
- **[Kartik Chaturvedi]**
