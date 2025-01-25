# Credit-Card-Fraud-Detection

This project focuses on detecting fraudulent transactions from credit card data. It implements multiple machine learning models to classify transactions as either `Fraud` or `Not Fraud` and compares their performances.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Approach](#approach)
- [Models Used](#models-used)
- [Results](#results)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)

---

## Overview
Credit card fraud is a significant financial issue worldwide. This project aims to identify fraudulent transactions effectively using supervised machine learning techniques. The challenge lies in handling the severe class imbalance, as fraudulent transactions are far fewer than legitimate ones.

## Dataset
The dataset used for this project contains anonymized credit card transactions. It includes the following:
- **Features:** Numerical values derived from PCA (Principal Component Analysis).
- **Target:**
  - `0` for Not Fraud.
  - `1` for Fraud.
- **Imbalance:** The dataset is highly imbalanced, with a significantly smaller number of fraudulent transactions compared to non-fraudulent ones.

---

## Project Structure
1. **Data Preprocessing:**
   - Cleaning and scaling the data.
   - Handling class imbalance through techniques like SMOTE (Synthetic Minority Over-sampling Technique).
2. **Exploratory Data Analysis (EDA):**
   - Distribution of classes.
   - Feature correlation and importance.
3. **Model Training:**
   - Implementation of several machine learning models.
4. **Evaluation:**
   - Metrics: Precision, Recall, F1-score, Accuracy, and AUC-ROC.
5. **Comparison:**
   - Analyze the strengths and weaknesses of each model.

---

## Requirements
Install the necessary Python libraries:
```bash
pip install -r requirements.txt
```
### Libraries Used
- `pandas` for data manipulation.
- `numpy` for numerical computations.
- `matplotlib` and `seaborn` for visualizations.
- `scikit-learn` for machine learning algorithms.
- `tensorflow` for neural network implementation.

---

## Approach
### Preprocessing
- **Data Splitting:**
  - Train-validation-test split to ensure fair model evaluation.
- **Scaling:**
  - StandardScaler to normalize features for consistent input to machine learning models.
- **Handling Imbalance:**
  - Applied SMOTE to oversample the minority (Fraud) class.

### Model Selection
Implemented the following models:
1. **Logistic Regression:**
   - Simple and interpretable baseline model.
2. **Random Forest:**
   - Ensemble method for robust predictions.
3. **Gradient Boosting (GBC):**
   - Strong performance on imbalanced datasets.
4. **Linear SVC:**
   - Balances classes using `class_weight`.
5. **Shallow Neural Network:**
   - Multi-layer perceptron for capturing complex patterns.

---

## Models Used
### Logistic Regression
- Pros: Fast, interpretable.
- Cons: Limited handling of complex patterns.

### Random Forest
- Pros: Robust, handles class imbalance.
- Cons: Prone to overfitting with high depth.

### Gradient Boosting (GBC)
- Pros: High performance, effective for imbalanced datasets.
- Cons: Computationally expensive.

### Linear SVC
- Pros: High precision and recall, especially with `class_weight` adjustment.
- Cons: Requires fine-tuning for convergence.

### Shallow Neural Network
- Pros: Learns complex patterns.
- Cons: Requires more data and tuning.

---

## Results
| Model                  | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) | Accuracy |
|------------------------|-------------------|----------------|------------------|----------|
| Logistic Regression    | 0.73              | 0.53           | 0.61             | 0.77     |
| Random Forest          | 1.00              | 0.54           | 0.70             | 0.77     |
| Gradient Boosting (GBC)| 1.00              | 0.76           | 0.86             | 0.88     |
| Linear SVC             | 1.00              | 0.87           | 0.93             | 0.94     |


### Key Insights
- **Linear SVC** performed the best overall with an F1-score of **0.93** for Fraud.

---

## Conclusion
- Effective fraud detection requires a balance between precision and recall to minimize false positives and negatives.
- The Linear SVC achieved the best results, but the Shallow Neural Network is also a viable alternative with further tuning.
- Future work could explore:
  - Hyperparameter optimization.
  - Feature engineering.
  - Ensemble methods.

---

## How to Run
1. Clone the repository:
   ```bash
   git clone [<repository_url>](https://github.com/SoumayaEloualid/Credit-Card-Fraud-Detection.git)
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook credit-card-fraud-detection.ipynb
   ```
4. Follow the notebook steps to preprocess data, train models, and evaluate results.

---

### Acknowledgments
Special thanks to the contributors of the dataset and the open-source libraries used in this project.

