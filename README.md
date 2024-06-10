# Fraudulent-Job-Posting-Detection

## Project Overview

**Project Name:** Fraudulent-Job-Posting-Detection

**Project Description:**
The project aims to identify fake job postings using machine learning techniques. This project focuses on processing job descriptions to extract relevant features and training models to accurately predict fraudulent postings.

## Data Preprocessing

1. **Description Column:**
   - Only the 'description' column is used as it contains the most valuable information for predicting fake job posts.
   - Other columns are dropped to simplify the training process and improve performance.

2. **Featurization:**
   - Stop words are removed.
   - Term frequency is used as features.
   - L2 normalization is applied to rows.

## Model Training

- **Model Used:** SGDClassifier
  - Different losses were experimented with:
    - `hinge` loss for linear SVM.
    - `log` loss for logistic regression.
    - `perceptron` loss for perceptron.

## Prediction

- The preprocessor and trained model are applied to predict whether a job posting is fraudulent or not.

## Model Evaluation

- The model was evaluated using a holdout set.
- **F1 Score:** 0.696
- **Runtime:** 26.9 minutes
  
