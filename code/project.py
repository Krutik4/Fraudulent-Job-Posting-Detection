# Project
# Krutik Rajesh Panchal, kp7514@rit.edu
# Did not use hint file
# Collaborated with Preet Jain as part of same study group no 1

import pandas as pd
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from sklearn.metrics import make_scorer, f1_score
from sklearn.utils import shuffle


class my_model():

    def preprocess(self, X):
        X = X.drop(columns=['location','telecommuting','has_questions'])
        X['title'] = X['title'].str.strip(" " + "*" + "?" + "#")
        X['description'] = X['description'].str.strip(" " + "*" + "?" + "#")
        X['requirements'] = X['requirements'].str.strip(" " + "*" + "?" + "#")

        return X

    def oversample(self,X,y):
        # Concatenate the features and labels for the training set
        X['fraudulent'] = y

        # Separate the majority and minority classes
        majority_class = X[X['fraudulent'] == 0]
        minority_class = X[X['fraudulent'] == 1]

        # Upsample the minority class to match the majority class
        minority_class_upsampled = resample(minority_class,
                                            replace=True,  # Sample with replacement
                                            n_samples=round(len(majority_class)),
                                            # Match the number of majority class samples
                                            random_state=42)

        # Concatenate the majority class and the upsampled minority class
        df_train_upsampled = pd.concat([majority_class, minority_class_upsampled])

        # Shuffling concatenated array
        df_shuffled = shuffle(df_train_upsampled, random_state=42)

        # Separate features and labels after oversampling
        X = df_shuffled.drop(columns=['fraudulent'])
        y = df_shuffled['fraudulent']

        return X,y
    def fit(self, X, y):
        # do not exceed 29 mins

        # preprocessing data
        X = self.preprocess(X)

        # Oversampling for 'fraudulent' = 1 class
        X,y = self.oversample(X,y)

        # Selecting text features to vectorize
        X_text = X['title'] + ' ' + X['description'] + ' '+ X['requirements']

        # Convert text to TF-IDF vectors
        self.tfidf_vectorizer = TfidfVectorizer( stop_words='english',
                                                 max_features=8000,
                                                 max_df=0.7 )

        text_features = self.tfidf_vectorizer.fit_transform(X_text).toarray()

        # Concatenate text features with other features
        X = np.concatenate([text_features, X[['has_company_logo']].values], axis=1)

        sgd = SGDClassifier( class_weight={0:1,1:2},
                             random_state=42 )

        # sgd = SGDClassifier(random_state=42)

        param_grid = {
            'loss': ['hinge', 'perceptron', 'log_loss']
            # 'alpha': [0.0001, 0.001],
            # 'penalty': ['l2', 'l1']
            # 'max_iter': [1000, 3000]
        }

        # Define the custom scorer for F1 score of class 1
        custom_scorer = make_scorer(f1_score, average=None, labels=[1])

        # Use GridSearchCV to search for the best hyperparameters
        self.model = GridSearchCV(estimator=sgd, param_grid=param_grid, cv=5, scoring=custom_scorer)

        self.model.fit(X, y)

        # grid_result = self.model.fit(X, y)
        # print("Best F1 Score for Class 1: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

        return

    def predict(self, X):

        # Applying preprocessing on Test data
        X = self.preprocess(X)
        X_text = X['title'] + ' ' + X['description'] + ' ' + X['requirements']
        text_features = self.tfidf_vectorizer.transform(X_text).toarray()
        X = np.concatenate([text_features, X[['has_company_logo']].values], axis=1)
        
        predictions = self.model.predict(X)
        
        return predictions