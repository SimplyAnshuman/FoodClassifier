from typing import Optional, Callable
import pandas as pd
import numpy as np
import yaml
import logging as log
from pandas import DataFrame

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import math


class FoodSencClassifier():
    def __init__(self, model_name = 'all-MiniLM-L6-v2'):
        self.st = SentenceTransformer(model_name) 
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def __repr__(self):
        return "Sentence classifier for food/emotional based sentences."

    def _generate_embeddings(self, X):
        """
        Generate Embeddings
        
        Arguments:
        X: Input Sentences

        """
        # features and data
        print("self.embeddings_")
        self.embeddings_ = self.st.encode(X)
        
    def fit(self, X=None, y=None, fit_intercept_=True):
        """
        Training the food classifier
        """
        
        self._generate_embeddings(X)

        if X != None:
#             if len(X.shape) == 1:
#                 X = X.reshape(-1, 1)
            self.features_ = self.embeddings_
        if y != None:
            self.target_ = y
            
        
        self.label_encoder.fit(y)
        train_labels = self.label_encoder.transform(y)

        embedding_list = list(self.features_)
        train_features = self.scaler.fit_transform(self.features_)

        self.sentence_classifier = LogisticRegression()
        self.sentence_classifier.fit(train_features, train_labels)
        
  
    def predict(self, X):
        """Output model prediction.
        """
        self._generate_embeddings(X)
        self.features_ = self.embeddings_
        features = self.scaler.transform(self.features_)
        preds = self.sentence_classifier.predict( features ) 
        self.predicted_ = self.label_encoder.inverse_transform(preds)
        return self.predicted_
