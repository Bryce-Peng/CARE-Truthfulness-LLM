import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

class CenterOfMeanClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, intercept=True, normalize=False, train=False, max_iter=1000, penalty=False, **kwargs):
        self.intercept = intercept
        self.normalize = normalize
        self.train = train
        self.max_iter = max_iter
        self.alpha_ = 1
        self.beta_ = 1
        self.intercept_ = 0
        self.penalty = penalty

        for key, value in kwargs.items():
            setattr(self, key, value)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in X and y must be equal.")
        
        unique_classes = np.unique(y)
        if len(unique_classes) != 2 or not all(cls in [0, 1] for cls in unique_classes):
            raise ValueError("This classifier only supports binary classification with classes 0 and 1.")

        pos = X[y == 1]
        neg = X[y == 0]

        if pos.size == 0 or neg.size == 0:
            raise ValueError("Both positive and negative examples are required.")

        self.pos_mean = np.mean(pos, axis=0)
        self.neg_mean = np.mean(neg, axis=0)

        if self.normalize:
            self.pos_mean = self.pos_mean / np.linalg.norm(self.pos_mean) if np.linalg.norm(self.pos_mean) > 0 else self.pos_mean
            self.neg_mean = self.neg_mean / np.linalg.norm(self.neg_mean) if np.linalg.norm(self.neg_mean) > 0 else self.neg_mean

        if self.intercept:
            self.intercept_ = (np.dot(self.neg_mean, self.neg_mean) - np.dot(self.pos_mean, self.pos_mean)) / 2

        if self.train:
            Z_pos = np.dot(X, self.pos_mean)
            Z_neg = -np.dot(X, self.neg_mean)
            Z = np.column_stack((Z_pos, Z_neg))

            if self.penalty:
                clf = LogisticRegression(random_state=self.random_state, max_iter=self.max_iter, fit_intercept=True).fit(Z, y)  
            else:
                clf = LogisticRegression(random_state=self.random_state, max_iter=self.max_iter, penalty=None, fit_intercept=True).fit(Z, y)  

            self.alpha_, self.beta_ = clf.coef_[0]
            self.intercept_ = clf.intercept_[0]
            print("alpha_, beta_, intercept_ = ", self.alpha_, self.beta_, self.intercept_)
        
        self.coef_ = self.alpha_ * self.pos_mean - self.beta_ * self.neg_mean

        return self

    def decision_function(self, X):
        return np.dot(X, self.coef_) + self.intercept_

    def predict_proba(self, X):
        return np.column_stack([1 - expit(self.decision_function(X)), expit(self.decision_function(X))])

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] >= 0.5).astype(int)
    
