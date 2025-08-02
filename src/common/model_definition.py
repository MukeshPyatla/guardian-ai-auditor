from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
import numpy as np

class TextComplianceModel:
    def __init__(self):
        """A simple Logistic Regression model for text compliance classification."""
        self.model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)

    def fit(self, X, y):
        """Fits the model with data X and labels y."""
        self.model.fit(X, y)

    def predict(self, X):
        """Makes predictions on new data X."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Returns probability estimates for the classes."""
        return self.model.predict_proba(X)

    def get_parameters(self):
        """Returns the model's coefficients and intercept as a dictionary."""
        # Flatten coef array to a 1D list for easier serialization in FL
        return {'coef': self.model.coef_.flatten().tolist(), 'intercept': self.model.intercept_.tolist()}

    def set_parameters(self, params):
        """Sets the model's parameters from a dictionary."""
        # Reshape coefficients back to the correct format
        self.model.coef_ = np.array(params['coef']).reshape(1, -1)
        self.model.intercept_ = np.array(params['intercept'])

class SensorAnomalyModel:
    def __init__(self):
        """An Isolation Forest model for unsupervised anomaly detection."""
        self.model = IsolationForest(random_state=42, contamination='auto')

    def fit(self, X):
        """Fits the model with data X."""
        self.model.fit(X)

    def predict(self, X):
        """Predicts if an instance is an anomaly (-1) or normal (1)."""
        return self.model.predict(X)

    def get_parameters(self):
        """This model's parameters are not federated for simplicity in this demo."""
        return {}

    def set_parameters(self, params):
        """Setting parameters is not applicable for this simplified model."""
        pass