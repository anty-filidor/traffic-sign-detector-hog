import numpy as np

class SVC:

    def __init__(self, svc, scaler):
        """
        This method creates a SVC for classification
        :param svc: supported vector classifier (sklearn.svm.SVC)
        :param scaler: a feature scaler for SVC (sklearn.preprocessing.StandardScaler)
        """
        self.svc = svc
        self.scaler = scaler

    def predict(self, features):
        """
        This method performs prediction on a given HOG features vector
        :param features: features from HOGExtractor.features()
        :return: prediction vector
        """
        features = self.scaler.transform([features])
        y_pred_proba = self.svc.predict_proba(features)
        label = np.argmax(y_pred_proba, axis=1)
        proba = np.float(y_pred_proba[0][label])
        return label, proba
