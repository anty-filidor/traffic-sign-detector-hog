import numpy as np


class SVC:

    def __init__(self, svc, scaler):
        """
        
        :param svc:
        :param scaler:
        """
        self.svc = svc
        self.scaler = scaler

    def predict(self, f):
        """

        :param f:
        :return:
        """
        f = self.scaler.transform([f])
        r = self.svc.predict(f)
        # print(type(r), len(r), '\n', r, '\n',)
        return np.int(r[0])

