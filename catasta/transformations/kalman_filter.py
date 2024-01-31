import numpy as np

from .transformation import Transformation

import numpy as np


class KalmanFilter(Transformation):
    def __init__(self, *,
                 F: np.ndarray,
                 H: np.ndarray,
                 B: int | None = None,
                 Q: np.ndarray | None = None,
                 R: np.ndarray | None = None,
                 P: np.ndarray | None = None,
                 x0: np.ndarray | None = None,
                 ) -> None:
        self.n: int = F.shape[1]
        self.m: int = H.shape[1]

        self.F: np.ndarray = F
        self.H: np.ndarray = H
        self.B: int = 0 if B is None else B
        self.Q: np.ndarray = np.eye(self.n) if Q is None else Q
        self.R: np.ndarray = np.eye(self.n) if R is None else R
        self.P: np.ndarray = np.eye(self.n) if P is None else P
        self.x: np.ndarray = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u: int = 0) -> np.ndarray:
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        return self.x

    def update(self, z: int) -> None:
        y = z - np.dot(self.H, self.x)

        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y)

        I = np.eye(self.n)

        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        predictions: list[np.ndarray] = []
        for z in x:
            predictions.append(np.dot(self.H, self.predict())[0])
            self.update(z)

        return np.array(predictions).flatten()
