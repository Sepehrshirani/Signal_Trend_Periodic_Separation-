import numpy as np
from .hankel import hankelize_matrix

class SSA:
    def __init__(self, signal: np.ndarray, window_ratio: float = 0.128):
        self.signal = signal
        self.window_ratio = window_ratio

    def _create_trajectory_matrix(self) -> np.ndarray:
        N = len(self.signal)
        L = int(N * self.window_ratio)
        K = N - L + 1
        return np.column_stack([self.signal[i:i+L] for i in range(K)])

    def decompose(self) -> tuple:
        X = self._create_trajectory_matrix()
        X_hankel = hankelize_matrix(X)
        U, Sigma, Vh = np.linalg.svd(X_hankel, full_matrices=False)
        
        # Simple component grouping
        trend = U[:, 0] @ np.diag(Sigma[:1]) @ Vh[:1, :]
        periodic = U[:, 1:3] @ np.diag(Sigma[1:3]) @ Vh[1:3, :]
        noise = X - trend - periodic
        
        return (
            np.mean(trend, axis=1), 
            np.mean(periodic, axis=1),
            np.mean(noise, axis=1)
