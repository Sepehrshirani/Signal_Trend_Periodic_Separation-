import numpy as np

def hankelize_matrix(matrix: np.ndarray) -> np.ndarray:
    rows, cols = matrix.shape
    hankel = np.zeros_like(matrix)
    for m in range(rows):
        for n in range(cols):
            s = m + n
            if 0 <= s <= rows - 1:
                hankel[m, n] = np.mean([matrix[l, s - l] for l in range(s + 1)])
            elif rows <= s <= cols - 1:
                hankel[m, n] = np.mean([matrix[l, s - l] for l in range(rows)])
            elif cols <= s <= cols + rows - 2:
                hankel[m, n] = np.mean([matrix[l, s - l] for l in range(s - cols + 1, rows)])
    return hankel
