"""
Singular Spectrum Analysis (SSA) for EEG Signal Decomposition

This script performs Singular Spectrum Analysis (SSA) on EEG data to decompose
a signal into trend, periodic, and noise components. The implementation includes
trajectory matrix construction, SVD decomposition, and component reconstruction.

Author: Sepehr Shirani
Date: March 25th
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from typing import Tuple, List
from scipy.io import savemat
import os
from datetime import datetime

# Constants
PLOT_STYLE = "seaborn-v0_8"  # Modern and clean plotting style
DEFAULT_WINDOW_RATIO = 0.128  # Ratio of window length to signal length
DEFAULT_COMPONENTS_RATIO = 0.25  # Show first 25% of components by default
OUTPUT_DIR = "output"  # Directory to save results


def ensure_output_dir_exists():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def generate_output_filename(prefix: str, extension: str) -> str:
    """
    Generate timestamped output filename.

    Args:
        prefix: File name prefix
        extension: File extension (without dot)

    Returns:
        Full output path
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUT_DIR, f"{prefix}_{timestamp}.{extension}")


def plot_2d_matrix(matrix: np.ndarray, title: str = "") -> None:
    """
    Plot a 2D matrix with clean formatting.

    Args:
        matrix: 2D numpy array to visualize
        title: Title for the plot
    """
    plt.imshow(matrix, cmap="viridis", aspect="auto")
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.colorbar(label="Value")


def hankelize_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Perform Hankelization (anti-diagonal averaging) on a matrix.

    Args:
        matrix: Input matrix to Hankelize

    Returns:
        Hankelized matrix with same dimensions as input
    """
    rows, cols = matrix.shape
    transpose = False

    # Work with the matrix in "landscape" orientation for easier indexing
    if rows > cols:
        matrix = matrix.T
        rows, cols = cols, rows
        transpose = True

    hankel_matrix = np.zeros((rows, cols))

    for m in range(rows):
        for n in range(cols):
            s = m + n  # Anti-diagonal index

            if 0 <= s <= rows - 1:
                # First triangular section
                hankel_matrix[m, n] = np.mean([matrix[l, s - l] for l in range(s + 1)])
            elif rows <= s <= cols - 1:
                # Middle rectangular section
                hankel_matrix[m, n] = np.mean([matrix[l, s - l] for l in range(rows)])
            elif cols <= s <= cols + rows - 2:
                # Second triangular section
                hankel_matrix[m, n] = np.mean([matrix[l, s - l]
                                               for l in range(s - cols + 1, rows)])

    return hankel_matrix.T if transpose else hankel_matrix


def elementary_to_timeseries(elementary_matrix: np.ndarray) -> np.ndarray:
    """
    Convert elementary matrix to time series through anti-diagonal averaging.

    Args:
        elementary_matrix: Elementary matrix from SVD decomposition

    Returns:
        Reconstructed time series
    """
    reversed_matrix = elementary_matrix[::-1]
    return np.array([reversed_matrix.diagonal(i).mean()
                     for i in range(-elementary_matrix.shape[0] + 1,
                                    elementary_matrix.shape[1])])


def load_eeg_data(filepath: str, channel: int = 5,
                  time_range: Tuple[int, int] = (11200, 12224)) -> Tuple[np.ndarray, float]:
    """
    Load EEG data from EDF file and extract specified channel and time range.

    Args:
        filepath: Path to EDF file
        channel: Channel index to extract
        time_range: Start and end indices for time segment

    Returns:
        Tuple of (signal, sampling_frequency)
    """
    raw = mne.io.read_raw_edf(filepath)
    fs = raw.info['sfreq']
    data, _ = raw[:]
    signal = data[channel, time_range[0]:time_range[1]]
    return signal - np.mean(signal), fs  # Remove DC component


def create_trajectory_matrix(signal: np.ndarray, window_ratio: float = DEFAULT_WINDOW_RATIO) -> np.ndarray:
    """
    Create trajectory matrix (Hankel matrix) from time series.

    Args:
        signal: Input time series
        window_ratio: Ratio of window length to signal length

    Returns:
        Trajectory matrix
    """
    N = len(signal)
    L = int(N * window_ratio)  # Window length
    K = N - L + 1  # Number of columns

    # Create trajectory matrix using stride tricks for efficiency
    strides = (signal.strides[0], signal.strides[0])
    shape = (L, K)
    return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)


def plot_component_contributions(singular_values: np.ndarray, n_components: int = 10) -> None:
    """
    Plot relative and cumulative contributions of singular components.

    Args:
        singular_values: Array of singular values from SVD
        n_components: Number of components to display
    """
    sigma_sumsq = (singular_values ** 2).sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Relative contribution plot
    ax1.plot(singular_values ** 2 / sigma_sumsq * 100, 'o-', lw=2.5)
    ax1.set_xlim(0, n_components - 1)
    ax1.set_title("Relative Contribution of Components")
    ax1.set_xlabel("Component Index ($i$)")
    ax1.set_ylabel("Contribution (%)")
    ax1.grid(True, alpha=0.3)

    # Cumulative contribution plot
    ax2.plot((singular_values ** 2).cumsum() / sigma_sumsq * 100, 'o-', lw=2.5)
    ax2.set_xlim(0, n_components - 1)
    ax2.set_title("Cumulative Contribution of Components")
    ax2.set_xlabel("Component Index ($i$)")
    ax2.set_ylabel("Contribution (%)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def save_results(trend: np.ndarray, periodic: np.ndarray, noise: np.ndarray,
                 time_vector: np.ndarray, sampling_rate: float) -> None:
    """
    Save the decomposition results to files.

    Args:
        trend: Trend component time series
        periodic: Periodic component time series
        noise: Noise component time series
        time_vector: Time axis values
        sampling_rate: Sampling frequency in Hz
    """
    # Save components as .mat files
    mat_data = {
        'trend': trend,
        'periodic': periodic,
        'noise': noise,
        'time': time_vector,
        'fs': sampling_rate
    }

    mat_filename = generate_output_filename("ssa_components", "mat")
    savemat(mat_filename, mat_data)
    print(f"Saved components to {mat_filename}")

    # Save final plot as JPG
    plot_filename = generate_output_filename("ssa_decomposition", "jpg")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', format='jpeg')
    print(f"Saved plot to {plot_filename}")


def main():
    # Set up output directory
    ensure_output_dir_exists()

    # Set plotting style
    plt.style.use(PLOT_STYLE)

    # Load and prepare data
    file_path = "...."
    signal, fs = load_eeg_data(file_path)
    N = len(signal)
    t = np.arange(N) / fs  # Time axis in seconds

    # Create trajectory matrix
    X = create_trajectory_matrix(signal)

    # Plot trajectory matrix
    plt.figure(figsize=(10, 6))
    ax = plt.matshow(X, fignum=0)
    plt.xlabel("$K=N-L+1$ (Columns)")
    plt.ylabel("$L$ (Window Length)")
    plt.colorbar(ax.colorbar, fraction=0.025)
    ax.colorbar.set_label("Amplitude ($\mu V$)")
    plt.title("Trajectory Matrix of EEG Signal", pad=20)
    plt.show()

    # Perform SVD decomposition
    U, Sigma, Vh = np.linalg.svd(X, full_matrices=False)
    V = Vh.T
    d = np.linalg.matrix_rank(X)

    # Create elementary matrices
    X_elem = np.array([Sigma[i] * np.outer(U[:, i], V[:, i]) for i in range(d)])

    # Verify decomposition
    if not np.allclose(X, X_elem.sum(axis=0), atol=1e-10):
        print("Warning: Sum of elementary matrices does not equal original matrix")

    # Plot first few components
    n_components = min(int(DEFAULT_COMPONENTS_RATIO * X.shape[0]), d)

    # Plot component contributions
    plot_component_contributions(Sigma, n_components)

    # Plot reconstructed components
    plt.figure(figsize=(12, 6))

    for i in range(n_components):
        component = elementary_to_timeseries(X_elem[i])
        plt.plot(t, component, lw=2, label=f"Component {i}")

    plt.plot(t, signal, 'k', alpha=0.3, lw=1, label="Original")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude ($\mu V$)")
    plt.title(f"First {n_components} Reconstructed Components")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Group components into meaningful categories
    trend = elementary_to_timeseries(X_elem[0])
    periodic = elementary_to_timeseries(X_elem[1:int(n_components * 0.66)].sum(axis=0))
    noise = elementary_to_timeseries(X_elem[int(n_components * 0.66) + 1:n_components].sum(axis=0))

    # Plot grouped components
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal, 'k', alpha=0.3, label="Original Signal")
    plt.plot(t, trend, label="Trend Component")
    plt.plot(t, periodic, label="Periodic Components")
    plt.plot(t, noise, alpha=0.6, label="Noise Components")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude ($\mu V$)")
    plt.title("SSA Decomposition of EEG Signal")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the results before showing the plot
    save_results(trend, periodic, noise, t, fs)

    plt.show()


if __name__ == "__main__":
    main()