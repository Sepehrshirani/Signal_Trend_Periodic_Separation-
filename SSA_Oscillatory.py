
# Make sure you have the necessary packages.
import numpy as np
import matplotlib.pyplot as plt
import mne
import scipy.signal as signal

# A simple little 2D matrix plotter, excluding x and y labels.
def plot_2d(m, title=""):
    plt.imshow(m)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)


def Hankelise(X):
    L, K = X.shape
    transpose = False
    if L > K:
        X = X.T
        L, K = K, L
        transpose = True
    HX = np.zeros((L, K))

    for m in range(L):
        for n in range(K):
            s = m + n
            if 0 <= s <= L - 1:
                for l in range(0, s + 1):
                    HX[m, n] += 1 / (s + 1) * X[l, s - l]
            elif L <= s <= K - 1:
                for l in range(0, L - 1):
                    HX[m, n] += 1 / (L - 1) * X[l, s - l]
            elif K <= s <= K + L - 2:
                for l in range(s - K + 1, L):
                    HX[m, n] += 1 / (K + L - s - 1) * X[l, s - l]
    if transpose:
        return HX.T
    else:
        return HX


"""For anti-diagonals averaging of the given elementary matrix, X_i, and returning a time series."""
def X_to_TS(X_i):
    X_rev = X_i[::-1]
    return np.array([X_rev.diagonal(i).mean() for i in range(-X_i.shape[0]+1, X_i.shape[1])])

#Choose the correct directory
file_path = "/Users/sepehrshirani/Desktop/test.edf"
raw = mne.io.read_raw_edf(file_path)
# Extract sampling frequency
fs = raw.info['sfreq']
# Get the data as a NumPy array
data, times = raw[:]

#Choosing channel and time
F=data[5,11200:12224]

#Setting up values
F=F-np.mean(F)
N = len(F)
t = np.arange(0,N)
L = int(N*0.128)  # Window length
K = N - L + 1 # The number of columns in the trajectory matrix.
# Create the trajectory matrix
X = np.column_stack([F[i:i+L] for i in range(0,K)])


ax = plt.matshow(X)
plt.xlabel("$L:$ Window Length")
plt.ylabel("$K=N-L+1$")
plt.colorbar(ax.colorbar, fraction=0.025)
ax.colorbar.set_label("Value")
plt.title("The Trajectory Matrix of the Signal")
plt.show()

d = np.linalg.matrix_rank(X) # The intrinsic dimensionality of the trajectory space.
U, Sigma, V = np.linalg.svd(X)
V = V.T
X_elem = np.array( [Sigma[i] * np.outer(U[:,i], V[:,i]) for i in range(0,d)] )

# Quick sanity check: the sum of all elementary matrices in X_elm should be equal to X, to within a
if not np.allclose(X, X_elem.sum(axis=0), atol=1e-10):
    print("WARNING: The sum of X's elementary matrices is not equal to X!")


n = min(20, d)
for i in range(n):
    plt.subplot(4,5,i+1)
    title = "X_{" + str(i) + "}"
    plot_2d(X_elem[i], title)
plt.show()


sigma_sumsq = (Sigma**2).sum()
fig, ax = plt.subplots(1, 2, figsize=(14,5))
ax[0].plot(Sigma**2 / sigma_sumsq * 100, lw=2.5)
ax[0].set_xlim(0,19)
ax[0].set_title("Relative Contribution to Trajectory Matrix")
ax[0].set_xlabel("$i$")
ax[0].set_ylabel("Contribution (%)")
ax[1].plot((Sigma**2).cumsum() / sigma_sumsq * 100, lw=2.5)
ax[1].set_xlim(0,19)
ax[1].set_title("Cumulative Contribution to Trajectory Matrix")
ax[1].set_xlabel("$i$")
ax[1].set_ylabel("Contribution (%)")
plt.show()


n = min(d, 20)
for j in range(0,n):
    plt.subplot(4,5,j+1)
    title = r"$\tilde{\mathbf{X}}_{" + str(j) + "}$"
    plot_2d(Hankelise(X_elem[j]), title)
plt.tight_layout()
plt.plot()



fig = plt.subplot()
# Convert elementary matrices straight to a time series - no need to construct any Hankel matrices.
for i in range(n):
    F_i = X_to_TS(X_elem[i])
    fig.axes.plot(t, F_i, lw=2)

fig.axes.plot(t, F, alpha=1, lw=1)
fig.set_xlabel("$t$")
fig.set_ylabel(r"$\tilde{F}_i(t)$")
legend = [r"$\tilde{F}_{%s}$" %i for i in range(n)] + ["$F$"]
fig.set_title(f"The First {n} Components of the original signal")
fig.legend(legend, loc=(1.05,0.1));
plt.show()


# Assemble the grouped components of the time series.
F_trend = X_to_TS(X_elem[0])
F_periodic = X_to_TS(X_elem[1:18].sum(axis=0))
F_noise = X_to_TS(X_elem[19:].sum(axis=0))

# Plot the toy time series and its separated components on a single plot.
plt.plot(t,F, lw=1,label='Original')
plt.plot(t, F_trend,label='Trend')
plt.plot(t, F_periodic,label='Periodic')
plt.plot(t, F_noise, alpha=0.5,label='Noise')
plt.xlabel("t")
plt.ylabel(r"$\tilde{F}^{(j)}$")
groups = ["trend", "periodic", "noise"]
plt.legend()
plt.title("Grouped Time Series Components")
plt.show()

# A list of tuples so we can create the next plot with a loop.
components = [("Trend", F_trend),
              ("Periodic 1",  F_periodic),
              ("Noise", F_noise)]

# Plot the separated components and original components together.
fig = plt.figure()
n=1
for name, ssa_comp in components:
    ax = fig.add_subplot(2,2,n)
    ax.plot(t, ssa_comp)
    ax.set_title(name, fontsize=16)
    ax.set_xticks([])
    n += 1
plt.show()


