from matplotlib.axes._axes import _log as matplotlib_axes_logger
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.datasets import make_blobs

indent = ' ' * 4
SEP = '_' * 100 + '\n'


def executeTest(useBKM=True, fit=True, predict=True, verbose=True):
    if verbose:
        print(SEP)
        print(("Bisecting K-Means" if useBKM else "Kmeans"), "Acceptance Testing")
    KM = BisectingKMeans if useBKM else KMeans
    
    # Create datasets
    verbose and print(SEP + "[1] Creating Dataset 1\n")
    X1, y1 = make_blobs(
        n_samples=200, n_features=2,
        centers=20, cluster_std=0.5,
        shuffle=True, random_state=0
    )
    verbose and print("X1 =", X1)
    
    verbose and print(SEP + "[2] Creating Dataset 2\n")
    X2, y2 = make_blobs(
        n_samples=200, n_features=2,
        centers=20, cluster_std=0.5,
        shuffle=True, random_state=2
    )
    verbose and print("X2 =", X2)
    
    verbose and print(SEP + "[3] Instantializing:\n")
    verbose and print(indent, "Setting:", "n_clusters=10,", "init='random',", 
        "n_init=10,", "max_iter=300,", "tol=1e-04,", "random_state=0")
    km = KM(
        n_clusters=10, init='random',
        n_init=10, max_iter=300, 
        tol=1e-04, random_state=0
    )
    
    verbose and print(SEP + "[4] Running\n")
    y_km1 = km.fit_predict(X1); y_km2 = km.predict(X2)
    colours = np.random.rand(np.amax(y_km2) + 1, 3)
    matplotlib_axes_logger.setLevel('ERROR')
    predict and producePredictResultPlot(km, X2, y_km2, colours, verbose)
    fit and produceFitResultPlot(km, X1, y_km1, colours, verbose)
    verbose and print(
        SEP + "[5] Finished. Please check your file manager for the resulting plots.")


def producePredictResultPlot(km, X2, y_km2, colours, verbose):
    verbose and print(indent, "Plotting Predict Results...")
    plt.figure(1)
    for i in np.unique(y_km2):
        plt.scatter(
            X2[y_km2 == i, 0], X2[y_km2 == i, 1],
            s=50, c=np.array(colours[i]),
            marker='s', edgecolor='black',
            label='Cluster: ' + str(i)
        )
    # Centroid plotting
    for i in range(km.cluster_centers_.shape[0]):
        plt.scatter(
            km.cluster_centers_[i, 0], km.cluster_centers_[i, 1],
            s=250, marker='*',
            c=np.array(colours[i]), edgecolor='black',
            label=': Centroid'
        )
    plt.legend(scatterpoints=1)
    plt.title("Results After predict() Invocation")
    plt.axis("square")
    plt.grid()
    plt.savefig('AT Predict Results.png')


def produceFitResultPlot(km, X1, y_km1, colours, verbose):
    verbose and print(indent, "Plotting Fit Results...")
    plt.figure(2)
    for i in np.unique(y_km1):
        plt.scatter(
            X1[y_km1 == i, 0], X1[y_km1 == i, 1],
            s=50, c=np.array(colours[i]),
            marker='s', edgecolor='black',
            label='cluster ' + str(i)
        )
    # plot the centroids
    for i in range(km.cluster_centers_.shape[0]):
        plt.scatter(
            km.cluster_centers_[i, 0], km.cluster_centers_[i, 1],
            s=250, marker='*',
            c=np.array(colours[i]), edgecolor='black',
            label='centroids'
        )
    plt.title("Results After fit() Invocation")
    plt.axis("square")
    plt.grid()
    plt.savefig('AT Fit Results.png')

executeTest()
