from matplotlib.axes._axes import _log as matplotlib_axes_logger
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.datasets import make_blobs


def main(useBKM=True, fit=True, predict=True):
    # Create datasets
    X1, y1 = make_blobs(
        n_samples=200, n_features=2,
        centers=20, cluster_std=0.5,
        shuffle=True, random_state=0
    )
    X2, y2 = make_blobs(
        n_samples=200, n_features=2,
        centers=20, cluster_std=0.5,
        shuffle=True, random_state=2
    )
    KM = BisectingKMeans if useBKM else KMeans
    km = KM(
        n_clusters=10, init='random',
        n_init=10, max_iter=300, 
        tol=1e-04, random_state=0
    )
    y_km1 = km.fit_predict(X1); y_km2 = km.predict(X2)
    colours = np.random.rand(np.amax(y_km2) + 1, 3)
    matplotlib_axes_logger.setLevel('ERROR')
    predict and producePredictResultPlot(km, X2, y_km2, colours)
    fit and produceFitResultPlot(km, X1, y_km1, colours)


def producePredictResultPlot(km, X2, y_km2, colours):
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
    plt.axis("square")
    plt.grid()
    plt.savefig('Predict Results.png')


def produceFitResultPlot(km, X1, y_km1, colours):
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
    plt.axis("square")
    plt.grid()
    plt.savefig('Fit Results.png')

