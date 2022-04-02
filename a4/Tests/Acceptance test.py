from sklearn.cluster import KMeans
from sklearn.cluster import BisectingKMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

# create dataset
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

plt.figure(1)

# plot
plt.scatter(
   X2[:, 0], X2[:, 1],
   c='white', marker='o',
   edgecolor='black', s=50
)


km = BisectingKMeans(
#km = KMeans(
    n_clusters=10, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km1 = km.fit_predict(X1)
y_km2 = km.predict(X2)

colours = np.random.rand(np.amax(y_km2) + 1, 3)

for i in np.unique(y_km2):
    plt.scatter(
    X2[y_km2 == i, 0], X2[y_km2 == i, 1],
    s=50, c=colours[i],
    marker='s', edgecolor='black',
    label='cluster ' + str(i)
    )

# plot the centroids
for i in range(km.cluster_centers_.shape[0]):
    plt.scatter(
        km.cluster_centers_[i, 0], km.cluster_centers_[i, 1],
        s=250, marker='*',
        c=colours[i], edgecolor='black',
        label='centroids'
    )
plt.legend(scatterpoints=1)
plt.axis("square")
plt.grid()
plt.savefig('predicted.png')

plt.figure(2)

plt.scatter(
   X1[:, 0], X1[:, 1],
   c='white', marker='o',
   edgecolor='black', s=50
)


for i in np.unique(y_km1):
    plt.scatter(
    X1[y_km1 == i, 0], X1[y_km1 == i, 1],
    s=50, c=colours[i],
    marker='s', edgecolor='black',
    label='cluster ' + str(i)
    )

# plot the centroids
for i in range(km.cluster_centers_.shape[0]):
    plt.scatter(
        km.cluster_centers_[i, 0], km.cluster_centers_[i, 1],
        s=250, marker='*',
        c=colours[i], edgecolor='black',
        label='centroids'
    )
#plt.legend(scatterpoints=1)
plt.axis("square")
plt.grid()
plt.savefig('fit.png')