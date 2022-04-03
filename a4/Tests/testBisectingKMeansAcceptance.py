from matplotlib.axes._axes import _log as matplotlib_axes_logger
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.datasets import make_blobs

indent = ' ' * 4
SEP = '_' * 100 + '\n'
LINE = '=' * 150


def executeTest(case, desc, case_num, useBKM=False, fit=True, predict=True,
                verbose=True, n_samples=200, n_features=2, fitted_clusters=20,
                predicted_clusters=20, n_clusters=20, n_init=10, max_iter=300,
                fit_random_state=0, predict_random_state=1):

    print(LINE)
    print(case, ":", desc)
    print(LINE)

    mode = "Bisecting K-Means" if useBKM else "K-Means"
    if verbose:
        print(SEP)
        print(mode, "Acceptance Testing")
    KM = BisectingKMeans if useBKM else KMeans

    # Create datasets
    verbose and print(SEP + "[1] Creating Dataset 1\n")
    X1, y1 = make_blobs(
        n_samples=n_samples, n_features=n_features,
        centers=fitted_clusters, cluster_std=0.5,
        shuffle=True, random_state=fit_random_state
    )
    #verbose and print("X1 =", X1)

    verbose and print(SEP + "[2] Creating Dataset 2\n")
    X2, y2 = make_blobs(
        n_samples=n_samples, n_features=n_features,
        centers=predicted_clusters, cluster_std=0.5,
        shuffle=True, random_state=predict_random_state
    )
    #verbose and print("X2 =", X2)

    verbose and print(SEP + "[3] Instantializing:\n")
    verbose and print(indent, "Setting:",
                      "n_clusters=" + str(n_clusters) + ",",
                      "init='random',",
                      "n_init=" + str(n_init) + ",",
                      "max_iter=" + str(max_iter) + ",",
                      "tol=1e-04,",
                      "random_state=" + str(fit_random_state))
    km = KM(
        n_clusters=n_clusters, init='random',
        n_init=n_init, max_iter=max_iter,
        tol=1e-04, random_state=fit_random_state
    )

    verbose and print(SEP + "[4] Running\n")
    y_km1 = km.fit_predict(X1)
    y_km2 = km.predict(X2)
    colours = np.random.rand(np.amax(km.labels_) + 1, 3)
    matplotlib_axes_logger.setLevel('ERROR')
    predict and producePredictResultPlot(km, X2, y_km2, colours, mode,
                                         verbose, case_num)
    fit and produceFitResultPlot(km, X1, y_km1, colours, mode,
                                 verbose, case_num)
    verbose and print(
        SEP +
        "[5] Finished. Please check your file manager " +
        "for the resulting plots.")


def producePredictResultPlot(km, X2, y_km2, colours, mode, verbose, case_num):
    verbose and print(indent, "Plotting", mode, "Predict Results...")
    plt.figure(2 * case_num + 1)
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
    plt.legend(scatterpoints=1, bbox_to_anchor=(1, 1))
    plt.title(mode + " Results After predict() Invocation")
    plt.axis("square")
    plt.grid()
    plt.savefig(mode + " Predict Results " + str(case_num) + ".png")


def produceFitResultPlot(km, X1, y_km1, colours, mode, verbose, case_num):
    verbose and print(indent, "Plotting", mode, "Fit Results...")
    plt.figure(2 * case_num + 2)
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
    plt.title(mode + " Results After fit() Invocation")
    plt.axis("square")
    plt.grid()
    plt.savefig(mode + " Fit Results " + str(case_num) + ".png")


testCases = {
    'Test1 : Small number of samples and features, large number of clusters.': {
        'desc': 'n_samples=200, n_features=2, fitted_clusters=20, ' + \
                'predicted_clusters=20, n_clusters=20, n_init=10, ' + \
                'max_iter=300, fit_random_state=0, predict_random_state=1',
        'args': {
            'n_samples': 200,
            'n_features': 2,
            'fitted_clusters': 20,
            'predicted_clusters': 20,
            'n_clusters': 20,
            'n_init': 10,
            'max_iter': 300,
            'fit_random_state': 0,
            'predict_random_state': 1
        }
    },
    'Test2 : Fewer generated clusters than n_clusters.': {
        'desc': 'n_samples=2000, fitted_clusters=20, ' + \
                'predicted_clusters=20, n_clusters=30',
        'args': {
            'n_samples': 2000,
            'fitted_clusters': 20,
            'predicted_clusters': 20,
            'n_clusters': 30,
        }
    }
}


def testAll(cases, useBKM=False, verbose=True):
    case_num = 0
    for case in cases.keys():
        testCase = cases[case]
        desc = testCase['desc']
        args = testCase['args']
        #n_samples = testCase['args']['n_samples']
        #n_features = testCase['args']['n_features']
        #fitted_clusters = testCase['args']['fitted_clusters']
        #predicted_clusters = testCase['args']['predicted_clusters']
        #n_clusters = testCase['args']['n_clusters']
        #n_init = testCase['args']['n_init']
        #max_iter = testCase['args']['max_iter']
        #fit_random_state = testCase['args']['fit_random_state']
        #predict_random_state = testCase['args']['predict_random_state']
        #executeTest(case, desc, useBKM=useBKM, n_samples=n_samples,
        #            n_features=n_features,
        #            fitted_clusters=fitted_clusters,
        #            predicted_clusters=predicted_clusters,
        #            n_clusters=n_clusters,
        #            n_init=n_init,
        #            max_iter=max_iter,
        #            fit_random_state=fit_random_state,
        #            predict_random_state=predict_random_state)
        executeTest(case,
                    desc,
                    case_num,
                    useBKM=useBKM,
                    fit=True,
                    predict=True,
                    verbose=verbose,
                    **args)
        case_num = case_num + 1


def main():
    print("This script is configured to compare" +
          "plotted results between K-Means and Bisecting K-Means.")
    prompt = "Please select one of the following:" + \
             "\n[1] K-Means\n[2] Bisecting K-Means\n"
    selection = input(prompt)
    useBKM = (selection == "2")
    testAll(cases=testCases, useBKM=useBKM, verbose=True)


main()
