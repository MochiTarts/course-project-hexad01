from sklearn.cluster import KMeans
from sklearn.cluster import BisectingKMeans
import math
import numpy as np

SEP = "=" * 100
data = {
    "Dataset 1" : {
        "K" : 2, "n" : 1,
        "fit" : [[0.2],[0.8]],
        "pre" : [[0.3], [0.9]],
        'tra' : [[0.34], [0.94]],

        "expected": {
            "labels" : [1, 0],
            "centers" : [[0.8], [0.2]]
        }
    },
    "Dataset 2" : {
        "K": 2, "n" : 2,
        "fit" : [[0.8,  0.5],
                [0.77, 0.58],
                [0.34, 0.4 ],
                [0.2,  0.43]],
        "pre" : [[0.26, 0.42],
                [0.70, 0.47],
                [0.47, 0.83]],
        "tra" : [[0.83, 0.92],
                [0.27, 0.18],
                [0.49, 0.03]],
        
        "expected" : {
            "labels" : [0, 0, 1, 1],
            "centers" : [[0.785, 0.54 ],
                        [0.27,  0.415]]
        }
    },
    "Dataset 3" : {
        "K": 7, "n" : 2,
        "fit" : [[0.80, 0.50],
                [0.74, 0.49],
                [0.77, 0.58],
                [0.34, 0.40],
                [0.20, 0.43],
                [0.25, 0.38],
                [0.63, 0.82]],
        "pre" : [[0.26, 0.42],
                [0.70, 0.47],
                [0.47, 0.83]],
        "tra" : [[0.34, 0.43],
                [0.94, 0.34]],

        "expected" : {
            "labels": [5, 0, 3, 4, 6, 1, 2],
            "centers": [[0.74, 0.49],
                        [0.25, 0.38],
                        [0.63, 0.82],
                        [0.77, 0.58],
                        [0.34, 0.4 ],
                        [0.8,  0.5 ],
                        [0.2,  0.43]]
        }
    },
    "Dataset 4" : {
        "K": 3, "n" : 2,
        "fit" : [[0.80, 0.50],
                [0.74, 0.49],
                [0.77, 0.58], 
                [0.34, 0.40],
                [0.20, 0.43],
                [0.25, 0.38],
                [0.63, 0.82]], 
        "pre" : [[0.26, 0.42],
                [0.70, 0.47],
                [0.47, 0.83]],
        
        "expected" : {
            "labels": None,
            "centers": None
        }
    }
}


def testAll(
    useBKM=True, 
    fit=True, 
    predict=True, 
    transform=True, 
    score=True
    ):
    for set in data.keys():

        if set == "Dataset 4": # Take this out when DS4 is done
            break

        print (SEP + "\n" + set + '\n' + SEP)
        dataset = data[set]; k = dataset["K"]
        km = BisectingKMeans(n_clusters=k) if useBKM else KMeans(n_clusters=k)
        X = np.array(dataset["fit"])

        if fit:
            print("Testing fit()")
            km.fit(X)
            print("\nLabels:\n", km.labels_)
            print("\nCluster Centers:\n", km.cluster_centers_)
            print(verifyFitResults(dataset, km))

            if predict:
                print("Testing predict()")
                predictOut = km.predict(np.array(dataset["pre"]))
                print("\nPredict:\n", predictOut)
                print(verifyPredictResults(dataset, predictOut, km))

            if transform:
                print("Testing transform()")
                transformOut = km.transform(dataset["tra"])
                print("\nTransform:\n", transformOut)
                print(verifyTransformResults(dataset, transformOut, km))
            
            if score:
                print("Testing score()")
        
        if fit and predict:
            print("Testing fit_predict()")
            predictOut = km.fit_predict(dataset["fit"])
            print("\nFit Predict:\n", predictOut)
            print(verifyFitPredictResults(useBKM, dataset, predictOut, km))
        
        if fit and transform:
            print("Testing fit_transform()")
            transformOut = km.fit_transform(dataset["fit"])
            print("\nFit Transform:\n", transformOut)
            print(verifyFitTransformResults(dataset, transformOut, km))

        
def verifyFitResults(dataset, km):
    expected = dataset["expected"]
    expLabels = expected["labels"] 
    expCenters = np.array(expected["centers"])
    # Condition 1: cluster centers contain the same values, but not necessarily in the same order
    received = [float('%.3f' % elem) for elem in list(np.sort(km.cluster_centers_.flat))]
    expected = [float('%.3f' % elem) for elem in list(np.sort(expCenters.flat))]
    # Condition 2: Contains the same number of labels, and labels are identical
    return received == expected and set(expLabels) == set(km.labels_)


def verifyPredictResults(dataset, results, km):
    data = np.array(dataset["pre"])
    # Condition 1: Each data point belongs to their correct cluster
    for id in range(len(data)):
        expected_min = sum(
            [(data[id][n] - km.cluster_centers_[
                results[id]][n]) ** 2 for n in range(km.n_features_in_)])

        for point in km.cluster_centers_:
            local_min = sum(
                [(data[id][n] - point[n]) ** 2 for n in range(km.n_features_in_)])
            if (local_min < expected_min):
                return False
    # Condition 2: Checks that result is the correct size
    return len(results) == len(data)


def verifyFitPredictResults(useBKM, dataset, results, km):
    data = np.array(dataset["fit"])

    # Condition 1: Checks that results is the correct size
    if (len(results) != len(data)):
        return False

    # Condition 2: Each data point belongs to their correct cluster
    for id in range(len(data)):
        expected_min = sum(
            [(data[id][n] - km.cluster_centers_[
                results[id]][n]) ** 2 for n in range(km.n_features_in_)])

        for point in km.cluster_centers_:
            local_min = sum(
                [(data[id][n] - point[n]) ** 2 for n in range(km.n_features_in_)])
            if (local_min < expected_min):
                return False

    # Condition 3: fit_predict() == fit().predict()
    KM2 = BisectingKMeans if useBKM else KMeans
    expected = list(KM2(n_clusters=km.n_clusters).fit(data).predict(data))
    return [reMap(list(results))[i] for i in results] == [
        reMap(expected)[i] for i in expected]


def verifyTransformResults(dataset, results, km, decimals=5):
    data = dataset["tra"]
    # Condition 1: Each sub result is the distance from a data point to a center
    result = [([round(j, decimals) for j in i]) for i in results]
    for id in range(len(data)):
        s = [round((math.sqrt(sum(
            [(data[id][n] - center[n]) ** 2 for n in range(
                km.n_features_in_)]))), decimals) for center in km.cluster_centers_]
        if s != result[id]:
            return False

    # Condition 2: Each result is the corrrect size
    return len(results) == len(data)


def verifyFitTransformResults(dataset, results, km):
    data = dataset["fit"]
    decimals = 5

    # Condition 1: Each result is the correct size
    if (len(results) != len(data)):
        return False

    # Condition 2: Each sub result is the distance from a data point to a center
    result = [([round(j, decimals) for j in i]) for i in results]
    for id in range(len(data)):
        s = [round((math.sqrt(sum([(data[id][n] - center[n])**2 for n in range(km.n_features_in_)]))), decimals) for center in km.cluster_centers_]
        if s != result[id]:
            return False

    # Condition 3: fit_transform() == fit().transform()
    expected = KMeans(n_clusters=km.n_clusters).fit(data).transform(data)
    return [(set([j for j in i])) for i in results] == [(set([j for j in i])) for i in expected]


def reMap(arr):
    dict = {}
    counter = 0
    for i in arr:
        if i not in dict:
            dict[i] = counter
            counter += 1
    return dict

   
if __name__ == '__main__':
    testAll()
