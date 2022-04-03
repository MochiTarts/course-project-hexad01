from sklearn.cluster import KMeans
from sklearn.cluster import BisectingKMeans
import math
import numpy as np

SEP = "=" * 100
DEP = "." * 100
DECIMALS = 5

data = {
    "Dataset 1" : {
        "description" : "2 datapoints and 2 cluster",
        "K" : 2, "n" : 1,
        "fit" : [[0.2],[0.8]],
        "pre" : [[0.3], [0.9]],
        'tra' : [[0.34], [0.94]],

        "expected": {
            "labels" : [1, 0],
            "centers" : [[0.8], [0.2]],
            "score": -0.02
        }
    },
    "Dataset 2" : {
        "description" : "4 datapoints and 2 cluster",
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
                        [0.27,  0.415]],
            "score": -0.195575
        }
    },
    "Dataset 3" : {
        "description" : "7 datapoints and 7 cluster",
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
                        [0.2,  0.43]],
            "score": -0.0294
        }
    },
    "Dataset 4" : {
        "description" : "7 datapoints and 3 cluster",
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
        "tra" : [[0.34, 0.43],
                [0.94, 0.34]],

        "expected" : {
            "labels": [0, 0, 0, 2, 2, 2, 1],
            "centers": [[0.263, 0.403],
                        [0.77, 0.523],
                        [0.63, 0.82]],
            "score": -0.03373
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
        print('\n' * 2)
        print (SEP + "\n" + set + " : " + data[set]["description"] + '\n' + SEP)

        dataset = data[set]; k = dataset["K"]
        print("Input:")
        print("\tK = " + str(dataset["K"]))
        print("\tfit on X = " + str(dataset["fit"]))
        print("\tpre on X = " + str(dataset["pre"]))
        print("\ttra on X = " + str(dataset["tra"]))
        print(SEP)

        km = BisectingKMeans(n_clusters=k) if useBKM else KMeans(n_clusters=k)
        X = np.array(dataset["fit"])

        if fit:
            print("Test 1: fit()")
            km.fit(X)
            print("    Output:")
            print("\tLabels: ", list(km.labels_))
            print("\tCluster Centers: ", [[round(j, DECIMALS) for j in i] for i in km.cluster_centers_])
            print("    Test Results:")
            verifyFitResults(dataset, km)
            print(DEP)

            if predict:
                print("Test 2: predict()")
                predictOut = km.predict(np.array(dataset["pre"]))
                print("    Output:")
                print("\tPredict: "+ str(list(predictOut)))
                print("    Test Results:")
                verifyPredictResults(dataset, predictOut, km)
                print(DEP)

            if transform:
                print("Test 3: transform()")
                transformOut = km.transform(dataset["tra"])
                print("    Output:")
                print("\tTransform: ", [[round(j, DECIMALS) for j in i] for i in transformOut])
                print("    Test Results:")
                verifyTransformResults(dataset, transformOut, km)
                print(DEP)

            if score:
                print("Test 4: score()")
                scoreOut = km.score(dataset["pre"])
                print("    Output:")
                print("\tScore: ", round(scoreOut, DECIMALS))
                print("    Test Results:")
                verifyScoreResults(dataset, scoreOut)
                print(DEP)

        if fit and predict:
            print("Test 5: fit_predict()")
            predictOut = km.fit_predict(dataset["fit"])
            print("    Output:")
            print("\tFit Predict: ", list(predictOut))
            print("    Test Results:")
            verifyFitPredictResults(useBKM, dataset, predictOut, km)
            print(DEP)

        if fit and transform:
            print("Test 6: fit_transform()")
            transformOut = km.fit_transform(dataset["fit"])
            print("    Output:")
            print("\tFit Transform: ", [[round(j, DECIMALS) for j in i] for i in transformOut]) if set != "Dataset 5" else print("\tFit Transform: ...")
            print("    Test Results:")
            verifyFitTransformResults(useBKM, dataset, transformOut, km)


def verifyFitResults(dataset, km):
    expected = dataset["expected"]
    expLabels = expected["labels"]
    expCenters = np.array(expected["centers"])

    # Condition 1: Cluster centers contain the same values, but not necessarily in the same order
    received = [float('%.3f' % elem) for elem in list(np.sort(km.cluster_centers_.flat))]
    expected = [float('%.3f' % elem) for elem in list(np.sort(expCenters.flat))]
    try :
        assert received == expected
        print("\tPASS Test 1.1: Datapoints belong to the correct cluster.")
    except :
        print("\tFAIL Test 1.1: Datapoint found in the incorrect cluster.")

    # Condition 2: Datapoints belong to the correct cluster
    try :
        assert [reMap(list(expLabels))[i] for i in expLabels] == [
            reMap(km.labels_)[i] for i in km.labels_]
        print("\tPASS Test 5.3: Datapoints belong to their correct cluster.")
    except :
        print("\tFAIL Test 5.3: Datapoints in incorrect cluster.")

    # Condition 3: Contains the same number of labels, and labels are identical
    try :
        assert set(expLabels) == set(km.labels_)
        print("\tPASS Test 1.2: Correct dimensions for result.")
    except :
        print("\tFAIL Test 1.2: Incorrect dimensions for result.")




def verifyPredictResults(dataset, results, km):
    pre_data = np.array(dataset["pre"])
    # Condition 1: Checks that result is the correct size
    try :
        assert len(results) == len(pre_data)
        print("\tPASS Test 2.1: Correct dimensions for result.")
    except :
        print("\tFAIL Test 2.1: Incorrect dimensions for result.")


    # Condition 2: Each data point belongs to their correct cluster
    try :
        for id in range(len(pre_data)):
            expected_min = sum(
                [(pre_data[id][n] - km.cluster_centers_[
                    results[id]][n]) ** 2 for n in range(km.n_features_in_)])

            for point in km.cluster_centers_:
                local_min = sum(
                    [(pre_data[id][n] - point[n]) ** 2 for n in range(km.n_features_in_)])
                assert local_min >= expected_min
        print("\tPASS Test 2.2: Correct values for result.")
    except :
        print("\tFAIL Test 2.2: Incorrect values for result.")


def verifyFitPredictResults(useBKM, dataset, results, km):
    fit_data = np.array(dataset["fit"])

    # Condition 1: Checks that results is the correct size
    try :
        assert len(results) == len(fit_data)
        print("\tPASS Test 5.1: Correct dimensions for result.")
    except :
        print("\tFAIL Test 5.1: Incorrect dimensions for result.")

    # Condition 2: Each data point belongs to their correct cluster
    try :
        for id in range(len(fit_data)):
            expected_min = sum(
                [(fit_data[id][n] - km.cluster_centers_[
                    results[id]][n]) ** 2 for n in range(km.n_features_in_)])

            for point in km.cluster_centers_:
                local_min = sum(
                    [(fit_data[id][n] - point[n]) ** 2 for n in range(km.n_features_in_)])
                assert local_min >= expected_min
        print("\tPASS Test 5.2: Correct values for result.")
    except :
        print("\tFAIL Test 5.2: Incorrect values for result.")

    # Condition 3: fit_predict() == fit().predict()
    KM2 = BisectingKMeans if useBKM else KMeans
    expected = list(KM2(n_clusters=km.n_clusters).fit(fit_data).predict(fit_data))
    try :
        assert [reMap(list(results))[i] for i in results] == [
            reMap(expected)[i] for i in expected]
        print("\tPASS Test 5.3: fit_predict() and fit().predict() are equivalent.")
    except :
        print("\tFAIL Test 5.3: fit_predict() differs from fit().predict().")


def verifyTransformResults(dataset, results, km):
    tra_data = dataset["tra"]
    # Condition 1: Each result is the correct size
    try :
        assert len(results) == len(tra_data)
        print("\tPASS Test 3.1: Correct dimensions for result.")
    except :
        print("\tFAIL Test 3.1: Incorrect dimensions for result.")

    # Condition 2: Each sub result is the distance from a data point to a center
    try :
        result = [([round(j, DECIMALS) for j in i]) for i in results]
        for id in range(len(tra_data)):
            s = [round((math.sqrt(sum(
                [(tra_data[id][n] - center[n]) ** 2 for n in range(
                    km.n_features_in_)]))), DECIMALS) for center in km.cluster_centers_]
            assert s == result[id]
        print("\tPASS Test 3.2: Correct values for result.")
    except :
        print("\tFAIL Test 3.2: Incorrect values for result.")



def verifyFitTransformResults(useBKM, dataset, results, km):
    fit_data = dataset["fit"]

    # Condition 1: Each result is the correct size
    try :
        assert len(results) == len(fit_data)
        print("\tPASS Test 6.1: Correct dimensions for result.")
    except :
        print("\tFAIL Test 6.1: Incorrect dimensions for result.")


    # Condition 2: Each sub result is the distance from a data point to a center
    result = [([round(j, DECIMALS) for j in i]) for i in results]
    try :
        for id in range(len(fit_data)):
            s = [round((math.sqrt(sum([(fit_data[id][n] - center[n])**2 for n in range(km.n_features_in_)]))), DECIMALS) for center in km.cluster_centers_]
            assert s == result[id]
        print("\tPASS Test 6.2: Correct values for result.")
    except :
        print("\tFAIL Test 6.2: Incorrect values for result.")

    # Condition 3: fit_transform() == fit().transform()
    KM2 = BisectingKMeans if useBKM else KMeans
    expected = KM2(n_clusters=km.n_clusters).fit(fit_data).transform(fit_data)
    try :
        assert [(set([round(j, DECIMALS) for j in i])) for i in results] == [(set([round(j, DECIMALS) for j in i])) for i in expected]
        print("\tPASS Test 6.3: fit_transform() and fit().transform() are equivalent.")
    except :
        print("\tFAIL Test 6.3: fit_transform() differs from fit().transform().")


def verifyScoreResults(dataset, results, DECIMALS = 5):
    try :
        assert round(dataset["expected"]["score"], DECIMALS) == round(results, DECIMALS)
        print("\tPASS Test 4.1: Correct score.")
    except :
        print("\tFAIL Test 4.1: Incorrect score.")


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
