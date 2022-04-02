from sklearn.cluster import KMeans
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
            "centers" : [[0.8], [0.2]],
            "transform" : [[0.14, 0.46], [0.74, 0.14]]
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
                        [0.27,  0.415]],
            "transform" : [[0.75407228, 0.3826552 ],
                            [0.235,      0.62835102],
                            [0.44342418, 0.58917315]]
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
                        [0.2,  0.43]],
            "transform" : [[0.4554119, 0.14, 0.48600412, 0.03,      
                            0.4652956, 0.1029563, 0.40447497],
                        [0.29410882, 0.74545288, 0.57140179, 0.60299254,  
                            0.21260292, 0.69115845, 0.25]]
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
            "centers": None,
            "transform": None
        }
    }
}


def testAll(fit=True, predict=False, transform=False):
    for set in data.keys():

        if set == "Dataset 4": # Take this out when DS4 is done
            break

        print (SEP + "\n" + set + '\n' + SEP)
        dataset = data[set]
        km = KMeans(n_clusters=dataset["K"])
        X = np.array(dataset["fit"])

        if fit:
            print("Testing fit()")
            km.fit(X)
            print("\nLabels:\n", km.labels_)
            print("\nCluster Centers:\n", km.cluster_centers_)
            print(verifyFitResults(dataset, km))

            if predict:
                print("Testing predict()")
                predictOut = km.predict(dataset["pre"])
                print("\nPredict:\n", predictOut)
                print(verifyPredictResults(dataset, predictOut, km))

            if transform:
                print("Testing transform()")

        
def verifyFitResults(dataset, km):
    expected = dataset["expected"]
    expLabels = expected["labels"] 
    expCenters = np.array(expected["centers"])
    received = [float('%.3f' % elem) for elem in list(np.sort(km.cluster_centers_.flat))]
    expected = [float('%.3f' % elem) for elem in list(np.sort(expCenters.flat))]
    return received == expected and set(expLabels) == set(km.labels_)


def verifyPredictResults(dataset, results, km):
    data = dataset["pre"]
    # Condition 1: Each data point belongs to their correct cluster
    for id in range(len(data)):
        expected_min = sum([(data[id][n] - km.cluster_centers_[results[id]][n])**2 for n in range(km.n_features_in_)])

        for point in km.cluster_centers_:
            local_min = sum([(data[id][n] - point[n])**2 for n in range(km.n_features_in_)])
            if (local_min < expected_min):
                return False
    # Condition 2: Checks that result is the correct size
    return len(results) == len(data)

   
if __name__ == '__main__':
    testAll(fit=True, predict=True)
