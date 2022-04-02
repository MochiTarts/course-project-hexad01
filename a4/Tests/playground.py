from sklearn.cluster import KMeans
import numpy as np
SEP = "=" * 100
data = {
    "Dataset 1" : {
        "K" : 2, "n" : 1,
        "fit" : [[0.2],[0.8]],
        "pre" : [[0.3], [0.9]],
        "expected" : {
            "labels": [1, 0],
            "centers": [[0.8], [0.2]],
            "predict": [1, 0]
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
        
        "expected" : {
            "labels": [0, 0, 1, 1],
            "centers": [[0.785, 0.54 ],
                        [0.27,  0.415]],
            "predict": [1, 0, 0]
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
        
        "expected" : {
            "labels": [5, 0, 3, 4, 6, 1, 2],
            "centers": [[0.74, 0.49],
                        [0.25, 0.38],
                        [0.63, 0.82],
                        [0.77, 0.58],
                        [0.34, 0.4 ],
                        [0.8,  0.5 ],
                        [0.2,  0.43]],
            "predict": [1, 0, 2]
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
            "predict": None
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
            print("fitting...")
            km.fit(X)
            print("\nLabels:\n", km.labels_)
            print("\nCluster Centers:\n", km.cluster_centers_)
            print(verifyFitResults(dataset, km))

        if predict:
            print("predicting...")
            predictOut = km.predict(dataset["pre"])
            print("\nPredict:\n", predictOut)
            print(verifyPredictResults(dataset, predictOut))

        if transform:
            print("transforming...")
        

def verifyFitResults(dataset, km):
    expected = dataset["expected"]
    expLabels, expCenters = expected["labels"], expected["centers"]
    # Each center must exist in both the output matrix and expected set
    for center in list(km.cluster_centers_):
        if list(center) not in expCenters:
            return False
    # Output should contain the correct number of labels and contain same elements
    return len(km.labels_) == len(expLabels) and set(km.labels_) == set(expLabels)


def verifyPredictResults(dataset, results):
    expResults = dataset["expected"]["predict"]
    return set(expResults) == set(results) and len(expResults) == len(results)

   
if __name__ == '__main__':
    testAll(fit=True, predict=True)



# arr2 = [[0.34, 0.43], [0.94, 0.34]]

# X1 = np.array(arr2)
# print(km.transform(X1))