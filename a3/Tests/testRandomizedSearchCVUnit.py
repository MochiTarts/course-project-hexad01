from functools import reduce
import itertools
import numpy as np
import os
import scipy
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

SEP = '.' * 150
LINE = '=' * 150
prompt = "Does the scikit-learn source code have the bugfix applied? [Y/N] "


def testRandomizedSearchCV(case, desc, params, expt, iters, repetition, bugFixed):
    '''
    [Issue #18057]
    RandomizedSearchCV is not sampling uniformly from param_distributions
    when passed a list of multiple dicts.
    Source: https://github.com/scikit-learn/scikit-learn/issues/18057
    '''
    print(LINE)
    print(case, ":", desc)
    print(LINE)

    X, y = load_iris(return_X_y=True)
    rf = RandomForestClassifier()

    print("Input:")

    for i in range(len(params)):
        print("\t dict" + str(i + 1) + ": " + str(params[i]))

    if bugFixed:
        rand = RandomizedSearchCV(
            rf, params, cv=5, scoring='accuracy',
                n_iter=iters, random_state=1, without_replacement = not repetition)
    else:
        rand = RandomizedSearchCV(
            rf, params, cv=5, scoring='accuracy',
                n_iter=iters, random_state=1)

    rand.fit(X, y)

    print(SEP)
    print("Output:")

    # Gets all the unique dictionary keys
    param_keys = [x for x in list(
        rand.cv_results_.keys()) if x.startswith("param_")]

    # Build results as tuples
    results = list(zip(*[list((
        j if j != np.ma.core.MaskedConstant else None)
            for j in rand.cv_results_[
                param_keys[i]]) for i in range(len(param_keys))]))

    print("\t" + str(tuple(param_keys)) + ": \n \t" + str(results))
    print(SEP)
    print("Test Results:")

    chosen = [0] * len(params)
    masterlist = rand.cv_results_['param_n_estimators']
    for i in masterlist:
        for j in range(len(params)):
            if i in params[j]['n_estimators']:
                chosen[j] += 1
                break

    # Test 1: Make sure selection has uniform distribution
    print("     Test 1: ")
    print("             dict#  times choosen  actual %  expected %")
    print("             ------------------------------------------")
    for i in range(len(chosen)):
        p = "{:>7}  {:>11}  {:>9}".format(str(chosen[i]),  str(round((
            chosen[i] / len(results)) * 100, 2)), str(
                round(float (expt[i]), 2)))
        print("             dict" + str(i + 1) + " | " + p)

    # Test 2: Make sure contains a valid amount of combos
    combos = []
    for k in range(len(params)):
        combos = combos + list(itertools.product(*(
        params[k][i] for i in params[k].keys() if type(params[k][i]) == list)))

    combinations = len(combos)

    try :
        assert (repetition and len(results) == iters) or (
            not repetition and len(results) == min(combinations, iters))
        print("PASS Test 2: The amount of resulting combinations is valid.")
    except :
        print("FAIL Test 2: The result should have a size of, " + str(
            iters if repetition else min(
                combinations, iters)) + ", not " + str(len(results)))

    # Test 3: Make sure each tuple belongs to a dictionary
    mismatch = []

    for i in range(len(results)):
        found = False
        for j in range(len(combos)):
            if set(combos[j]).issubset(results[i]):
                found = True
                break
        if not found:
            mismatch.append(results[i])

    try :
        assert len(mismatch) == 0
        print("PASS Test 3: The resulting combinations are valid.")
    except :
        print("FAIL Test 3: The following are invalid combinations, " + str(
            mismatch))

    # Test 4: If repetition is off, make sure each tuple is unique
    if not repetition:
        try :
            for result in results:
                assert results.count(result) == 1
            print("PASS Test 4: No repeating results found")
        except :
            print("FAIL Test 4: Repeating result found.")

    print(LINE, '\n' * 2)


def testAll(cases):
    os.system('cls' if os.name == 'nt' else 'clear')

    selection = input(prompt)
    while not selection in 'yYnN':
        selection = input(prompt)
    bugFixed = selection.upper() == "Y"

    for case in cases.keys():
        testCase = cases[case]
        desc = testCase['desc']
        iters = testCase['args'][0]
        repetition = testCase['args'][1]
        params = [testCase['args'][i] for i in range(2, len(testCase['args']))]
        expt = testCase['expt']
        testRandomizedSearchCV(case, desc, params, expt, iters, repetition, bugFixed)


testCases = {
    'Test Set 1: Repetition off and Equal Distribution, 1 Param Dictionaries' : {
        'desc': 'repetition = False; iterations = 10',
        'args': [10, False,
                 {'n_estimators': [10, 20],
                    'min_samples_leaf': [1, 2, 3, 4, 5]}],
        'expt': [100]
        },
    'Test Set 2: Repetition off and Equal Distribution, 2 Param Dictionaries' : {
        'desc': 'repetition = False; iterations = 16',
        'args': [16, False,
                 {'n_estimators': [10, 20, 30], 'min_samples_leaf': [5, 6],
                    'bootstrap': [True, False]},
                 {'n_estimators': [40, 50], 'min_samples_leaf': [1, 2, 3, 4]}],
        'expt': [50, 50]
        },
    'Test Set 3: Repetition off and Equal Distribution, 3 Param Dictionaries' : {
        'desc': 'repetition = False; iterations = 50',
        'args': [50, False,
                 {'n_estimators': [10, 20, 30],
                    'min_samples_leaf': [1, 2, 3, 4, 5]},
                 {'n_estimators': [40, 50, 60],
                    'min_samples_leaf': [4, 5, 6, 7, 8, 9, 10]},
                 {'n_estimators': [50, 60, 70, 80, 90, 100, 110],
                    'bootstrap': [False, True]}],
        'expt': [33.33, 33.33, 33.33]
        },
    'Test Set 4: Repetition off and Unequal Distribution, 2 Param Dictionaries' : {
        'desc': 'repetition = False; iterations = 7',
        'args': [7, False,
                 {'n_estimators': [10, 20], 'min_samples_leaf': [1, 2]},
                 {'n_estimators': [30], 'min_samples_leaf': [4, 5, 6]}],
        'expt': [57.14, 42.86]
        },
    'Test Set 5: Repetition on and Unequal Distribution, 4 Param Dictionaries' : {
        'desc': 'repetition = False; iterations = 14',
        'args': [14, False,
                 {'n_estimators': [10, 20, 30], 'min_samples_leaf': [1, 2]},
                 {'n_estimators': [40], 'min_samples_leaf': [4, 5, 6]},
                 {'n_estimators': [50, 60], 'min_samples_leaf': [1, 2]},
                 {'n_estimators': [70]}],
        'expt': [42.86, 21.43, 28.57, 7.14]
        },
    'Test Set 6: Repetition on and Unequal Distribution, 3 Param Dictionaries' : {
        'desc': 'repetition = False; iterations = 20',
        'args': [25, False,
                 {'n_estimators': [10, 20, 30, 40],
                    'min_samples_leaf': [1, 2, 3]},
                 {'n_estimators': [50], 'min_samples_leaf': [4, 5, 6, 7, 8, 9]},
                 {'n_estimators': [50, 60, 70, 80, 90, 100, 110],
                    'min_samples_leaf': [1, 2], 'bootstrap': [False, True]}],
        'expt': [38, 24.0, 38]
        },
    'Test Set 7: Repetition on, 1 Param Dictionaries' : {
        'desc': 'repetition = True; iterations = 50',
        'args': [50, True,
                 {'n_estimators': [10, 20], 'min_samples_leaf': [1, 2, 3]}],
        'expt': [100]
        },
    'Test Set 8: Repetition on, 2 Param Dictionaries' : {
        'desc': 'repetition = True; iterations = 50',
        'args': [50, True,
                 {'n_estimators': [10, 20, 30, 40],
                    'min_samples_leaf': [1, 2, 3]},
                 {'n_estimators': [50, 60, 70, 80, 90, 100, 110],
                    'bootstrap': [False, True]}],
        'expt': [50, 50]
        },
    'Test Set 9: Repetition on, 4 Param Dictionaries' : {
        'desc': 'repetition = True; iterations = 50;',
        'args': [50, True,
                 {'n_estimators': [130]},
                 {'n_estimators': [10, 20, 30, 40],
                    'min_samples_leaf': [1, 2, 3, 4]},
                 {'n_estimators': [80, 90, 100, 110, 120],
                    'bootstrap': [False, True]},
                 {'n_estimators': [50, 60, 70, 80],
                    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8],
                    'max_features': ['auto', None],
                    'bootstrap': [True, False]}],
        'expt': [25, 25, 25, 25]
        },
    'Test Set 10: Repetition with non-list Parameters' : {
        'desc': 'repetition = True; iterations = 50;',
        'args': [50, True,
                 {'n_estimators': [130]},
                 {'n_estimators': [10, 20, 30, 40],
                    'min_samples_leaf': scipy.stats.uniform(
                        loc=0.25, scale=0.2)},
                 {'n_estimators': [80, 90, 100, 110, 120],
                    'bootstrap': [False, True]},
                 {'n_estimators': [50, 60, 70, 80],
                    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8],
                    'max_features': ['auto', None],
                    'bootstrap': [True, False]}],
        'expt': [25, 25, 25, 25]
        }
}

testAll(testCases)
