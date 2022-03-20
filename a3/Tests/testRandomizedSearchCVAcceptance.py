import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import sys
import os


SEP = '.' * 150
LINE = '=' * 150


def testRandomizedSearchCV(case, desc, params1ns, params2ns, iters, no_replacement):
    '''
    [Issue #18057]
    RandomizedSearchCV is not sampling uniformly from param_distributions
    when passed a list of multiple dicts.
    Source: https://github.com/scikit-learn/scikit-learn/issues/18057
    '''
    print(LINE)
    print(case, ":", desc)
    print(LINE)

    print("Loading iris dataset...")

    X, y = load_iris(return_X_y=True)
    print("Creating randomForestClassifier...")

    rf = RandomForestClassifier()


    print("Setting Params...")
    print("params1 n_estimators:" + str(params1ns))
    print("params1 min_samples_leaf:" + str([1, 2, 3, 4, 5, 6]))


    params1 = {}
    params1['n_estimators'] = params1ns
    params1['min_samples_leaf'] = [1, 2, 3, 4, 5, 6]

    print("params2 n_estimators:" + str(params2ns))
    print("params2 min_samples_leaf:" + str([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

    params2 = {}
    params2['n_estimators'] = params2ns
    params2['min_samples_leaf'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    params2['max_features'] = ['auto', None]
    params2['bootstrap'] = [True, False]

    params_both = [params1, params2]

    print("Creating RandomizedSearchCV...")
    rand = RandomizedSearchCV(rf, params_both, cv=5, scoring='accuracy', n_iter=iters, random_state=2, without_replacement= no_replacement)
    print("Fitting...")
    rand.fit(X, y)

    dict1Chosen = len(list(filter(lambda x: (x < 45), rand.cv_results_['param_n_estimators'])))
    dict2Chosen = len(list(filter(lambda x: (x > 45), rand.cv_results_['param_n_estimators'])))

    print(SEP)

    print("Test Results:")
    print("Selected n_estimators:")
    print(rand.cv_results_['param_n_estimators'])
    print("Params from dict 1 chosen: " + str(dict1Chosen) + " times")
    print("Params from dict 2 chosen: " + str(dict2Chosen) + " times")
    print('\n')
    print("Final percent of choices from dict 1: " + str(((dict1Chosen / len(rand.cv_results_['param_n_estimators']) ) * 100)))
    print("Final percent of choices from dict 2: " + str(((dict2Chosen / len(rand.cv_results_['param_n_estimators']) ) * 100)))
    print("Expected percentages: near 50%")

    print(LINE, '\n' * 2)


def testAll(cases):
    os.system('cls' if os.name == 'nt' else 'clear')
    for case in cases.keys():
        testCase = cases[case]
        desc = testCase['desc']
        iters = testCase['args'][0]
        params1 = testCase['args'][1]
        params2 = testCase['args'][2]
        without_replacement = testCase['args'][3]


        testRandomizedSearchCV(case, desc, params1, params2, iters, without_replacement)


testCases = {
    'Test 1: few iterations, without replacement' : {
        'desc': 'iterations = 50; params1 n_estimators: [10, 20, 30, 40], params2 n_estimators [50, 60, 70, 80], without_replacement=true',
        'args': [50, [10, 20, 30, 40], [50, 60, 70, 80], True]
        },
    'Test 2: many iterations, without replacement' : {
        'desc': 'iterations = 100; params1 n_estimators: [10, 20, 30, 40], params2 n_estimators [50, 60, 70, 80], without_replacement=true',
        'args': [100, [10, 20, 30, 40], [50, 60, 70, 80], True]
        },
    'Test 3: different list values for params2, without replacement': {
        'desc': 'iterations = 50; params1 n_estimators: [10, 20, 30, 40], params2 n_estimators [50, 55, 60, 65, 70, 75, 80, 85], without_replacement=true',
        'args': [50, [10, 20, 30, 40], [50, 55, 60, 65, 70, 75, 80, 85], True]
        },
    'Test 4: different list values for params1, without replacement': {
        'desc': 'iterations = 50; params1 n_estimators: [11, 22, 32, 44], params2 n_estimators [50, 55, 60, 65, 70, 75, 80, 85], without_replacement=true',
        'args': [50, [11, 22, 32, 44], [50, 60, 70, 80], True ]
        },
    'Test 5: few iterations, with replacement' : {
        'desc': 'iterations = 50; params1 n_estimators: [10, 20, 30, 40], params2 n_estimators [50, 60, 70, 80], without_replacement=false',
        'args': [50, [10, 20, 30, 40], [50, 60, 70, 80], False]
        },
    'Test 6: many iterations, with replacement' : {
        'desc': 'iterations = 100; params1 n_estimators: [10, 20, 30, 40], params2 n_estimators [50, 60, 70, 80], without_replacement=false',
        'args': [100, [10, 20, 30, 40], [50, 60, 70, 80], False]
        },
    'Test 7: different list values for params2, with replacement': {
        'desc': 'iterations = 50; params1 n_estimators: [10, 20, 30, 40], params2 n_estimators [50, 55, 60, 65, 70, 75, 80, 85], without_replacement=false',
        'args': [50, [10, 20, 30, 40], [50, 55, 60, 65, 70, 75, 80, 85], False]
        },
    'Test 8: different list values for params1, with replacement': {
        'desc': 'iterations = 50; params1 n_estimators: [11, 22, 32, 44], params2 n_estimators [50, 55, 60, 65, 70, 75, 80, 85], without_replacement=false',
        'args': [50, [11, 22, 32, 44], [50, 60, 70, 80], False ]
        }
}


testAll(testCases)
