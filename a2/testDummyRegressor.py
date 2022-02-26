import numpy as np
from sklearn.dummy import DummyRegressor
import sys
import os


SEP = '.' * 100
LINE = '=' * 100


def testDummyRegressor(case, desc, const, X, y):
    '''
    [Issue #22478]
    Some parameters passed into DummyRegressor are erroneously converted
    into Numpy formart after the fit() method is invoked.
    Source: https://github.com/scikit-learn/scikit-learn/issues/22478
    '''
    print(LINE)
    print(case, ":", desc)
    print(LINE)

    print("Creating a DummyRegressor...")
    regressor = DummyRegressor(constant=const, strategy="constant")
    params = regressor.get_params()
    print("Before fit() Method Invocation:")
    print(" " * 4, "params =", params)
    print(" " * 4, "X =", X)
    print(" " * 4, "y =", y)
    print(SEP)
    
    regressor.fit(X=X, y=y)
    result = regressor.get_params()
    print("After fit() Method Invocation:")

    print("    Expected:", params)
    print("    Actual:  ", result)
    print(SEP)

    print("Test Results:")
    
    try:
        assert type(result['constant']) == type(const)
        assert type(result['constant']) != np.ndarray
        print("    [PASS] Test 1: constant is of the expected type.")
    except:
        print("    [FAIL] Test 1: constant should be", type(const), "not", type(result['constant']))
    
    try:
        assert result == params
        assert result['constant'] == const
        print("    [PASS] Test 2: constant holds the expected value.")
    except:
        print("    [FAIL] Test 2: incorrect value of constant received.")

    print(LINE, '\n' * 2)


def testAll(cases):
    os.system('cls' if os.name == 'nt' else 'clear')
    for case in cases.keys():
        testCase = cases[case]
        desc = testCase['desc']
        const = testCase['args'][0]
        X = testCase['args'][1]
        y = testCase['args'][2]
        testDummyRegressor(case, desc, const, X, y)


testCases = {
    'Test Set 1: 0 Float Constant' : {
        'desc': 'strategy = constant; constant = 0.0', 
        'args': [0.0, [[1,1], [2,2], [3,3]], [[0], [0], [0]]]
        },
    'Test Set 2: 0 Integer Constant' : {
        'desc': 'strategy = constant; constant = 0', 
        'args': [0, [[1,1], [2,2], [3,3]], [[0], [0], [0]]]
        },
    'Test Set 3: Postive Integer Constant' : {
        'desc': 'strategy = constant; constant = 2', 
        'args': [2, [[1,1], [2,2], [3,3]], [[0], [0], [0]]]
        },
    'Test Set 4: Negative Integer Constant' : {
        'desc': 'strategy = constant; constant = -2', 
        'args': [-2, [[1,1], [2,2], [3,3]], [[0], [0], [0]]]
        },
    'Test Set 5: Negative Float Constant Part 1' : {
        'desc': 'strategy = constant; constant = -3.14', 
        'args': [-3.14, [[1,1], [2,2], [3,3]], [[0], [0], [0]]]
        },
    'Test Set 6: Negative Float Constant Part 2' : {
        'desc': 'strategy = constant; constant = -3.14', 
        'args': [-3.14, [[1,2], [2,1], [3,1]], [[3], [1], [4]]]
        },
    'Test Set 7: Large Positive Float Constant' : {
        'desc': 'strategy = constant; constant = sys.float_info.max', 
        'args': [sys.float_info.max, [[1,2], [2,1], [3,1]], [[3], [1], [4]]]
        },
    'Test Set 8: Large Negative Float Constant' : {
        'desc': 'strategy = constant; constant = -sys.float_info.max', 
        'args': [-sys.float_info.max, [[1,2], [2,1], [3,1]], [[3], [1], [4]]]
        },
    'Test Set 9: Small Positive Float Constant' : {
        'desc': 'strategy = constant; constant = sys.float_info.min', 
        'args': [sys.float_info.min, [[1,2], [2,1], [3,1]], [[3], [1], [4]]]
        },
    'Test Set 10: Small Negative Float Constant' : {
        'desc': 'strategy = constant; constant = -sys.float_info.min', 
        'args': [-sys.float_info.min, [[1,2], [2,1], [3,1]], [[3], [1], [4]]]
        } 
}


testAll(testCases)
