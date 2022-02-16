import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge, Lasso
import os

SEP = '.' * 80


def demoDummyRegressor():
    '''
    [Issue #22478]
    Some parameters passed into DummyRegressor are erroneously converted
    into Numpy formart after the fit() method is invoked.
    Source: https://github.com/scikit-learn/scikit-learn/issues/22478
    '''
    print("Creating a DummyRegressor...")
    regressor = DummyRegressor(constant=0.0, strategy="constant")
    params = regressor.get_params()
    print("Before fit() Method Invocation:")
    print(" " * 4, params)
    print(SEP)
    
    regressor.fit(X=[[1,1], [2,2], [3,3]], y=[[0], [0], [0]])
    result = regressor.get_params()
    print("After fit() Method Invocation:")
    print("    Expected:", params)
    print("    Actual:  ", result)
    print(SEP, '\n')


def demoIterativeImputer():
    '''
    [Issue #19352]
    When setting the estimator as PLSRegression(), 
    a ValueError is triggered by module '_iteractive.py' in line 348, 
    caused by "shape mismatch"
    Source: https://github.com/scikit-learn/scikit-learn/issues/19352
    '''
    X_california, y_california = fetch_california_housing(return_X_y=True)
    X_california = X_california[:400]
    y_california = y_california[:400]

    X_miss_california, y_miss_california = add_missing_values(
        X_california, y_california)
    print("Setting an estimator as PLSRegression()...")
    imputer = IterativeImputer(estimator=PLSRegression(n_components=2))

    try:
        imputer.fit_transform(X_miss_california)
    except:
        print("Expected:")
        print("[[   8.3252       41.            6.98412698 ...    2.55555556\
                    37.88       -122.25930206]\
                [   8.3014       21.            6.23813708 ...    2.10984183\
                    37.86       -122.22      ]\
                [   7.2574       52.            8.28813559 ...    2.80225989\
                    37.85       -122.24      ]\
                ...\
                [   3.60438721   50.            5.33480176 ...    2.30396476\
                    37.88       -122.29      ]\
                [   5.1675       52.            6.39869281 ...    2.44444444\
                    37.89       -122.29      ]\
                [   5.1696       52.            6.11590296 ...    2.70619946\
                    37.8709526  -122.29      ]]")
        print(SEP)
        print("Actual:")
        print("ValueError: shape mismatch: value array of shape (27,1) ")
        print("could not be broadcast to indexing result of shape (27,) ")
        print(SEP, '\n')


def add_missing_values(X_full, y_full):
    '''
    [Issue #19352]
    Helper function for demoIterativeImputer
    Source: https://github.com/scikit-learn/scikit-learn/issues/19352
    '''
    rng = np.random.RandomState(42)
    n_samples, n_features = X_full.shape

    # Add missing values in 75% of the lines
    missing_rate = 0.75
    n_missing_samples = int(n_samples * missing_rate)

    missing_samples = np.zeros(n_samples, dtype=bool)
    missing_samples[: n_missing_samples] = True

    rng.shuffle(missing_samples)
    missing_features = rng.randint(0, n_features, n_missing_samples)
    X_missing = X_full.copy()
    X_missing[missing_samples, missing_features] = np.nan
    y_missing = y_full.copy()

    return X_missing, y_missing


def demoCountVectorizer():
    '''
    [Issue #21207]
    Some characters are transformed to uppercase chars (eg ™ -> TM), despite lowercase=True.
    This then gives warning messages "UserWarning: 
    Upper case characters found in vocabulary while 'lowercase' is True"
    Source: https://github.com/scikit-learn/scikit-learn/issues/21207
    '''
    print("Setting CountVectorizer with lowercase set to True...")
    x = ['This is Problematic™.','THIS IS NOT']
    print("Input:  ", x)
    cv = CountVectorizer(
        lowercase=True,
        strip_accents='unicode',
        ngram_range=(1,1))
    x_v = cv.fit_transform(x)
    y = [1,0]
    xtest = ['This is not']
    ytest = [0]

    print(SEP)
    print("Output:  ", x) 
    expected = ['is', 'not', 'problematictm', 'this']
    result = cv.get_feature_names_out()

    print("    Expected:", expected)
    print("    Actual:  ", result)
    print(SEP, '\n')


def demoRidgeLasso():
    '''
    [Issue #19693]
    Ridge.coef_ returns an array with shape (1, n_features), 
    while Lasso.coef_ returns an array with shape (n_features,). 
    LinearRegression.coef_ resembles Ridge.coef_
    Source: https://github.com/scikit-learn/scikit-learn/issues/19693
    '''
    print("Instantiating Ridge and Lasso")
    x = np.array([1,2,3,4,5]).reshape(-1,1)
    y = np.array([2,4,6,8,10]).reshape(-1,1)
    print("Input:")
    print("  x:", [1,2,3,4,5])
    print("  y:", [2,4,6,8,10])
    ridge_reg = Ridge()
    lasso_reg = Lasso()
    _ = ridge_reg.fit(x,y)
    _ = lasso_reg.fit(x,y)
    print(SEP)
    print("Output:")
    print("  Expected:") 
    print("  ", (1, 1))
    print("  ", (1, 1))
    print("  Actual:") 
    print("  ", ridge_reg.coef_.shape)
    print("  ", lasso_reg.coef_.shape)
    print(SEP, '\n')


def demoFitTransform():
    '''
    [Issue #18941]
    PCA's fit_transform returns different results than the application of 
    fit and transform methods individually
    Source: https://github.com/scikit-learn/scikit-learn/issues/18941
    '''
    print("Calling fit transform [First time]...")
    print("Received:")
    nn = np.array([[0,1,2],[3,4,5],[6,7,8]])
    pca = PCA(n_components=2, random_state=42)
    out1 = pca.fit_transform(nn)
    print(out1)
    print(SEP)
    
    print("Calling fit transform [Second time]...")
    print("Received:")
    nn = np.array([[0,1,2],[3,4,5],[6,7,8]])
    pca = PCA(n_components=2, random_state=42)
    pca.fit(nn)
    out2 = pca.transform(nn)
    print(out2)
    print(SEP)
    
    if out1.all() == out2.all():
        print("Outputs (only roughly) matched but should be identical.")
        print(SEP, '\n')
    else:
        print("Outputs differed significantly but should be identical.")
        print(SEP, '\n')

def getSelection(bugs):
    menu = "Please Select an Issue:\n" + '\n'.join(['(' + option + ') ' + bugs[option]['desc'] for option in bugs.keys()]) + '\n'
    bugIndex = input(menu)
    while bugIndex.upper() not in bugs.keys():
        bugIndex = input(bugIndex + " is not a valid selection. Please try again: ")
    return bugIndex.upper()

def main(bugs):
    selection = getSelection(bugs)
    while selection != 'Q':
        os.system('cls' if os.name == 'nt' else 'clear')
        print(bugs[selection]["desc"], "\n")
        bugs[selection]["func"]()
        selection = getSelection(bugs)

if __name__ == '__main__':
    bugs = {
        "1": {"desc": "[Issue #22478] Dummy Regressor", "func": demoDummyRegressor},
        "2": {"desc": "[Issue #19352] Iterative Imputer", "func": demoIterativeImputer},
        "3": {"desc": "[Issue #21207] Count Vectorizer", "func": demoCountVectorizer},
        "4": {"desc": "[Issue #19693] Ridge & Lasso", "func": demoRidgeLasso},
        "5": {"desc": "[Issue #18941] FitTransform", "func": demoFitTransform},
        "Q": {"desc": "Quit", "func": None}
    }
    main(bugs)
    
    

    