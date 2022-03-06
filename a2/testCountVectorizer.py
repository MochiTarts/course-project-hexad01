from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os


SEP = '.' * 100
LINE = '=' * 100


def testDummyRegressor(case, desc, x, args, expected):
    '''
    [Issue #21207]
    Some characters (ie. accented characters) are transformed to uppercase
    chars despite setting lowercase=True.
    Source: https://github.com/scikit-learn/scikit-learn/issues/21207
    '''
    print(LINE)
    print(case, ":", desc)
    print(LINE)

    lowercase = args['lowercase'] if args['lowercase'] != '' else True
    strip_accents = args['strip_accents'] if args['strip_accents'] != '' else 'unicode'
    lower_first = args['lower_first'] if args['lower_first'] != '' else False

    print("Creating a CountVectorizer...")
    cv = CountVectorizer(lowercase=lowercase,
        strip_accents=strip_accents,
        lower_first=lower_first)
    print("Before running analyzer:")
    print(" " * 4, "x =", x)
    print(" " * 4, "lowercase =", lowercase)
    print(" " * 4, "strip_accents =", strip_accents)
    print(" " * 4, "lower_first =", lower_first)
    print(SEP)

    analyze = cv.build_analyzer()
    result = []
    for doc in x:
        result = analyze(doc)

    print("After running analyzer:")

    print("    Expected:", expected)
    print("    Actual:", result)

    print("Test Results:")

    try:
        assert np.array_equal(result, expected)
        print("    [PASS] Test: actual matches expected")
    except:
        print("    [FAIL] Test: actual does not match expected")

    print(LINE, '\n' * 2)


def testAll(cases):
    os.system('cls' if os.name == 'nt' else 'clear')
    for case in cases.keys():
        testCase = cases[case]
        desc = testCase['desc']
        x = testCase['x']
        args = testCase['args']
        expected = testCase['expected']
        testDummyRegressor(case, desc, x, args, expected)


testCases = {
    'Test Set 1: Lowercase and New Behaviour' : {
        'desc': 'lowercase = True, lower_first = False', 
        'args': {
            'lowercase': True,
            'strip_accents': 'unicode',
            'lower_first': False
        },
        'x': ['This is Problematic™.'],
        'expected': np.array(['this', 'is', 'problematictm'])
    },
    'Test Set 2: Not lowercasing and New Behaviour' : {
        'desc': 'lowercase = False; lower_first = False', 
        'args': {
            'lowercase': False,
            'strip_accents': 'unicode',
            'lower_first': False
        },
        'x': ['This is Problematic™.'],
        'expected': np.array(['This', 'is', 'ProblematicTM'])
    },
    'Test Set 3: Lowercase and Old Behaviour' : {
        'desc': 'lowercase = True; lower_first = True', 
        'args': {
            'lowercase': True,
            'strip_accents': 'unicode',
            'lower_first': True
        },
        'x': ['This is Problematic™.'],
        'expected': np.array(['this', 'is', 'problematicTM'])
    },
    'Test Set 4: Not lowercasing and Old Behaviour' : {
        'desc': 'lowercase = False; lower_first = True', 
        'args': {
            'lowercase': False,
            'strip_accents': 'unicode',
            'lower_first': True
        },
        'x': ['This is Problematic™.'],
        'expected': np.array(['This', 'is', 'ProblematicTM'])
    },
    'Test Set 5: No unicode characters': {
        'desc': 'x = ["Ace"]; lowercase = True; lower_first = False',
        'args' : {
            'lowercase': True,
            'strip_accents': 'unicode',
            'lower_first': False
        },
        'x': ['Ace'],
        'expected': np.array(['ace'])
    },
    'Test Set 6: Empty string': {
        'desc': 'x = [""]; lowercase = True; lower_first = False',
        'args': {
            'lowercase': True,
            'strip_accents': 'unicode',
            'lower_first': False
        },
        'x': [""],
        'expected': np.array([])
    },
    'Test Set 7: 1 non-accented ascii character': {
        'desc': 'x = ["@"]; lowercase = True; lower_first = False',
        'args': {
            'lowercase': True,
            'strip_accents': 'ascii',
            'lower_first': False
        },
        'x': ["@"],
        'expected': np.array([])
    },
    'Test Set 8: Mix non-accented ascii and normal': {
        'desc': 'x = ["This way » this direction"]; lowercase = True; lower_first = False',
        'args': {
            'lowercase': True,
            'strip_accents': 'ascii',
            'lower_first': False
        },
        'x': ["This way » this direction"],
        'expected': np.array(["this", "way", "this", "direction"])
    },
    'Test Set 9: Only ascii characters (accented and non)': {
        'desc': 'x = ["@!Aé"]; lowercase = True; lower_first = False',
        'args': {
            'lowercase': True,
            'strip_accents': 'ascii',
            'lower_first': False
        },
        'x': ["@!Aé"],
        'expected': np.array(["ae"])
    },
    'Test Set 10: Only unicode characters (accented and non)': {
        'desc': 'x = ["çè©¶"]; lowercase = True; lower_first = False',
        'args': {
            'lowercase': True,
            'strip_accents': 'ascii',
            'lower_first': False
        },
        'x': ["çè©¶"],
        'expected': np.array(["ce"])
    },
    'Test Set 11: Mix non-accented ascii and normal (lowercase = False)' :{
        'desc': 'x = ["This way » this direction"]; lowercase = False; lower_first = False',
        'args': {
            'lowercase': False,
            'strip_accents': 'ascii',
            'lower_first': False
        },
        'x': ["This way » this direction"],
        'expected': np.array(["This", "way", "this", "direction"])
    },
    'Test Set 12: Mix accented ascii and normal': {
        'desc': 'x = ["Real Ëstate"]; lowercase = True; lower_first = False',
        'args': {
            'lowercase': True,
            'strip_accents': 'ascii',
            'lower_first': False
        },
        'x': ["Real Ëstate"],
        'expected': np.array(["real", "estate"])
    }
    
}


testAll(testCases)
