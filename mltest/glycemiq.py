import numpy as np
from mltest import test_data

# feature engineering
test_data['diff5'] = np.log(test_data['bg']) - np.log(test_data['bg-5'])
test_data['diff10'] = np.log(test_data['bg-5']) - np.log(test_data['bg-10'])
test_data['diff15'] = (np.log(test_data['bg-10']) - np.log(test_data['bg-15'])) / 2
test_data['diff20'] = (np.log(test_data['bg-15']) - np.log(test_data['bg-20'])) / 4
test_data['diff25'] = (np.log(test_data['bg-20']) - np.log(test_data['bg-25'])) / 8
test_data['diff30'] = (np.log(test_data['bg-25']) - np.log(test_data['bg-30'])) / 16
test_data['difflabel'] = np.log(test_data['label']) - np.log(test_data['bg'])
cols = ['label', 'bg', 'difflabel', 'diff5', 'diff10', 'diff15', 'diff20', 'diff25',
        'diff30', 'glycemicindex', 'calories', 'carbs', 'fiber', 'sugar',
        'basal_insulin', 'bolus_insulin']

x_full = test_data[cols]
ix = x_full.isnull().any(axis=1)
x_full = x_full.loc[~ix, :]

# LSTM Params
feature_count = len(cols) - 1
label_count = 1


def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))


