import numpy as np
from scipy.optimize import curve_fit



# logaritmic regression fit function
def logarithmic_regression_fit(x, y):
    """
    Fit a logarithmic regression function to the data
    :param x: x data
    :param y: y data
    :return: a, b
    """
    def log_func(x, a, b):
        return a + b * np.log(x)

    # Finding the optimal parameters :
    popt, pcov = curve_fit(log_func, x, y)
    print("a  = ", popt[0])
    print("b  = ", popt[1])

    # Predicting values:
    y_pred = log_func(x, popt[0], popt[1])

    # Check the accuracy :
    from sklearn.metrics import r2_score
    Accuracy = r2_score(y, y_pred)
    print(f'R**2: {Accuracy}')

    return popt[0], popt[1], y_pred



