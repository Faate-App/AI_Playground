import numpy as np


def sae(model_func, exact_y_arr):
    sum_of_errors = 0
    for y in exact_y_arr:
        sum_of_errors += np.abs(y - model_func(y))

    return sum_of_errors


def mae(model_func, exact_y_arr):
    return sae(model_func, exact_y_arr) / len(exact_y_arr)


def sse(model_func, exact_y_arr):
    sum_of_errors = 0
    for y in exact_y_arr:
        difference = (y - model_func(y))
        sum_of_errors += difference * difference

    return sum_of_errors


def mse(model_func, exact_y_arr):
    return sse(model_func, exact_y_arr) / len(exact_y_arr)


def sse_tot(model_func, exact_y_arr):
    average = np.average(exact_y_arr)
    sum_of_errors = 0
    for y in exact_y_arr:
        difference = (average - model_func(y))
        sum_of_errors += difference * difference

    return sum_of_errors


def r_squarred(model_func, exact_y_arr):
    return 1 - sse(model_func, exact_y_arr) / sse_tot(model_func, exact_y_arr)
