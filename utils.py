import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# This file contains functions that are used in the notebooks

# create a function to create feature tensors from the data

def create_dataset(dataset, look_back=100, predict_ahead=30, target_col_index=-1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - predict_ahead):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back + predict_ahead, target_col_index])
    return np.array(X), np.array(Y)


from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))


# create a function that takes in X, Y and train, val and test size and returns the train, val and test sets

def train_val_test_split_old(X, Y, train_size, val_size, test_size):

    X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
    Y_train, Y_val, Y_test = Y[:train_size], Y[train_size:train_size + val_size], Y[train_size + val_size:]

    # reshape X_train, X_val and X_test to be 2D arrays
    X_train_2D = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
    X_val_2D = X_val.reshape((X_val.shape[0], X_val.shape[1] * X_val.shape[2]))
    X_test_2D = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))
    
    # scale X data based on X_train to avoid look-ahead bias

    scaler = StandardScaler()
    scaler.fit(X_train_2D)
    X_train_2D_scaled = scaler.transform(X_train_2D)
    X_val_2D_scaled = scaler.transform(X_val_2D)
    X_test_2D_scaled = scaler.transform(X_test_2D)

    # reshape X_train_2D_scaled, X_val_2D_scaled and X_test_2D_scaled to be 3D arrays again
    X_train_scaled = X_train_2D_scaled.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_val_scaled = X_val_2D_scaled.reshape((X_val.shape[0], X_val.shape[1], X_val.shape[2]))
    X_test_scaled = X_test_2D_scaled.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    return X_train_scaled, Y_train, X_val_scaled, Y_val, X_test_scaled, Y_test


def train_val_test_split(X_arr, Y_arr, val_per, test_perc, n_samples):

    val_size = int(n_samples * val_per)
    test_size = int(n_samples * test_perc)
    train_size = len(X_arr) - val_size - test_size


    X_train, X_val, X_test = X_arr[:train_size], X_arr[train_size:train_size + val_size], X_arr[train_size + val_size:]
    Y_train, Y_val, Y_test = Y_arr[:train_size], Y_arr[train_size:train_size + val_size], Y_arr[train_size + val_size:]

    # reshape X_train, X_val and X_test to be 2D arrays
    X_train_2D = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
    X_val_2D = X_val.reshape((X_val.shape[0], X_val.shape[1] * X_val.shape[2]))
    X_test_2D = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))
    
    # scale X data based on X_train to avoid look-ahead bias

    scaler = StandardScaler()
    scaler.fit(X_train_2D)
    X_train_2D_scaled = scaler.transform(X_train_2D)
    X_val_2D_scaled = scaler.transform(X_val_2D)
    X_test_2D_scaled = scaler.transform(X_test_2D)

    # reshape X_train_2D_scaled, X_val_2D_scaled and X_test_2D_scaled to be 3D arrays again
    X_train_scaled = X_train_2D_scaled.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_val_scaled = X_val_2D_scaled.reshape((X_val.shape[0], X_val.shape[1], X_val.shape[2]))
    X_test_scaled = X_test_2D_scaled.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    return X_train_scaled, Y_train, X_val_scaled, Y_val, X_test_scaled, Y_test