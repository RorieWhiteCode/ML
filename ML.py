import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# import graphviz
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error
%matplotlib
inline

data_path = 'cancer-data.csv'  # Path specific, it's linked to my Juypter notebooks directory
ca = pd.read_csv(data_path)
ca = ca.set_axis(
    ['index', 'logCancerVol', 'logCancerWeight', 'age', 'logBenighHP', 'svi', 'logCP', 'gleasonScore', 'gleasonS45',
     'levelCancerAntigen', 'train'], axis=1)

print('number of samples: {}, number of attributes: {}'.format(ca.shape[0], ca.shape[1]))

# Pre-processing: we split our training data with rows that have 'T' and test data with rows that 'F'
# Index is not reflective of any relationship with levelCancerAntigen + row column covers that, so we drop it from our dataframe
train_ca = ca[ca['train'] == 'T'].drop(['index', 'train'], axis=1)
test_ca = ca[ca['train'] == 'F'].drop(['index', 'train'], axis=1)
train_set = train_ca.values  # Convert df into 2d array, each row is data sample, each column to feature
test_set = test_ca.values
features = train_ca.columns[:-1]
print(train_set.shape)
print(train_set)

X_train = train_set[:, :-1]  # Extract all rows with : and :-1 is to exclude the target
y_train = train_set[:, -1]  # Extract only the target
X_test = test_set[:, :-1]
y_test = test_set[:, -1]

norm = np.linalg.norm(X_train, axis=0)
X_train = X_train / norm[np.newaxis, :]
X_test = X_test / norm[np.newaxis, :]
print(X_train)


def linear_regression(X_train, y_train, X_test, y_test, features):
    linear = LinearRegression(fit_intercept=True, copy_X=True)
    linear.fit(X_train, y_train)
    results = {"Training accuracy": linear.score(X_train, y_train),
               "Testing accuracy": linear.score(X_test, y_test),
               "MSE": mean_squared_error(y_test, linear.predict(X_test)),
               "Intercept": linear.intercept_,
               "Coefficient": linear.coef_}
    # for feature, coef in zip(features, linear.coef_):
    # print("Coefficient for", feature, "is: ", coef)
    return results


model_one = linear_regression(X_train, y_train, X_test, y_test, features)
print(model_one)


def ridge_regression(X_train, y_train, X_test, y_test, features, alpha):
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    results = {"Training accuracy": ridge.score(X_train, y_train),
               "Testing accuracy": ridge.score(X_test, y_test),
               "MSE": mean_squared_error(y_test, ridge.predict(X_test)),
               "Intercept": ridge.intercept_,
               "Coefficient": ridge.coef_}
    # y_pred = ridge.predict(X_test)
    # residuals = y_test - y_pred
    # plt.scatter(y_test, y_pred)
    # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal line
    # plt.xlabel('Actual Values')
    # plt.ylabel('Predicted Values')
    # plt.title('Actual vs Predicted Values')
    # plt.show()
    # for feature, coef in zip(features, ridge.coef_):
    # print("Coefficient for", feature, "is:", coef)
    return results


model_two = ridge_regression(X_train, y_train, X_test, y_test, features, alpha=0.01)
print(model_two)


def k_fold(X_train, y_train, k, method, features, alpha=None):
    fold_output = []
    coefficients = []  # Coeffecients of each fold
    indices = list(range(len(y_train)))  # Create all indices from 0 to the size of our y_train (66 values)
    np.random.shuffle(indices)  # Shuffle them to prevent bias
    size = len(y_train) // k  # The size of each fold is calculated: how much data can we evenly distribute among folds
    for i in range(k):
        start_index = (size * i)  # First index with 66 samples is 13*0 = 0
        end_index = size * (i + 1)  # End index with 13*(0+1) = 13 (66/13) = k
        test_index = indices[start_index:end_index]  # Slice list to get data from 0-13
        train_index = np.setdiff1d(indices, test_index)  # Finds indices not in test_index (intersection)
        x_train_fold = X_train[train_index]
        x_test_fold = X_train[test_index]
        y_train_fold = y_train[train_index]
        y_test_fold = y_train[test_index]
        # if alpha is provided we assume Ridge regression otherwise Linear regression
        if alpha:
            fold = method(x_train_fold, y_train_fold, x_test_fold, y_test_fold, features, alpha)
        else:
            fold = method(x_train_fold, y_train_fold, x_test_fold, y_test_fold, features)
        coefficients.append(fold["Coefficient"])
        fold_output.append(fold)
    return fold_output, coefficients


k_values = []  # Store all of our values for each k so we can graphically analyze the results
training_values = []  # For graphics
testing_values = []  # Graphics
mse_values = []  # Graphics
max_k = 13  # Can max at 33 but it's not necessary
for k in range(2, max_k):  # 67 samples => 67 // 2 possible k values
    total_train_acc = 0
    total_test_acc = 0
    total_mse_acc = 0
    iterations = 1000
    avg_coef = np.zeros(X_train.shape[1])
    for i in range(iterations):
        folds, coeffecients = k_fold(X_train, y_train, k, linear_regression,
                                     features)  # Choose: linear_regression or ridge_regression
        train_acc_avg = 0  # Store our total averages over dem folds
        test_acc_avg = 0
        total_mse = 0
        average_mse = 0
        for fold in folds:
            train_acc_avg += fold["Training accuracy"] / len(folds)  # calculating averages for each fold
            test_acc_avg += fold["Testing accuracy"] / len(folds)
            total_mse += fold["MSE"]  # total for each fold
        average_mse += total_mse / len(folds)
        train_acc_avg = (train_acc_avg * 100)  # Get these sweet calculations in the right form
        test_acc_avg = (test_acc_avg * 100)
        total_train_acc += (train_acc_avg / (iterations))
        total_test_acc += (test_acc_avg / (iterations))
        total_mse_acc += average_mse
        avg_coef += np.mean(coeffecients, axis=0)  # Update values

    avg_coef /= iterations
    training = round(total_train_acc, 2)  # Round just at the end, not within calculations
    testing = round(total_test_acc, 2)
    mse = round(total_mse_acc / (iterations), 2)
    k_values.append(k)
    mse_values.append(mse)
    training_values.append(training)
    testing_values.append(testing)
    print("k value:", k)
    print(" Linear Regression:")  # Display information to user
    print("|Average training accuracy|:", training, "%", "|Iterations|:", iterations, "|Folds|", len(folds))
    print("|Average testing accuracy |:", testing, "%", "|Iterations|:", iterations, "|Folds|", len(folds))
    print("|Average MSE value        |:", mse, "  ", "|Iterations|:", iterations, "|Folds|", len(folds))
    print("| Average Coefficients    |:", avg_coef)

plt.figure(figsize=(8, 6))
plt.plot(k_values, training_values, marker='o', label="Training accuracy", color='red')
plt.title("Training relationship with k")
plt.xlabel("k")
plt.ylabel("Training accuracy %")
plt.legend
plt.grid(True)
plt.figure(figsize=(8, 6))
plt.plot(k_values, testing_values, marker='o', label="Testing accuracy", color='orange')
plt.title("Testing relationship with k")
plt.xlabel("k")
plt.ylabel("Testing accuracy %")
plt.legend
plt.grid(True)
plt.figure(figsize=(8, 6))
plt.plot(k_values, mse_values, marker='o', label="MSE value", color='yellow')
plt.title("MSE relationship with k")
plt.xlabel("k")
plt.ylabel("MSE value")
plt.legend
plt.grid(True)
plt.show()





