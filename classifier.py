# -*- coding: utf-8 -*-

from sklearn import svm, decomposition, metrics
import sklearn.cross_validation as cross_val
import numpy as np


def run_svm(params, X):

    # normalize features
    X = normalize(params, X)

    # get hyperparameters
    C = params['C']
    num_folds = params['num_folds']
    num_trials = params['num_trials']
    num_actors = params['num_actors']
    num_emotions = params['num_emotions']
    labels = params['labels']
     
    print('Running Linear SVM with %d-fold Cross Validation' % (num_folds))

    overall_acc = 0.0
    cnf = np.zeros((num_emotions, num_emotions))

    for trial in range(num_trials):
        kf = cross_val.KFold(num_actors, num_folds, shuffle=True)

        acc_sum = 0.0
        count = 1
        for train_idx, test_idx in kf:
            X_train, y_train = get_folds(params, train_idx, X)
            X_test, y_test = get_folds(params, test_idx, X)

            X_train_pca, model = run_pca(params, X_train)
            X_test_pca = model.transform(X_test)

            svc = svm.SVC(kernel='linear', C=C)
            svc.fit(X_train_pca, y_train)            
            acc = svc.score(X_test_pca, y_test)
            y_pred = svc.predict(X_test_pca)
            cnf += metrics.confusion_matrix(y_test, y_pred)

            print('Trial %2d, Fold %2d of %2d: Accuracy = %.2f\r' \
                % (trial+1, count, num_folds, acc), end='', flush=True)
            acc_sum += acc
            count += 1

        print('\nMean Accuracy after %d folds = %.2f' % (num_folds, acc_sum / num_folds))
        print()
        
        overall_acc += acc_sum / num_folds

    print('Mean Accuracy after %d trials = %.2f' % (num_trials, overall_acc / num_trials))

    cnf /= num_folds
    print_cnf(cnf, labels)


# Perform Principal Component Analysis
def run_pca(params, X):
    n_components = params['num_components']
    whiten = params['whiten']

    model = decomposition.PCA(n_components=n_components, whiten=whiten)
    model.fit(X)
    X_pca = model.transform(X)
    
    return (X_pca, model)


# Split dataset into folds
def get_folds(params, idxs, X):

    D = X[1].shape[1] - 1
    num_actors = params['num_actors']

    count = 0
    for i in range(idxs.shape[0]):
        actor = idxs[i] + 1
        count += X[actor].shape[0]

    X_split = np.zeros((count, D))
    y_split = np.zeros((count,))

    count = 0
    for i in range(idxs.shape[0]):
        actor = idxs[i] + 1
        rows_per_actor = X[actor].shape[0]
        for row in range(rows_per_actor):
            X_split[count, :] = X[actor][row, 1:]
            y_split[count] = X[actor][row, 0]
            count += 1

    return (X_split, y_split)


# L2 normalization
def normalize(params, X):
    num_actors = params['num_actors']
    X_norm = {}
    for actor in range(1, num_actors+1):
        rows_per_actor = X[actor].shape[0]
        X_norm[actor] = np.zeros_like(X[actor])
        for row in range(rows_per_actor):
            X_norm[actor][row, 0] = X[actor][row, 0]
            X_norm[actor][row, 1:] = X[actor][row, 1:] / np.linalg.norm(X[actor][row, 1:])

    return X_norm


# Print confusion matrix
def print_cnf(cnf, labels):
    s = '{:^10}|'.format('')
    for i in range(len(labels)):
        s += '{:^10}|'.format(labels[i])
    print(s)

    for i in range(len(labels)):
        s = '{:^10}|'.format(labels[i])
        for x in range(len(labels)):
            s += '{:^10.1f}|'.format(cnf[i,x])
        print(s)
    

