# -*- coding: utf-8 -*-

from sklearn import decomposition as dcmp
from matplotlib import pyplot as plt
from os import path
import numpy as np
import pickle

def generate_pca_features(params, features):
    f = {}
    _,D = features[1].shape
    num_images = params['num_images']
    num_actors = params['num_actors']
    idxs = np.zeros((num_images, 1))
    y = np.zeros((num_images, 1))
    X = np.zeros((num_images, D-1))

    count = 0
    for actor in range(1, num_actors+1):
        rows_per_actor = features[actor].shape[0]
        for row in range(rows_per_actor):
            X[count, :] = features[actor][row, 1:]
            y[count] = features[actor][row, 0]
            idxs[count] = int(actor)
            count += 1

    
    n_components = params['num_components']
    whiten = params['whiten']

    print('Extracting PCA features with %d components' % (n_components))
    
    pca = dcmp.PCA(n_components=n_components, whiten=whiten)
    pca.fit(X)
    X_pca = pca.transform(X)
    
    pca_feat = {}

    for actor in range(1, num_actors+1):
        pca_feat[actor] = np.zeros((0, n_components+1))

    for n in range(num_images):
        actor = int(idxs[n])
        label = y[n]
        pca_vec = np.hstack((label, X_pca[n, :]))
        pca_feat[actor] = np.vstack((pca_feat[actor], pca_vec))
        

    f['pca_feat'] = pca_feat
    f['model'] = pca
 
    if not path.exists('pca_features.pickle'):
        open('pca_features.pickle', 'wb').close()
    
    with open('pca_features.pickle', 'wb') as pca_dict:
        pickle.dump(f, pca_dict, protocol=0)

