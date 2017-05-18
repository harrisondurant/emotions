# -*- coding: utf-8 -*-

import scipy.misc as misc
from os import path, listdir
import numpy as np
import argparse
import json
import pickle
import classifier
import pca
import lg


def main(params):
    if params['use_batches']:
        batch_no = params['curr_batch']
        num_batches = params['num_batches']
        if batch_no > num_batches or batch_no < 1:
            print('ERROR - batch number: please enter a number 1-%d' % (num_batches))
            return

    if params['gen_imageset']:
        generate_tilburg_imageset(params)
    
    if params['gen_log_gabor']:
        generate_features(params)    

    if params['run_pca']:
        features = load_full_features(params)
        pca.generate_pca_features(params, features)

    if params['run_svm']:
        features = load_full_features(params)
        classifier.run_svm(params, features)

    if params['gen_grid_imgs']:
        imgpath = params['grid_img_path']
        outfile_name = params['outfile_name']
        scales = params['scales']
        orientations = params['orientations']
        im = misc.imresize(misc.imread(imgpath), (200, 75))
        features = lg.get_log_gabor_features(params, im, scales, orientations)
        print('Generating frequency response image for %s:' % (imgpath))
        response_img = freq_response_grid(params, features)
        misc.imsave(outfile_name, response_img)


def generate_features(params):
    """
    This function generates Log-Gabor features for each image in the Tilburg imageset,
    concatenates features into a numpy matrix, and writes to an output file.
    """
    dataset = load_tilburg_imageset(params)

    scales = params['scales']
    orientations = params['orientations']

    print('Generating Log-Gabor features')
    print('%d scales and %d orientations' % (scales, orientations))

    H,W = params['height'],params['width']
    D = H * W

    if params['max-pool']:
        pool_height = params['pool_height']
        pool_width = params['pool_width']
        stride = params['stride']
        h_ = int((H - pool_height) / stride + 1)
        w_ = int((W - pool_width) / stride + 1)
        D = h_ * w_ + 1

    features = {}
    
    for actor in dataset.keys():
        N = dataset[actor].shape[0]
        features[actor] = np.zeros((N, (D-1) * scales * orientations + 1))
        for n in range(N):
            print('Generating Log-Gabor features for actor %2s, image %3d of %3d\r' \
                    % (actor, n+1, N), end='', flush=True)
            im = dataset[actor][n,1:].reshape(H, W)
            features[actor][n, 0] = dataset[actor][n, 0]
            features[actor][n, 1:] = lg.get_log_gabor_features(params, im, scales, orientations)

    print()

    outfile_name = 'full_features.pickle'

    if not path.exists(outfile_name):
        open(outfile_name, 'wb').close()
    
    with open(outfile_name, 'wb') as curr_batch:
        pickle.dump(features, curr_batch, protocol=0)


def generate_tilburg_imageset(params):
    """
    This function generates pickle file containing a numpy array of Tilburg images
    """
    image_dir = params['image_dir']
    dataset_path = params['dataset_path']

    H = params['height']
    W = params['width']
    dataset = {}
    emotions = ['neu', 'ang', 'dis', 'fea', 'hap', 'sad', 'sur']
    
    print('Generating Tilburg imageset')

    count = 0
    actor_number = 1
               
    for folder in listdir(image_dir):

        actor_array = np.zeros((0, 1 + H*W))
        subject = path.join(image_dir,folder)

        if path.isdir(subject):
            for img_name in listdir(subject):

                for idx,key in enumerate(emotions):
                    if key in img_name:
                        if '_' in img_name and not 'montage' in img_name:
                            # read in the image
                            img = misc.imread(path.join(subject, img_name))
                        
                            # the first number in each row will be the label 
                            label = np.array([idx]).reshape((1, 1))

                            # add image and flipped image to dataset 
                            img = misc.imresize(img, (H, W))
                            img_flip = np.fliplr(img)
                            img = img.reshape(1, H*W)
                            img_flip = img_flip.reshape(1, H*W)
                            img_vec = np.hstack((label, img))
                            img_flip_vec = np.hstack((label, img_flip))

                            actor_array = np.vstack((actor_array, img_vec, img_flip_vec))
                            
                            print('%s.%s : %d' % (folder, key, count))
                            count += 1
                            print('%s.%s (flipped): %d' % (folder, key, count))
                            count += 1

            dataset[actor_number] = actor_array
            actor_number += 1


    if not path.exists(dataset_path):
        open(dataset_path, 'wb').close()
    
    with open(dataset_path, 'wb') as tilburg:
        pickle.dump(dataset, tilburg, protocol=0)

    return dataset


# Load full set of Log-Gabor features
def load_full_features(params):
    print('Loading full set of Log-Gabor features')

    features = None

    if params['use_batches']:
        num_batches = params['num_batches']
        features = load_batch(1)    
    
        for i in range(2,num_batches+1):
            features = np.vstack((features, load_batch(i)))
    else:
        with open('full_features.pickle', 'rb') as full_features:
            features = pickle.load(full_features)

    return features


# Load a batch of images
def load_batch(batch_no):
    filename = 'batch-%d.pickle' % (batch_no)
    print('Loading batch %d' % (batch_no))

    batch = None
    with open(filename, 'rb') as curr_batch:
        batch = pickle.load(curr_batch)

    return batch


# Load the image dataset
def load_tilburg_imageset(params):

    dataset_path = params['dataset_path']

    dataset = {}
    with open(dataset_path, 'rb') as tilburg:
        dataset = pickle.load(tilburg)
    
    return dataset


# Generate a grid of frequency responses for an image at each scale and orientation
def freq_response_grid(params, arr):
    """
    returns an image containing a grid of frequency 
    response images at each scale and orientation
    """ 
    rows = params['scales']
    cols = params['orientations']

    print('%d scales and %d orientations' % (rows, cols))
    
    H,W = params['height'],params['width']
    if params['max-pool']:
        pool_height = params['pool_height']
        pool_width = params['pool_width']
        stride = params['stride']
        H = int((H - pool_height) / stride + 1)
        W = int((W - pool_width) / stride + 1)
        
    scale = params['scale']
    spacing = params['spacing']
    H_ = int(H * scale)
    W_ = int(W * scale)
    
    grid = np.ones((W_*cols + spacing*(cols+1), H_*rows + spacing*(rows+1))) * 255
    
    count = 0
    xpos = spacing
    for j in range(cols):
        ypos = spacing
        for i in range(rows):
            idx = count * (H * W)
            im = arr[idx:idx+H*W].reshape(H,W).T
            im = misc.imresize(im,scale)
            grid[xpos:xpos+W_,ypos:ypos+H_] = im
            count += 1
            ypos += H_ + spacing
        xpos += W_ + spacing
    
    return grid.T


# Parse parameters from file
def get_params(filepath):
    params = None
    with open(filepath) as param_file:
        params = json.load(param_file)

    return params


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--param', help='path to parameter file')
    args = parser.parse_args()
    
    params = get_params(args.param)
    
    main(params)
