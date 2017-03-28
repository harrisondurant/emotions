# emotions

**All necessary flags/parameters should be defined in parameter file 'params.json'**

Usage of 'start.py':

```python start.py -p params.json```

Requirements: 
* json
* pickle/cPickle
* sklearn (for PCA)
* scipy.misc (for reading in, saving, and resizing images. could also use opencv)
* matplotlib.pyplot
* numpy
* os
* argparse

Generating dataset (tilburg.pickle):
* To generate dataset pickle file, set ```"gen_imageset": true``` in parameter file
* 348 images + 348 horizontally flipped images, each of size (200, 75)
* will contain a dictionary with keys for each actor (numbered 1-50)
* ```dataset[1]``` contains a numpy array of images for actor 1
* array size = (14, 15001) with one image per row, with exception of actors 26 and 30, which have size = (12, 15001)
* first number in each row is the image's ground truth label, so for some image 'n',
  ```dataset[1][n,0]``` contains its label [0-6].
* labels are as follows: ```[0=neutral, 1=anger, 2=disgust, 3=fear, 4=happy, 5=sad, 6=surprise]```

Generating features:
* To generate Log-Gabor features, set ```"gen_log_gabor": true``` in parameter file.
* All hyperparameters such as #scales and #orientations should be defined in parameter file
* Features are generated one batch at a time by default.
* To generate all features at once, set ```"use_batches": false```

Generating PCA Features:
* To generate PCA features set ```"run_pca": true```
* Default number of components is 365
* Output file will contain a dictionary with the following keys: 
* 'model', containing the sklearn pca object for transformation
* 'pca_feat', containing a dictionary with keys for each actor (numbered 1-50)
* ```pca_dict['pca_feat'][1]``` contains numpy array of pca transformed features for actor 1

Generating Frequency Response Grid Images:
* You can also generate a frequency response grid for an image at all scales and orientations.
* set ```"gen_grid_imgs": true``` and supply path to image file as well as the name for the output image file
