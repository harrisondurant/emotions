# -*- coding: utf-8 -*-

import numpy as np

def get_log_gabor_features(params, im, scales, orientations):
    """
    Copyright (c) 2001-2010 Peter Kovesi
    School of Computer Science & Software Engineering
    The University of Western Australia
    http://www.csse.uwa.edu.au/

    Original function was written by Peter Kovesi in Matlab,
    and can be found in /matlab/gaborconvolve.m

    This is my translation into python
    """
    H,W = im.shape

    # calculate necessary max pool dimensions
    pool_height = params['pool_height']
    pool_width = params['pool_width']
    stride = params['stride']

    # some hyperparameters
    Lnorm = params['Lnorm']
    minWaveLength = params['minWaveLength']
    mult = params['mult']
    sigmaOnf = params['sigmaOnf']
    dThetaOnSigma = params['dThetaOnSigma']
    
    imagefft = np.fft.fft2(im)

    x_range, y_range = None, None
    
    if W % 2:
        x_range = np.arange(-(W-1)/2,(W-1)/2+1) / (W-1)
    else:
        x_range = np.arange(-W/2,(W/2)) / W

    if H % 2:
        y_range = np.arange(-(H-1)/2,(H-1)/2+1) / (H-1)
    else:
        y_range = np.arange(-H/2,(H/2)) / H
    
    x,y = np.meshgrid(x_range, y_range)
    
    radius = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y,x)
    
    radius = np.fft.ifftshift(radius)
    theta  = np.fft.ifftshift(theta)

    radius[0,0] = 1
    
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    # Radius .45, 'sharpness' 15
    lowpass = (1.0 / (1.0 + (radius / 0.45)**(2 * 15)))

    lG = {}

    h_ = int((H - pool_height) / stride + 1)
    w_ = int((W - pool_width) / stride + 1)

    if not params['max-pool']:
        h_ = H
        w_ = W

    response_vec = np.zeros((scales * orientations * h_ * w_))
    
    for s in range(scales):
        wavelength = minWaveLength * mult**s
        fo = 1.0/wavelength
        lG[s] = np.exp((-(np.log(radius/fo))**2) / (2 * np.log(sigmaOnf)**2))

        lG[s] = lG[s] * lowpass
        lG[s][0,0] = 0
        
        L = 1       
        if Lnorm == 2:
            L = np.sqrt(np.sum(lG[s]**2))
        elif Lnorm == 1:
            L = np.sum(np.absolute(np.real(np.fft.ifft2(lG[s]))))
        
        lG[s] /= L

    # for successive storage of img vectors
    count = 0

    for o in range(orientations):
        
        angl = o * np.pi/orientations
        wavelength = minWaveLength

        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
        
        dtheta = np.abs(np.arctan2(ds,dc))
        
        thetaSigma = np.pi/orientations/dThetaOnSigma
        spread = np.exp((-dtheta**2) / (2 * thetaSigma**2))
        
        for s in range(scales):
            filter = lG[s] * spread

            L = 1       
            if Lnorm == 2:
                L = np.sqrt(np.sum(np.real(filter)**2 + np.imag(filter)**2) / np.sqrt(2))
            elif Lnorm == 1:
                L = np.sum(np.absolute(np.real(np.fft.ifft2(lG[s]))))
            
            filter /= L

            idx = count * (h_ * w_)

            lg_im = np.absolute(np.fft.ifft2(imagefft * filter).reshape(1, H * W))

            if params['max-pool']:
                lg_im = max_pool(params, lg_im.reshape(H,W)).reshape(1, h_*w_)

            response_vec[idx:idx+h_*w_] = lg_im

            wavelength = wavelength * mult
            count += 1

    return response_vec


# max-pooling function 
def max_pool(params, im):
    H,W = im.shape
    pool_height = params['pool_height']
    pool_width = params['pool_width']
    stride = params['stride']

    h_ = int((H - pool_height) / stride + 1)
    w_ = int((W - pool_width) / stride + 1)
    out = np.zeros((h_, w_))

    for h in range(h_):
        for w in range(w_):
            h1 = h * stride
            h2 = h * stride + pool_height
            w1 = w * stride
            w2 = w * stride + pool_width
            out[h,w] = np.max(im[h1:h2,w1:w2])

    return out


