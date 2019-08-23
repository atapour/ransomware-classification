import itertools
import math
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import PIL
import PIL.ImageFile
import sklearn.metrics as metrics
import sklearn.preprocessing as preprocessing
import torch
from PIL import Image, ImageDraw
from scipy.signal import convolve2d
from skimage.draw import circle, line
from torchvision import datasets


#-----------------------------------------
# add noise to an image, already converted to a torch tensor
class AddNoise(object):

    def __call__(self, x):
        l1 = 0.000001
        l2 = 0.002
        level = random.uniform(l1, l2)
        noise = torch.randn(*(x.size())) * level

        return x + noise
#-----------------------------------------

#-----------------------------------------
# simulates defocus blur. Taken from https://github.com/lospooky/pyblur/blob/master/pyblur/DefocusBlur.py
class AddDefocusBlur(object):
    def __init__(self):
        self.defocusKernelDims = [3, 5, 7, 9]
        self.coin = 0.3

    def __call__(self, x):

        if random.random() < self.coin:

            kernelidx = random.randint(0, len(self.defocusKernelDims)-1)    
            kerneldim = self.defocusKernelDims[kernelidx]

            return self.DefocusBlur(x, kerneldim)
        return x

    def DefocusBlur(self, img, dim):
        imgarray = np.array(img, dtype="float32")
        kernel = self.DiskKernel(dim)

        if imgarray.ndim==3 and imgarray.shape[-1]==3:
            convolved = np.stack([convolve2d(imgarray[...,channel_id], 
                        kernel, mode='same', 
                        fillvalue=255.0).astype("uint8") 
                        for channel_id in range(3)], axis=2)
        else:
            convolved = convolve2d(imgarray, kernel, mode='same', fillvalue=255.0).astype("uint8")        

        img = Image.fromarray(convolved)

        return img

    def DiskKernel(self, dim):
        kernelwidth = dim
        kernel = np.zeros((kernelwidth, kernelwidth), dtype=np.float32)
        circleCenterCoord = dim / 2
        circleRadius = circleCenterCoord +1
        
        rr, cc = circle(circleCenterCoord, circleCenterCoord, circleRadius)
        kernel[rr-1,cc-1]=1
        
        if(dim == 3 or dim == 5):
            kernel = self.Adjust(kernel, dim)
            
        normalizationFactor = np.count_nonzero(kernel)
        kernel = kernel / normalizationFactor

        return kernel

    def Adjust(self, kernel, kernelwidth):
        kernel[0,0] = 0
        kernel[0,kernelwidth-1]=0
        kernel[kernelwidth-1,0]=0
        kernel[kernelwidth-1, kernelwidth-1] =0 

        return kernel
#-----------------------------------------

#-----------------------------------------
# simulate motion blur. Take from https://github.com/lospooky/pyblur/blob/master/pyblur/LinearMotionBlur.py
class AddMotionBlur(object):
    def __init__(self):
        self.lineLengths = [3, 5, 7]
        self.lineTypes = ["full", "right", "left"]
        self.lineDict = LineDictionary()
        self.coin = 0.3

    def __call__(self, x):

        if random.random() < self.coin:

            lineLengthIdx = random.randint(0, len(self.lineLengths)-1)
            lineTypeIdx = random.randint(0, len(self.lineTypes)-1) 
            lineLength = self.lineLengths[lineLengthIdx]
            lineType = self.lineTypes[lineTypeIdx]
            lineAngle = self.randomAngle(lineLength)

            return self.LinearMotionBlur(x, lineLength, lineAngle, lineType)
        else:
            return x

    def LinearMotionBlur(self, img, dim, angle, linetype):
        imgarray = np.array(img, dtype="float32")

        kernel = self.LineKernel(dim, angle, linetype)

        if imgarray.ndim==3 and imgarray.shape[-1]==3:
            convolved = np.stack([convolve2d(imgarray[...,channel_id], 
                        kernel, mode='same', 
                        fillvalue=255.0).astype("uint8") 
                        for channel_id in range(3)], axis=2)
        else:
            convolved = convolve2d(imgarray, kernel, mode='same', fillvalue=255.0).astype("uint8")

        img = Image.fromarray(convolved)

        return img

    def LineKernel(self, dim, angle, linetype):
        kernelwidth = dim
        kernelCenter = int(math.floor(dim/2))
        angle = self.SanitizeAngleValue(kernelCenter, angle)
        kernel = np.zeros((kernelwidth, kernelwidth), dtype=np.float32)
        lineAnchors = self.lineDict.lines[dim][angle]
        if(linetype == 'right'):
            lineAnchors[0] = kernelCenter
            lineAnchors[1] = kernelCenter
        if(linetype == 'left'):
            lineAnchors[2] = kernelCenter
            lineAnchors[3] = kernelCenter
        rr,cc = line(lineAnchors[0], lineAnchors[1], lineAnchors[2], lineAnchors[3])
        kernel[rr,cc]=1
        normalizationFactor = np.count_nonzero(kernel)
        kernel = kernel / normalizationFactor        

        return kernel

    def SanitizeAngleValue(self, kernelCenter, angle):
        numDistinctLines = kernelCenter * 4
        angle = math.fmod(angle, 180.0)
        validLineAngles = np.linspace(0,180, numDistinctLines, endpoint=False)
        angle = self.nearestValue(angle, validLineAngles)

        return angle

    def nearestValue(self, theta, validAngles):
        idx = (np.abs(validAngles-theta)).argmin()

        return validAngles[idx]

    def randomAngle(self, kerneldim):
        kernelCenter = int(math.floor(kerneldim/2))
        numDistinctLines = kernelCenter * 4
        validLineAngles = np.linspace(0,180, numDistinctLines, endpoint = False)

        top_idx = len(validLineAngles) - 1
        if top_idx < 0:
            top_idx = 0

        angleIdx = random.randint(0, top_idx)

        return int(validLineAngles[angleIdx])
#-----------------------------------------

#-----------------------------------------
# blur an image. Regular old box and gaussian blurs randomly selected
class AddBlur(object):

    def __call__(self, x):

        coin_1 = 0.1
        coin_2 = 0.3

        radius = 3

        if random.random() < coin_1:
            return x.filter(PIL.ImageFilter.BoxBlur(radius=random.randint(0, radius)))

        elif random.random() < coin_2:
            return x.filter(PIL.ImageFilter.GaussianBlur(radius=random.randint(0, radius)))

        else:
            return x
#-----------------------------------------

#-----------------------------------------
# create an occlusion over the images, mainly to simulate glare
class AddOcclusion(object):
    def __call__(self, x):

        make_colour = lambda : (random.randint(170, 255), random.randint(140, 230), random.randint(1, 80), random.randint(1, 255))

        # coin = 0.4
        coin = 0.1

        if random.random() < coin:

            w, h = x.size

            radius = min(w // 2, h // 2)

            radius_w = random.randint(1, radius)
            radius_h = random.randint(1, radius)

            pos_h = random.randint(1, h - radius_h)
            pos_w = random.randint(1, w - radius_w)

            draw = ImageDraw.Draw(x)
            draw.ellipse((pos_h, pos_w, pos_h + radius_h, pos_w + radius_w), make_colour())
        return x
#-----------------------------------------

#-----------------------------------------
# change contrast
class ChangeContrast(object):

    def __call__(self, x):

        level = random.randint(1, 256)
        factor = (259 * (level + 255)) / (255 * (259 - level))

        def contrast(c):
            return 128 + factor * (c - 128)

        return x.point(contrast)
#-----------------------------------------

#-----------------------------------------
# change brightness
class ChangeBrightness(object):

    def __call__(self, x):
        l1 = 0
        l2 = 2
        level = random.uniform(l1, l2)
        enhancer = PIL.ImageEnhance.Brightness(x)
        return enhancer.enhance(level)
#-----------------------------------------

#-----------------------------------------
# randomly changes perspective. it is supposed to be in torchvision transforms but not what we have
class RandomPerspective(object):

    def __init__(self, distortion_scale=0.5, p=0.5, interpolation=Image.BICUBIC):
        self.p = p
        self.interpolation = interpolation
        self.distortion_scale = distortion_scale

    def __call__(self, img):

        if random.random() < self.p:
            width, height = img.size
            startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
            return self.perspective(img, startpoints, endpoints, self.interpolation)

        return img

    def get_params(self, width, height, distortion_scale):

        half_height = int(height / 2)
        half_width = int(width / 2)
        topleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(0, int(distortion_scale * half_height)))
        topright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(0, int(distortion_scale * half_height)))
        botright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        botleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

    def get_perspective_coeffs(self, startpoints, endpoints):

        matrix = []

        for p1, p2 in zip(endpoints, startpoints):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

        A = torch.tensor(matrix, dtype=torch.float)
        B = torch.tensor(startpoints, dtype=torch.float).view(8)
        res = torch.gels(B, A)[0]

        return res.squeeze_(1).tolist()

    def perspective(self, img, startpoints, endpoints, interpolation=Image.BICUBIC):

        coeffs = self.get_perspective_coeffs(startpoints, endpoints)

        return img.transform(img.size, Image.PERSPECTIVE, coeffs, interpolation)
#-----------------------------------------

#-----------------------------------------
# for the purposes of blurring. Taken from https://github.com/lospooky/pyblur/blob/master/pyblur/LineDictionary.py
class LineDictionary:
    def __init__(self):
        self.lines = {}
        self.Create3x3Lines()
        self.Create5x5Lines()
        self.Create7x7Lines()
        self.Create9x9Lines()
        return

    def Create3x3Lines(self):
        lines = {}
        lines[0] = [1,0,1,2]
        lines[45] = [2,0,0,2]
        lines[90] = [0,1,2,1]
        lines[135] = [0,0,2,2]
        self.lines[3] = lines
        return

    def Create5x5Lines(self):
        lines = {}        
        lines[0] = [2,0,2,4]
        lines[22.5] = [3,0,1,4]
        lines[45] = [0,4,4,0]
        lines[67.5] = [0,3,4,1]
        lines[90] = [0,2,4,2]
        lines[112.5] = [0,1,4,3]
        lines[135] = [0,0,4,4]
        lines[157.5]= [1,0,3,4]
        self.lines[5] = lines
        return

    def Create7x7Lines(self):
        lines = {}
        lines[0] = [3,0,3,6]
        lines[15] = [4,0,2,6]
        lines[30] = [5,0,1,6]
        lines[45] = [6,0,0,6]
        lines[60] = [6,1,0,5]
        lines[75] = [6,2,0,4]
        lines[90] = [0,3,6,3]
        lines[105] = [0,2,6,4]
        lines[120] = [0,1,6,5]
        lines[135] = [0,0,6,6]
        lines[150] = [1,0,5,6]
        lines[165] = [2,0,4,6]
        self.lines[7] = lines 
        return

    def Create9x9Lines(self):
        lines = {}
        lines[0] = [4,0,4,8]
        lines[11.25] = [5,0,3,8]
        lines[22.5] = [6,0,2,8]
        lines[33.75] = [7,0,1,8]
        lines[45] = [8,0,0,8]
        lines[56.25] = [8,1,0,7]
        lines[67.5] = [8,2,0,6]
        lines[78.75] = [8,3,0,5]
        lines[90] = [8,4,0,4]
        lines[101.25] = [0,3,8,5]
        lines[112.5] = [0,2,8,6]
        lines[123.75] = [0,1,8,7]
        lines[135] = [0,0,8,8]
        lines[146.25] = [1,0,7,8]
        lines[157.5] = [2,0,6,8]
        lines[168.75] = [3,0,5,8]
        self.lines[9] = lines
        return
#-----------------------------------------
