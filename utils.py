import itertools
import math
import os
import random
import sys
import time

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

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


class Display():
    def __init__(self, args):
        self.display_id = args.display_id
        self.win_size = args.display_winsize
        self.name = args.name
        self.args = args
        if self.display_id > 0:
            import visdom
            self.ncols = args.display_ncols
            self.vis = visdom.Visdom(server=args.display_server, port=args.display_port, env=args.display_env, raise_exceptions=True)

        dir = os.path.join(args.checkpoints_dir, args.name)
        mkdir(args.checkpoints_dir)
        mkdir(dir)
        self.log_name = os.path.join(dir, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)


    def throw_visdom_connection_error(self):
        print('\n\nCould not connect to Visdom server (https://github.com/facebookresearch/visdom) for displaying training progress.\nYou can suppress connection to Visdom using the option --display_id -1. To install visdom, run \n$ pip install visdom\n, and start the server by \n$ python -m visdom.server.\n\n')
        exit(1)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals):
        if self.display_id > 0:  # show images in the browser
            ncols = self.ncols
            if ncols > 0:
                label = 'Results'

                try:
                    self.vis.image(visuals, opts=dict(caption=str(label), store_history=False))
                except VisdomExceptionBase:
                    self.throw_visdom_connection_error()

            else:
                idx = 1
                label = 'Results'

                self.vis.image(visuals, opts=dict(title=label), win=self.display_id + idx)
                idx += 1

    # losses: dictionary of error labels and values
    def plot_current_loss(self, step, loss):

        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': ['Loss']}
        self.plot_data['X'].append(step)
        self.plot_data['Y'].append([np.float(loss)])

        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' -- loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'step',
                    'ylabel': 'training loss'},
                win=self.display_id)

        except VisdomExceptionBase:
            self.throw_visdom_connection_error()

    # losses: same format as |losses| of plot_current_losses
    def print_current_loss(self, i, loss, t, t_data):
        message = '(step: %d, time: %.4f, data: %.4f) ' % (i, t, t_data)

        message += '%s: %.5f ' % ('loss', loss)

        print(message)

        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

#-----------------------------------------
# calculates the f1 score
def calculate_f1_score(gt, pred, average="weighted"):

    return metrics.f1_score(gt, pred, average=average)
#-----------------------------------------

#-----------------------------------------
# calculates the precision and recall
def calculate_precision_recall(gt, pred):

    cm = metrics.confusion_matrix(gt, pred)
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    precision = np.diag(cm) / np.sum(cm, axis = 0)

    return np.mean(precision), np.mean(recall)
#-----------------------------------------

#-----------------------------------------
# calculates the ROC AUC for multi-class 
def multiclass_roc_auc_score(gt, pred, average="macro"):

    binarizer = preprocessing.LabelBinarizer()
    binarizer.fit(gt)

    gt = binarizer.transform(gt)
    pred = binarizer.transform(pred)

    return metrics.roc_auc_score(gt, pred, average=average)
#-----------------------------------------

# this function is used to make a directory if it does not already exist
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

