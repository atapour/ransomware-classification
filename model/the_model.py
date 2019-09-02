import glob
import os
from collections import OrderedDict

import colorama
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from model.network import load_model


class TheModel():
    """
    This class used to represent our overall model
    All the functionalities of the network with respect to both training and testing is implemented in this class.
    """

    def initialize(self, args, weights, classes):
        """
        Initialize all requirements for the model

        Parameters
        ----------
        args : arguments class

        weights: numpy array
            weights used for balancing the classes during training
        classes: list of strings
            name of the classes in the dataset
        """

        self.args = args
        self.phase = args.phase
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.net = load_model(args)
        self.net.to(self.device)

        self.classes = classes

        if self.phase == 'train':
            self.checkpoint_save_dir = os.path.join(args.checkpoints_dir, args.name)
            self.criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(weights).float())
            self.criterion = self.criterion.to(self.device)


            if args.arch == 'AmirNet_CDO':
                self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, amsgrad=True)
            else:
                self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)

    def set_up(self, args):
        """
        Set up the model by loading and printing the model if necessary

        Parameters
        ----------
        args : arguments class
        """

        if self.phase == 'test':
            if args.test_checkpoint_path is not None:

                print('loading the checkpoint from %s' % args.test_checkpoint_path)

                state_dict = torch.load(args.test_checkpoint_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                if 'state_dict' in state_dict.keys():
                    state_dict = state_dict['state_dict']

                self.net.load_state_dict(state_dict)

            else:
                raise Exception('For inference, a checkpoint path must be passed as an argument.')

        else:
            if args.resume:
                if not os.listdir(self.checkpoint_save_dir):
                    raise Exception('The specified checkpoints directory is empty. Resuming is not possible.')
                if args.which_checkpoint == 'latest':
                    checkpoints = glob.glob(os.path.join(self.checkpoint_save_dir, '*.pth'))
                    checkpoints.sort()
                    latest = checkpoints[-1]
                    step = latest.split('_')[1]
                elif args.which_checkpoint != 'latest' and args.which_checkpoint.isdigit():
                    step = args.which_checkpoint
                else:
                    raise Exception('The specified checkpoint to load is invalid.')
                self.load_networks(step)
        self.print_networks()

    def assign_inputs(self, input):
        """
        Assign inputs used by the model

        Parameters
        ----------
        input : the image and the ground truth label
        """

        self.image, self.gt = input

        self.image = self.image.to(self.device)
        self.gt = self.gt.to(self.device)

    def forward(self):
        """
        Pass the image through the network
        """

        self.out = self.net(self.image)

    def backward(self, args):
        """
        Take a backward pass and calculate the loss

        Parameters
        ----------
        args : arguments class
        """

        self.loss = self.criterion(self.out, self.gt)

        if args.arch == 'AmirNet_CDO' or args.arch == 'AmirNet_VDO':
            self.loss += torch.sum(self.net.regularisation())

        self.loss.backward()

    # optimize the model parameters
    def optimize(self, args):
        """
        Take a step in the model optimisation process

        Parameters
        ----------
        args : arguments class
        """

        self.net.train()

        self.forward()
        self.optimizer.zero_grad()
        self.backward(args)
        self.optimizer.step()

    def test(self):
        """
        Pass the image through the network for evaluation
        """

        self.net.eval()

        self.forward()

    def save_networks(self, step):
        """
        Save the network weights to disk

        Parameters
        ----------
        step: the number of step at which the saving takes place
        """

        save_filename = 'checkpoint_%s_steps.pth' % (step)
        save_path = os.path.join(self.checkpoint_save_dir, save_filename)

        print('saving the checkpoint to %s' % save_path)

        torch.save(self.net.state_dict(), save_path)

    def load_networks(self, step):
        """
        Load the network weights from disk

        Parameters
        ----------
        step: the number of step at which the network was previously saved
        """

        load_filename = 'checkpoint_%s_steps.pth' % (step)
        load_path = os.path.join(self.checkpoint_save_dir, load_filename)

        print('loading the checkpoint from %s' % load_path)

        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']

        self.net.load_state_dict(state_dict)

    def print_networks(self):
        """
        Print network details
        """

        # setting up the pretty colors:
        reset = colorama.Style.RESET_ALL
        blue = colorama.Fore.BLUE
        red = colorama.Fore.RED

        num_params = 0
        for param in self.net.parameters():
            num_params += param.numel()

        print(f'{blue}There are a total number of {red}{num_params} parameters{blue} in the model.{reset}')
        print('')

    def get_loss(self):
        """
        Print network details

        Returns
        ----------
        loss
        """

        return self.loss

    def get_train_images(self, step):
        """
        Generate an image that contains the training image, the prediction and the label

        Parameters
        ----------
        step: the number of step during the optimisation process
        """

        t = ToTensor()

        _, output = torch.max(self.out, dim=1)

        image = self.image[0]
        gt_txt = f'Step: {step} - {self.classes[self.gt[0]]}'
        output_txt = 'Pred: ' + self.classes[output[0]]


        delta_w = 500 - image.shape[1]
        delta_h = 0
        top, bottom = delta_h // 2, delta_h - (delta_h//2)
        left, right = delta_w // 2, delta_w - (delta_w//2)

        image = np.transpose(image.cpu().numpy(), (1, 2, 0))    

        image_padded = cv2.copyMakeBorder(image * 255, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        gt_colour = (0, 0, 0)

        if self.gt[0] == output[0]:
            output_colour = (0, 255, 0)
        else:
            output_colour = (255, 0, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX

        height = 40
        width = 500

        gt_img = np.ones((height, width, 3), np.uint8) * 255
        output_img = np.ones((height, width, 3), np.uint8) * 255

        # get boundary of this text
        textsize_gt = cv2.getTextSize(gt_txt, font, 1, 2)[0]
        textsize_output = cv2.getTextSize(output_txt, font, 1, 2)[0]

        # get coords based on boundary
        textX_gt = (gt_img.shape[1] - textsize_gt[0]) // 2
        textY_gt = (gt_img.shape[0] + textsize_gt[1]) // 2
        textX_output = (output_img.shape[1] - textsize_output[0]) // 2
        textY_output = (output_img.shape[0] + textsize_output[1]) // 2

        # add text centered on image
        cv2.putText(gt_img, gt_txt, (textX_gt, textY_gt), font, 1, gt_colour, 2)
        cv2.putText(output_img, output_txt, (textX_output, textY_output), font, 1, output_colour, 2)

        labelled_image = np.concatenate((image_padded, gt_img, output_img), axis=0)

        return t(labelled_image)

    def get_test_outputs(self):
        """
        Return the output image and the RGB image during testing

        Returns
        ----------
        ret: dictionary containing the image, ground truth label and the prediction
        """

        ret = OrderedDict()
        ret['image'] = self.image
        ret['gt'] = self.gt

        _, output = torch.max(self.out, dim=1)
        ret['out'] = output

        return ret
    
    def return_model(self):
        """
        Return the network

        Returns
        ----------
        net: the network
        """
        return self.net
