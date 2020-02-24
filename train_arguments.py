import argparse
import os

class Arguments():

    def __init__(self):

        self.initialized = False

    def initialize(self, parser):

        parser.add_argument('name', type=str, help='experiment name.')
        parser.add_argument('--phase', default='train', type=str, choices=['train', 'test'], help='determining whether the model is being trained or used for inference. Since this is the train_arguments file, this needs to set to train!!')
        parser.add_argument('--data_root', default='../../Data/ransom_ware/train', type=str, help='path to the training data directory.')
        parser.add_argument('--num_classes', default=50, type=int, help='number of classes in the classification task.')
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--num_steps', default=24000, type=int, help='number of steps for which the model is trained.')
        parser.add_argument('--break_count', default=600, type=int, help='how many steps to before training is stopped when the loss value does not change.')
        parser.add_argument('--arch', type=str, default='AmirNet', help='which architecture is used to create the classifier', choices=['inception', 'resnet34', 'resnet50', 'resnet101', 'resnext50', 'resnext101', 'densenet161', 'densenet169', 'densenet201', 'vgg16_bn', 'vgg19_bn', 'squeezenet', 'shufflenet', 'mobilenet', 'AmirNet', 'AmirNet_DO', 'AmirNet_CDO', 'AmirNet_VDO'])
        parser.add_argument('--augs', nargs='+', help='which augmentations are used to help in the training process', choices=['rotate', 'vflip', 'hflip', 'contrast', 'brightness', 'noise', 'occlusion', 'regularblur', 'defocusblur', 'motionblur', 'perspective', 'gray', 'colorjitter'])
        parser.add_argument('--input_size', type=int, default=128, help='size of the input image.')
        parser.add_argument('--pretrained', action='store_true', help='the model is initialized with weights pre-trained on imagenet.')
        parser.add_argument('--num_workers', default=2, type=int, help='number of workers used in the dataloader.')
        parser.add_argument('--lr', type=float, default=5e-4, help='learning rate.')
        parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay.')
        parser.add_argument('--resume', action='store_true', help='resume from a checkpoint')
        parser.add_argument('--which_checkpoint', type=str, default='latest', help='the checkpoint to be loaded to resume training. Checkpoints are identified and saved by the number of steps passed during training.')
        parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='the path to where the model is saved.')
        parser.add_argument('--print_freq', default=50, type=int, help='how many steps before printing the loss values to the standard output for inspection purposes only.')
        parser.add_argument('--display', action='store_true', help='display the results periodically via visdom')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for visdom.')
        parser.add_argument('--display_freq', type=int, default=50, help='frequency of showing training results on screen using visdom.')
        parser.add_argument('--display_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display.')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display.')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main").')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display.')
        parser.add_argument('--save_checkpoint_freq', default=5000, type=int, help='how many steps before saving one sequence of images to disk for inspection purposes only.')

        self.initialized = True

        return parser

    def get_args(self):

        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_args(self, args):

        txt = '\n'

        txt += '-------------------- Arguments --------------------\n'

        for k, v in sorted(vars(args).items()):

            comment = ''
            default = self.parser.get_default(k)

            if v != default:
                comment = '\t[default: %s]' % str(default)

            txt += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)

        txt += '----------------------- End -----------------------'
        txt += '\n'

        print(txt)

    def parse(self):

        args = self.get_args()
        self.print_args(args)
        self.args = args

        return self.args
