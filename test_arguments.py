import argparse
import os

class Arguments():

    def __init__(self):

        self.initialized = False

    def initialize(self, parser):

        parser.add_argument('--phase', default='test', type=str, choices=['train', 'test'], help='determining whether the model is being trained or used for inference. Since this is the test_arguments file, this needs to be test!!')
        parser.add_argument('--test_checkpoint_path', type=str, help='during inference, the path to checkpoint is needed.')
        parser.add_argument('--pos_root', type=str, help='path to the positive test data directory to test the accuracy of the model.')
        parser.add_argument('--neg_root', type=str, help='path to the negative test data directory to evaluate model uncertainty.')
        parser.add_argument('--batch_size', default=1, type=int, help='It is the size of your batch.')
        parser.add_argument('--num_workers', default=2, type=int, help='number of workers used in the dataloader.')
        parser.add_argument('--num_classes', default=50, type=int)
        parser.add_argument('--arch', type=str, default='AmirNet', help='which architecture is used to create the classifier', choices=['inception', 'resnet34', 'resnet50', 'resnet101', 'resnext50', 'resnext101', 'densenet161', 'densenet169', 'densenet201', 'vgg16_bn', 'vgg19_bn', 'squeezenet', 'shufflenet', 'mobilenet', 'AmirNet', 'AmirNet_DO', 'AmirNet_CDO', 'AmirNet_VDO'])
        parser.add_argument('--num_samples', default=100, type=int, help='how many samples do we get from the network for monte carlo sampling.')
        parser.add_argument('--input_size', type=int, default=128, help='size of the input image. Should be the same as what was used during training.')
        parser.add_argument('--pretrained', action='store_true', help='prior to training, the model was initialized with weights pre-trained on imagenet.')
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
