import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.modules.module import Module
from torchvision import datasets, models


# --------------------------------------------------
def load_model(args):
    """Load a given model specified in the arguments (--arch) and return it"""

    if args.arch == 'inception': # 25,214,714 parameters
        model = models.inception_v3(pretrained=args.pretrained, num_classes=1000, aux_logits=True)
        num_infea = model.fc.in_features
        model.aux_logits = False
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
        print("Inception has been loaded.")

    elif args.arch == 'resnet34': # 21,310,322 parameters
        model = models.resnet34(pretrained=args.pretrained)
        num_infea = model.fc.in_features
        model.fc = nn.Linear(num_infea, args.num_classes)
        print("ResNet34 has been loaded.")

    elif args.arch == 'resnet50': # 23,610,482 parameters
        model = models.resnet50(pretrained=args.pretrained)
        num_infea = model.fc.in_features
        model.fc = nn.Linear(num_infea, args.num_classes)
        print("ResNet50 has been loaded.")

    elif args.arch == 'resnet101': # 42,602,610 parameters
        model = models.resnet101(pretrained=args.pretrained)
        num_infea = model.fc.in_features
        model.fc = nn.Linear(num_infea, args.num_classes)
        print("ResNet101 has been loaded.")

    elif args.arch == 'resnext50': # 23,082,354 parameters
        model = models.resnext50_32x4d(pretrained=args.pretrained)
        num_infea = model.fc.in_features
        model.fc = nn.Linear(num_infea, args.num_classes)
        print("ResNext50 has been loaded.")

    elif args.arch == 'resnext101': # 86,844,786 parameters
        model = models.resnext101_32x8d(pretrained=args.pretrained)
        num_infea = model.fc.in_features
        model.fc = nn.Linear(num_infea, args.num_classes)
        print("ResNext101 has been loaded.")

    elif args.arch == 'densenet161': # 26,582,450 parameters
        model = models.densenet161(pretrained=args.pretrained)
        num_infea = model.classifier.in_features
        model.classifier = nn.Linear(num_infea, args.num_classes)
        print("Densenet161 has been loaded.")

    elif args.arch == 'densenet169': # 12,567,730 parameters
        model = models.densenet169(pretrained=args.pretrained)
        num_infea = model.classifier.in_features
        model.classifier = nn.Linear(num_infea, args.num_classes)
        print("Densenet166 has been loaded.")

    elif args.arch == 'densenet201': # 18,188,978 parameters
        model = models.densenet201(pretrained=args.pretrained)
        num_infea = model.classifier.in_features
        model.classifier = nn.Linear(num_infea, args.num_classes)
        print("Densenet201 has been loaded.")

    elif args.arch == 'vgg16_bn': # 134,473,842 parameters
        model = models.vgg16_bn(pretrained=args.pretrained)
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1], nn.Linear(4096, args.num_classes))
        print("VGG16_BN has been loaded.")

    elif args.arch == 'vgg19_bn': # 139,786,098 parameters
        model = models.vgg19_bn(pretrained=args.pretrained)
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1], nn.Linear(4096, args.num_classes))
        print("VGG19_BN has been loaded.")

    elif args.arch == 'squeezenet': # 748,146 parameters
        model = models.squeezenet1_1(pretrained=args.pretrained)
        model.classifier = nn.Sequential(list(model.classifier.children())[0], nn.Conv2d(512, args.num_classes, kernel_size=1), *list(model.classifier.children())[2:])
        print("SqueezeNet has been loaded.")

    elif args.arch == 'shufflenet': # 1,304,854 parameters
        model = models.shufflenet_v2_x1_0(pretrained=args.pretrained)
        num_infea = model.fc.in_features
        model.fc = nn.Linear(num_infea, args.num_classes)
        print("ShuffleNet has been loaded.")

    elif args.arch == 'mobilenet': # 2,287,922 parameters
        model = models.mobilenet_v2(pretrained=args.pretrained)
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1], nn.Linear(1280, args.num_classes))
        print("MobileNet has been loaded.")

    elif args.arch == 'AmirNet': # 1,875,666 parameters
        model = AmirNet(num_classes=args.num_classes)
        if args.input_size != 128:
            raise ValueError('AmirNet only accepts images of size 128 x 128!')
        print("AmirNet has been loaded.")

    elif args.arch == 'AmirNet_DO': # 1,875,666 parameters
        model = AmirNet_DO(num_classes=args.num_classes)
        if args.input_size != 128:
            raise ValueError('AmirNet only accepts images of size 128 x 128!')
        print("AmirNet_DO has been loaded with dropout.")

    elif args.arch == 'AmirNet_CDO': # 1,875,672 parameters
        model = AmirNet_CDO(num_classes=args.num_classes)
        if args.input_size != 128:
            raise ValueError('AmirNet only accepts images of size 128 x 128!')
        print("AmirNet_CDO has been loaded with concrete dropout.")

    elif args.arch == 'AmirNet_VDO': # 1,875,672 parameters
        model = AmirNet_VDO(num_classes=args.num_classes)
        if args.input_size != 128:
            raise ValueError('AmirNet only accepts images of size 128 x 128!')
        print("AmirNet_VDO has been loaded with variational dropout.")

    else:
        raise ValueError('Choose a real model, please!')

    return model
# --------------------------------------------------

# --------------------------------------------------
# Based on https://github.com/sungyubkim/MCDO/blob/master/Concrete_dropout_and_Variational_dropout.ipynb
class ConcreteDropout(nn.Module):
    """
    A class used to represent the Concrete dropout module

    ...

    Attributes
    ----------
    p_logit : float
        default value of -2.0
    temp : float
        default value of 0.01
    eps : float
        very small number with a default value of 1e-8

    Methods
    -------
    forward(x)
        The dropout rate is updated and applied
    p()
        The dropout probability is calculated
    """

    def __init__(self, p_logit=-2.0, temp=0.01, eps=1e-8):
        """
        Parameters
        ----------
        p_logit : float
            default value of -2.0
        temp : float
            default value of 0.01
        eps : float
            very small number with a default value of 1e-8
        """

        super(ConcreteDropout, self).__init__()
        self.p_logit = nn.Parameter(torch.Tensor([p_logit]))
        self.temp = temp
        self.eps = eps

    @property
    def p(self):
        """
        Returns
        -------
        float
            sigmoid of p_logit (probability)
        """
        return torch.sigmoid(self.p_logit)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch tensor
            features coming from the previous layer

        Returns
        -------
        torch tensor
            feature after this layer's dropout has been applied
        """
        if self.train():
            unif_noise = torch.rand_like(x)
            drop_prob = torch.log(self.p + self.eps) - torch.log(1-self.p + self.eps)+ torch.log(unif_noise + self.eps)- torch.log(1-unif_noise + self.eps)
            drop_prob = torch.sigmoid(drop_prob / self.temp)
            random_tensor = 1. - drop_prob
            retain_prob = 1. - self.p
            x *= random_tensor
            x /= retain_prob
        return x
# --------------------------------------------------

# --------------------------------------------------
# Based on https://github.com/sungyubkim/MCDO/blob/master/Concrete_dropout_and_Variational_dropout.ipynb
class VariationalDropout(nn.Module):
    """
    A class used to represent the Variational dropout module

    ...

    Attributes
    ----------
    log_alpha : float
        default value of -3.0
    max_log_alpha : float

    Methods
    -------
    forward(x)
        The dropout rate is updated and applied
    alpha()
        Calculates alpha value
    """

    def __init__(self, log_alpha=-3.):
        """
        Parameters
        ----------
        log_alpha : float
            default value of -3.0
        """
        super(VariationalDropout, self).__init__()
        self.max_log_alpha = 0.0
        self.log_alpha = nn.Parameter(torch.Tensor([log_alpha]))

    @property
    def alpha(self):
        """
        Returns
        -------
        float
            exponent of log_alpha
        """
        return torch.exp(self.log_alpha)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch tensor
            features coming from the previous layer

        Returns
        -------
        torch tensor
            feature after this layer's dropout has been applied
        """
        if self.train():
            normal_noise = torch.randn_like(x)
            self.log_alpha.data = torch.clamp(self.log_alpha.data, max=self.max_log_alpha)
            random_tensor = 1. + normal_noise * torch.sqrt(self.alpha)
            x *= random_tensor
        return x
# --------------------------------------------------

# --------------------------------------------------
class AmirNet(nn.Module):
    """
    A class used to represent a simple custom architecture for classification. This model contains 1,875,666 parameters.
    This architecture will only accept images of size 128 x 128, and is made up of six convolutional layers, three max-pooling operations and a linear layer.
    """
    def __init__(self, num_classes=50):
        super(AmirNet, self).__init__()

        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv_5 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv_6 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)

        self.fc_1 = nn.Linear(4*4*256, num_classes)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.relu(self.conv_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.relu(self.conv_4(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.relu(self.conv_5(x))
        x = self.relu(self.conv_6(x))
        x = x.view(-1, 4*4*256)
        x = self.fc_1(x)
        return x
# --------------------------------------------------

# --------------------------------------------------
class AmirNet_DO(nn.Module): # 1,875,666 parameters
    """
    A class used to represent a simple custom architecture for classification. The architecture contains a dropout module after each weight layer to enable Monte Carlo sampling for Bayesian approximation. This model contains 1,875,666 parameters.
    This architecture will only accept images of size 128 x 128, and is made up of six convolutional layers, three max-pooling operations and a linear layer.
    """
    def __init__(self, num_classes=50, dropout_level=0.05):
        super(AmirNet_DO, self).__init__()

        self.dropout_level = dropout_level

        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv_5 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv_6 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)

        self.fc_1 = nn.Linear(4*4*256, num_classes)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        x = self.relu(self.conv_1(x))
        x = F.dropout(x, p=self.dropout_level, training=True)
        x = self.relu(self.conv_2(x))
        x = F.dropout(x, p=self.dropout_level, training=True)
        x = F.max_pool2d(x, 2, 2)
        x = self.relu(self.conv_3(x))
        x = F.dropout(x, p=self.dropout_level, training=True)
        x = F.max_pool2d(x, 2, 2)
        x = self.relu(self.conv_4(x))
        x = F.dropout(x, p=self.dropout_level, training=True)
        x = F.max_pool2d(x, 2, 2)
        x = self.relu(self.conv_5(x))
        x = F.dropout(x, p=self.dropout_level, training=True)
        x = self.relu(self.conv_6(x))
        x = F.dropout(x, p=self.dropout_level, training=True)
        x = x.view(-1, 4*4*256)
        x = self.fc_1(x)
        return x
# --------------------------------------------------

# --------------------------------------------------
class AmirNet_CDO(nn.Module): # 1,875,672 parameters
    """
    A class used to represent a simple custom architecture for classification. The architecture contains the concrete dropout module (https://arxiv.org/abs/1705.07832) after each weight layer to enable Monte Carlo sampling for Bayesian approximation. This model contains 1,875,672 parameters.
    This architecture will only accept images of size 128 x 128, and is made up of six convolutional layers, three max-pooling operations and a linear layer.
    """
    def __init__(self, num_classes=50, weight_reg_coef=5e-4, dropout_reg_coef=1e-2):
        super(AmirNet_CDO, self).__init__()

        self.weight_reg_coef = weight_reg_coef
        self.dropout_reg_coef = dropout_reg_coef

        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv_5 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv_6 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)

        self.fc_1 = nn.Linear(4*4*256, num_classes)

        self.relu = nn.LeakyReLU(0.2)

        self.cdo_1 = ConcreteDropout()
        self.cdo_2 = ConcreteDropout()
        self.cdo_3 = ConcreteDropout()
        self.cdo_4 = ConcreteDropout()
        self.cdo_5 = ConcreteDropout()
        self.cdo_6 = ConcreteDropout()

    def forward(self, x):
        x = self.relu(self.conv_1(x))
        x = self.cdo_1(x)
        x = self.relu(self.conv_2(x))
        x = self.cdo_2(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.relu(self.conv_3(x))
        x = self.cdo_3(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.relu(self.conv_4(x))
        x = self.cdo_4(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.relu(self.conv_5(x))
        x = self.cdo_5(x)
        x = self.relu(self.conv_6(x))
        x = self.cdo_6(x)
        x = x.view(-1, 4*4*256)
        x = self.fc_1(x)
        return x

    def entropy(self, cdo):
        return -cdo.p * torch.log(cdo.p + 1e-8) - (1 - cdo.p) * torch.log(1 - cdo.p + 1e-8)

    def regularisation(self):
        weight_reg =  (self.fc_1.weight.norm()**2 + self.fc_1.bias.norm()**2) \
                    + (self.conv_1.weight.norm()**2 + self.conv_1.bias.norm()**2) / (1 - self.cdo_1.p) \
                    + (self.conv_2.weight.norm()**2 + self.conv_2.bias.norm()**2) / (1 - self.cdo_2.p) \
                    + (self.conv_3.weight.norm()**2 + self.conv_3.bias.norm()**2) / (1 - self.cdo_3.p) \
                    + (self.conv_4.weight.norm()**2 + self.conv_4.bias.norm()**2) / (1 - self.cdo_4.p) \
                    + (self.conv_5.weight.norm()**2 + self.conv_5.bias.norm()**2) / (1 - self.cdo_5.p) \
                    + (self.conv_6.weight.norm()**2 + self.conv_6.bias.norm()**2) / (1 - self.cdo_6.p)


        weight_reg *= self.weight_reg_coef

        # according to https://github.com/sungyubkim/MCDO/blob/master/Concrete_dropout_and_Variational_dropout.ipynb
        dropout_reg = self.entropy(self.cdo_1) + self.entropy(self.cdo_2) + self.entropy(self.cdo_3) + self.entropy(self.cdo_4) + self.entropy(self.cdo_5) + self.entropy(self.cdo_6)

        dropout_reg *= self.dropout_reg_coef

        return weight_reg + dropout_reg
# --------------------------------------------------

# --------------------------------------------------
class AmirNet_VDO(nn.Module):
    """
    A class used to represent a simple custom architecture for classification. The architecture contains the variational dropout module (https://arxiv.org/abs/1506.02557) after each weight layer to enable Monte Carlo sampling for Bayesian approximation. This model contains 1,875,672 parameters.
    This architecture will only accept images of size 128 x 128, and is made up of six convolutional layers, three max-pooling operations and a linear layer.
    """
    def __init__(self, num_classes=50, reg_coef=5e-4):
        super(AmirNet_VDO, self).__init__()

        self.reg_coef = reg_coef

        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv_5 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv_6 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)

        self.fc_1 = nn.Linear(4*4*256, num_classes)

        self.relu = nn.LeakyReLU(0.2)

        self.vdo_1 = VariationalDropout()
        self.vdo_2 = VariationalDropout()
        self.vdo_3 = VariationalDropout()
        self.vdo_4 = VariationalDropout()
        self.vdo_5 = VariationalDropout()
        self.vdo_6 = VariationalDropout()

    def forward(self, x):
        x = self.relu(self.conv_1(x))
        x = self.vdo_1(x)
        x = self.relu(self.conv_2(x))
        x = self.vdo_2(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.relu(self.conv_3(x))
        x = self.vdo_3(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.relu(self.conv_4(x))
        x = self.vdo_4(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.relu(self.conv_5(x))
        x = self.vdo_5(x)
        x = self.relu(self.conv_6(x))
        x = self.vdo_6(x)
        x = x.view(-1, 4*4*256)
        x = self.fc_1(x)
        return x

    def kl(self, vdo):

        c1 = 1.16145124
        c2 = -1.50204118
        c3 = 0.58629921

        return -0.5 * vdo.log_alpha - c1 * vdo.alpha - c2 * (vdo.alpha ** 2) - c3 * (vdo.alpha ** 3)

    def regularisation(self):
        return self.reg_coef * (self.kl(self.vdo_1) + self.kl(self.vdo_2) + self.kl(self.vdo_3) +  self.kl(self.vdo_4) + self.kl(self.vdo_5) + self.kl(self.vdo_6))
# --------------------------------------------------
