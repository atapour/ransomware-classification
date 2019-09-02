import torch.utils.data
import torchvision.datasets as datasets
import numpy as np
from torchvision.transforms import (CenterCrop, ColorJitter, Compose,
                                    Grayscale, Normalize, RandomCrop,
                                    RandomGrayscale, RandomHorizontalFlip,
                                    RandomRotation, RandomVerticalFlip, Resize,
                                    ToTensor)

from data.augs import (AddBlur, AddDefocusBlur, AddMotionBlur, AddNoise,
                       AddOcclusion, ChangeBrightness, ChangeContrast,
                       RandomPerspective)


def get_transforms(args):
    """
    Compose and return the transforms needed for the augmentation process

    Parameters
    ----------
    args: arguments class

    Returns
    ----------
    transforms: the composed list of transforms

    """

    input_size = args.input_size

    if args.phase == 'test':

        if args.pretrained:
            transforms = Compose([Resize(input_size), CenterCrop(input_size), ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            transforms = Compose([Resize(input_size), CenterCrop(input_size), ToTensor()])

    elif args.phase == 'train':

        transform_list = []
        transform_list.append(Resize(input_size))
        transform_list.append(RandomCrop(input_size))

        if args.augs:
            if 'rotate' in args.augs:
                transform_list.append(RandomRotation(100, expand=True))
            if 'hflip' in args.augs:
                transform_list.append(RandomHorizontalFlip())
            if 'vflip' in args.augs:
                transform_list.append(RandomVerticalFlip())
            if 'contrast' in args.augs:
                transform_list.append(ChangeContrast())
            if 'brightness' in args.augs:
                transform_list.append(ChangeBrightness())
            if 'occlusion' in args.augs:
                transform_list.append(AddOcclusion())
            if 'regularblur' in args.augs:
                transform_list.append(AddBlur())
            if 'motionblur' in args.augs:
                transform_list.append(AddMotionBlur())
            if 'defocusblur' in args.augs:
                transform_list.append(AddDefocusBlur())
            if 'perspective' in args.augs:
                transform_list.append(RandomPerspective())
            if 'colorjitter' in args.augs:
                transform_list.append(ColorJitter(brightness=0, contrast=0, saturation=2, hue=0.05))
            if 'gray' in args.augs:
                transform_list.append(RandomGrayscale(p=0.1))

        transform_list.append(Resize(input_size))
        transform_list.append(ToTensor())

        if args.augs and 'noise' in args.augs:
            transform_list.append(AddNoise())

        if args.pretrained:
            transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

        transforms = Compose(transform_list)
    return transforms

def create_dataset(args):
    """
    Create and return the dataset class

    Parameters
    ----------
    args: arguments class

    Returns
    ----------
    dataset: the dataset
    weights: class weights needed for class balancing

    """

    transforms = get_transforms(args)

    if args.phase == 'train':
        path = args.data_root

    elif args.phase == 'test':
        path = args.pos_root

    dataset = datasets.ImageFolder(path)
    dataset.transform = transforms

    # Compute the class weights for balancing
    counts, _ = np.histogram(dataset.targets, bins=np.arange(len(dataset.classes) + 1))
    weights = np.array(1. / counts)

    return dataset, weights

def create_loader(args):
    """
    Create and return the dataloader

    Parameters
    ----------
    args: arguments class

    Returns
    ----------
    dataloader: the dataloader
    weights: class weights needed for class balancing

    """

    data_loader = DataLoader()
    data_loader.initialize(args)

    print("Training data has been loaded.")

    return data_loader, data_loader.return_weights()

class DataLoader():
    """
    A class used to create the dataloader 
    """

    def initialize(self, args):

        self.dataset, self.weights = create_dataset(args)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=args.batch_size, shuffle=args.phase == 'train', num_workers=int(args.num_workers), drop_last=False)

    def return_weights(self):
        return self.weights

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for _, data in enumerate(self.dataloader):
            yield data
