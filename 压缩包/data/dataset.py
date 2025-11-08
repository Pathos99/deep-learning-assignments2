import torchvision
import torch
import torchvision.transforms as transforms

'''
load dataset
'''
class DataSet():
    def __init__(self, train_loader, test_loader, classes):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.classes = classes

'''
dataset preprocessing
'''
class DataBuilder():
    '''
    Build training dataset or test dataset
    '''
    def __init__(self, args) -> None:
        self.args = args
        self.stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    ## Data transforms (normalization & data augmentation)
    def train_transform(self):
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*self.stats,inplace=True) # x = (x - mean) / std
        ])
        return transform
    
    def test_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Normalize(*self.stats)
        ])
        return transform

    
    # Build Training Dataset
    def train_builder(self):
        train_set = torchvision.datasets.CIFAR10(
            root=self.args.data_path,
            train=True,
            download=self.args.is_download,
            transform=self.train_transform()
        )

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers
        )
        return train_loader
    
    # Build Test Dataset
    def test_builder(self):
        test_set = torchvision.datasets.CIFAR10(
            root=self.args.data_path,
            train=False,
            download=self.args.is_download,
            transform=self.test_transform()
        )

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers
        )
        return test_loader