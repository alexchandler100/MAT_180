import os

import torch
import torch.nn as nn
from avalanche.training.strategies import Naive, AGEM, EWC
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from avalanche.benchmarks.datasets.tiny_imagenet import TinyImagenet
from avalanche.benchmarks.classic.ccifar100 import _default_cifar100_train_transform
from avalanche.benchmarks.classic.ccifar10 import _default_cifar10_train_transform
from torchvision.datasets import CIFAR100, CIFAR10
import einops as eo

_default_train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])


def load_TinyImagenet_loader(args, imagenet_root=None):
    if imagenet_root is None:
        train_set = TinyImagenet(root=args.root_dir, train=True, transform=_default_train_transform)
    else:
        train_set = TinyImagenet(root=imagenet_root, train=True, transform=_default_train_transform)

    train_data_loader = torch.utils.data.DataLoader(train_set,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.num_workers,
                                                    pin_memory=True)

    return train_data_loader


def load_TinyImagenet_loader_for_robustness(args):
    tiny_imagenet_train = os.path.join(args.root_dir, "Tiny-ImageNet", "train")
    tiny_imagenet_test = os.path.join(args.root_dir, "Tiny-ImageNet", "test")
    # root_dir = "/Users/ftraeuble/Downloads"
    # tiny_imagenet_train = os.path.join(root_dir, "IMagenet", "tiny-imagenet-200", "train")
    # tiny_imagenet_test = os.path.join(root_dir, "imagenet_val_bbox_crop")

    data_set_train = torchvision.datasets.ImageFolder(root=tiny_imagenet_train,
                                                      transform=_default_train_transform)
    data_set_test = torchvision.datasets.ImageFolder(root=tiny_imagenet_test,
                                                     transform=_default_train_transform)
    data_loader_train = torch.utils.data.DataLoader(data_set_train,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.num_workers,
                                                    pin_memory=True)
    data_loader_test = torch.utils.data.DataLoader(data_set_test,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True)
    return data_loader_train, data_loader_test


def load_TinyImagenetC_loader(args, num=0):
    tiny_imagenet_c = os.path.join(args.root_dir, "Tiny-ImageNet-C")
    # get all subdirectory names of Tiny-ImageNet-C
    subdirs = [name for name in os.listdir(tiny_imagenet_c) if os.path.isdir(os.path.join(tiny_imagenet_c, name))]
    severity = str((num % 5) + 1)
    subdir = subdirs[int(num // 5)]
    test_case = subdir + "_" + severity
    root = os.path.join(tiny_imagenet_c, subdir, severity)
    data_set = torchvision.datasets.ImageFolder(root=root,
                                                transform=_default_train_transform)
    data_loader = torch.utils.data.DataLoader(data_set,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True)
    return data_loader, test_case


def load_cl_data_loaders(args):
    if args.dataset_name == "CIFAR100":
        train_set = CIFAR100(args.root_dir, train=True, download=True, transform=_default_cifar100_train_transform)
        test_set = CIFAR100(args.root_dir, train=False, download=True, transform=_default_cifar100_train_transform)
    elif args.dataset_name == "CIFAR10":
        train_set = CIFAR10(args.root_dir, train=True, download=True, transform=_default_cifar10_train_transform)
        test_set = CIFAR10(args.root_dir, train=False, download=True, transform=_default_cifar10_train_transform)
    elif args.dataset_name == "TinyImagenet":
        train_set = TinyImagenet(root=args.root_dir, train=True, transform=_default_train_transform)
        test_set = TinyImagenet(root=args.root_dir, train=False, transform=_default_train_transform)
    else:
        raise ValueError("Dataset not supported")

    train_data_loader = torch.utils.data.DataLoader(train_set,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.num_workers,
                                                    pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(test_set,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True)

    return train_data_loader, test_data_loader


def get_class_nums(args):
    if args.dataset_name == "CIFAR10":
        class_nums = 10
    elif args.dataset_name == "CIFAR100":
        class_nums = 100
    elif args.dataset_name == "TinyImagenet":
        class_nums = 200
    else:
        raise NotImplementedError("Dataset {} not supported".format(args.dataset_name))
    return class_nums


def get_few_shot_data_loader(dataloader_train, args):
    few_shot_data = dataloader_train.dataset
    dataset_val_small = []
    shuffle = True

    class_nums = get_class_nums(args)

    counts_per_class = [0 for _ in range(class_nums)]
    samples_per_class = int(0.01 * len(few_shot_data) / class_nums)

    for i in range(len(few_shot_data)):
        if counts_per_class[few_shot_data[i][1]] < samples_per_class:
            dataset_val_small.append(few_shot_data[i])
            counts_per_class[few_shot_data[i][1]] += 1

    class FewShotDataset(Dataset):
        """TensorDataset with support of transforms.
        """

        def __init__(self, dataset_list, transform=None):
            self.dataset_list = dataset_list
            self.transform = transform

        def __getitem__(self, index):
            x = self.dataset_list[index][0]

            if self.transform:
                x = self.transform(x)

            y = self.dataset_list[index][1]

            return x, y

        def __len__(self):
            return len(self.dataset_list)

    dataset_val_small = FewShotDataset(dataset_val_small, transform=None)
    few_shot_data_loader = DataLoader(dataset_val_small,
                                      batch_size=args.batch_size,
                                      shuffle=shuffle, num_workers=0)
    return few_shot_data_loader


def compute_key_overlap(model, dataloader_train, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    key_counts = torch.zeros((get_class_nums(args), args.num_pairs, args.num_books))
    for i, (inputs, labels) in enumerate(dataloader_train):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            embed_idx = model[:3](inputs)[1]
            # embed idx is of size (batch_size, h, w, book)
            for j in range(embed_idx.shape[0]):
                for book in range(args.num_books):
                    key_book_matrix = eo.rearrange(embed_idx[j, :, :, book], "... -> (...)")
                    key_book_counts = torch.bincount(key_book_matrix, minlength=args.num_pairs)
                    key_counts[labels[j], :, book] += key_book_counts
    key_counts = key_counts.cpu().numpy()
    return key_counts


def get_cl_strategy(model, args, eval_plugin, device):
    if args.cl_strat == "AGEM":
        cl_strategy = AGEM(
            model,
            torch.optim.SGD(model.parameters(), lr=args.learning_rate),
            nn.CrossEntropyLoss(),
            patterns_per_exp=args.agem_patterns_per_exp,
            sample_size=args.agem_sample_size,
            train_mb_size=args.batch_size,
            train_epochs=args.cl_epochs,
            eval_mb_size=args.batch_size,
            evaluator=eval_plugin,
            device=device,
        )
    elif args.cl_strat == "EWC":
        cl_strategy = EWC(
            model,
            torch.optim.SGD(model.parameters(), lr=args.learning_rate),
            nn.CrossEntropyLoss(),
            ewc_lambda=args.ewc_lambda,
            train_mb_size=args.batch_size,
            train_epochs=args.cl_epochs,
            eval_mb_size=args.batch_size,
            evaluator=eval_plugin,
            device=device,
        )
    elif args.cl_strat == "Naive":
        cl_strategy = Naive(
            model,
            torch.optim.SGD(model.parameters(), lr=args.learning_rate),
            nn.CrossEntropyLoss(),
            train_mb_size=args.batch_size,
            train_epochs=args.cl_epochs,
            eval_mb_size=args.batch_size,
            evaluator=eval_plugin,
            device=device,
        )
    else:
        raise NotImplementedError()

    return cl_strategy
