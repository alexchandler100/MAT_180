import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
import wandb
from torch.utils.data import Dataset, DataLoader

_default_cifar10_train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

_default_cifar10_eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])


def get_trainloader_subset(train_data_loader, samples_per_class):
    dataset_train = train_data_loader.dataset
    dataset_train_small = []
    counts_per_class = [0 for _ in range(10)]
    for i in range(len(dataset_train)):
        if counts_per_class[dataset_train[i][1]] < samples_per_class:
            dataset_train_small.append(dataset_train[i])
            counts_per_class[dataset_train[i][1]] += 1

    class AdaptDataset(Dataset):
        """TensorDataset with support of transforms.
        """

        def __init__(self, dataset_list, transform=None, target_transform=None):
            self.dataset_list = dataset_list
            self.transform = transform
            self.target_transform = target_transform

        def __getitem__(self, index):
            x = self.dataset_list[index][0]

            if self.transform:
                x = self.transform(x)

            y = self.dataset_list[index][1]

            if self.target_transform is not None:
                y = self.target_transform(y)

            return x, y

        def __len__(self):
            return len(self.dataset_list)

    dataset_train_small = AdaptDataset(dataset_train_small)
    train_loader = DataLoader(dataset_train_small, batch_size=100,
                              shuffle=True, num_workers=0)
    return train_loader


def get_dataloader(args, task="task1", samples_per_class="full"):
    binary_transform = lambda x: 0 if x < 5 else 1
    if args.dataset_name == "CIFAR10":
        if task == "task1":
            target_transform = None
        elif task == "task2":
            target_transform = binary_transform
        else:
            raise ValueError("Invalid task name: {}".format(task))
    elif args.dataset_name == "CIFAR10-Binary":
        if task == "task1":
            target_transform = binary_transform
        elif task == "task2":
            target_transform = None
        else:
            raise ValueError("Invalid task name: {}".format(task))
    else:
        raise ValueError("Invalid dataset name: {}".format(args.dataset_name))
    train_set = CIFAR10(args.root_dir, train=True, download=True,
                        transform=_default_cifar10_train_transform,
                        target_transform=target_transform)
    test_set = CIFAR10(args.root_dir, train=False, download=True,
                       transform=_default_cifar10_eval_transform,
                       target_transform=target_transform)

    train_data_loader = torch.utils.data.DataLoader(train_set,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.num_workers)
    test_data_loader = torch.utils.data.DataLoader(test_set,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.num_workers)
    if samples_per_class != "full":
        train_data_loader = get_trainloader_subset(train_data_loader, samples_per_class)

    return train_data_loader, test_data_loader


def get_class_nums(args, task="task1"):
    if args.dataset_name == "CIFAR10":
        if task == "task1":
            num_classes = 10
        elif task == "task2":
            num_classes = 2
        else:
            raise ValueError("Invalid task name: {}".format(task))
    elif args.dataset_name == "CIFAR10-Binary":
        if task == "task1":
            num_classes = 2
        elif task == "task2":
            num_classes = 10
        else:
            raise ValueError("Invalid task name: {}".format(task))
    else:
        raise ValueError("Invalid dataset name: {}".format(args.dataset_name))
    return num_classes


def evaluate_test_accuracy(model, dataloader, train_step, epoch, args, prefix=""):
    if prefix != "":
        prefix = prefix + "/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc1 = 100 * float(correct / total)
        log_dict = {prefix + "test_accuracy": acc1,
                    "epochs": epoch,
                    "train_step": train_step}
        wandb.log(log_dict)
