import os
import torch
import wandb
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset
from avalanche.benchmarks.datasets.tiny_imagenet import TinyImagenet


def get_adapt_modes(args):
    if args.dataset_name == "CIFAR10":
        num_classes = 10
        adapt_modes = [
            "ValSet100-P100",
            "ValSet100-P10",
            "ValSet100-P20",
            "ValSet1000-P100",
            "ValSet1000-P10",
            "ValSet1000-P20",
            "ValSet",
            "TrainSetImbalanced",
            "TrainSet",
            "ValSet10B",
            "ValSet50B",
            "ValSet100B",
            "ValSet500B",
            "ValSet1000B",
            "ValSet-P1",
        ]
    elif args.dataset_name == "CIFAR100":
        num_classes = 100
        adapt_modes = [
            "ValSet100-P100",
            "ValSet100-P10",
            "ValSet100-P20",
            "ValSet",
            "TrainSetImbalanced",
            # "TrainSet",
            "ValSet1B",
            "ValSet5B",
            "ValSet10B",
            "ValSet20B",
            "ValSet50B",
            "ValSet70B",
            #"ValSet-P1",
        ]
    elif args.dataset_name == "TinyImagenet":
        num_classes = 200
        adapt_modes = [
            "ValSet100-P100",
            "ValSet100-P10",
            "ValSet100-P20",
            "ValSet",
            "TrainSetImbalanced",
            "TrainSet",
            "ValSet10B",
            "ValSet50B",
            "ValSet-P1",
        ]
    else:
        raise NotImplementedError("Dataset {} not supported".format(args.dataset_name))
    uni_modal_adapt_modes = [
        "ValSet100-C{i}".format(i=i) for i in range(0, num_classes)
    ]
    adapt_modes = adapt_modes + uni_modal_adapt_modes

    return adapt_modes


def evaluate_train_accuracy(
    model, dataloader, train_step, epoch, args, prefix="", additional_loader=None
):
    if prefix != "":
        prefix = prefix + "/"
    if args.dataset_name == "CIFAR10":
        num_classes = 10
    elif args.dataset_name == "CIFAR100":
        num_classes = 100
    elif args.dataset_name == "TinyImagenet":
        num_classes = 200
    else:
        raise NotImplementedError("Dataset {} not supported".format(args.dataset_name))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss().to(device)
    correct_pred = [0] * num_classes
    total_pred = [0] * num_classes
    with torch.no_grad():
        correct = 0
        total = 0
        total_loss = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += float(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if not args.no_per_class_acc:
                for label, prediction in zip(labels, predicted):
                    if label == prediction:
                        correct_pred[label] += 1
                    total_pred[label] += 1
        if additional_loader is not None:
            for inputs, labels in additional_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += float(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if not args.no_per_class_acc:
                    for label, prediction in zip(labels, predicted):
                        if label == prediction:
                            correct_pred[label] += 1
                        total_pred[label] += 1
        acc1 = 100 * float(correct / total)
        total_loss = float(total_loss / total)
        log_dict = {
            prefix + "train_accuracy": acc1,
            prefix + "train_loss": total_loss,
            "epochs": epoch,
            "train_step": train_step,
        }
        if not args.no_per_class_acc:
            for class_label, correct_count in enumerate(correct_pred):
                accuracy = 100 * float(correct_count) / total_pred[class_label]
                log_dict[prefix + "train_accuracy_" + str(class_label)] = accuracy
        wandb.log(log_dict)


def evaluate_test_accuracy(
    model, dataloader, train_step, epoch, args, prefix="", reference_accuracy=None
):
    if prefix != "":
        prefix = prefix + "/"
    if args.dataset_name == "CIFAR10":
        num_classes = 10
    elif args.dataset_name == "CIFAR100":
        num_classes = 100
    elif args.dataset_name == "TinyImagenet":
        num_classes = 200
    else:
        raise NotImplementedError("Dataset {} not supported".format(args.dataset_name))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct_pred = [0] * num_classes
    total_pred = [0] * num_classes
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
            if not args.no_per_class_acc:
                for label, prediction in zip(labels, predicted):
                    if label == prediction:
                        correct_pred[label] += 1
                    total_pred[label] += 1
        acc1 = 100 * float(correct / total)
        log_dict = {
            prefix + "test_accuracy": acc1,
            "epochs": epoch,
            "train_step": train_step,
        }
        if reference_accuracy is not None:
            log_dict[prefix + "adapt_accuracy_gain"] = acc1 - reference_accuracy
        if not args.no_per_class_acc:
            for class_label, correct_count in enumerate(correct_pred):
                accuracy = 100 * float(correct_count) / total_pred[class_label]
                log_dict[prefix + "test_accuracy_" + str(class_label)] = accuracy
        wandb.log(log_dict)

    if args.eval_with_gaussian_noise != 0.0:
        correct_pred = [0] * num_classes
        total_pred = [0] * num_classes
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                inputs = (
                    inputs + torch.randn_like(inputs) * args.eval_with_gaussian_noise
                )
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if not args.no_per_class_acc:
                    for label, prediction in zip(labels, predicted):
                        if label == prediction:
                            correct_pred[label] += 1
                        total_pred[label] += 1
            acc1 = 100 * float(correct / total)
            log_dict = {
                prefix + "test_accuracy_on_noisy": acc1,
                "epochs": epoch,
                "train_step": train_step,
            }
            if not args.no_per_class_acc:
                for class_label, correct_count in enumerate(correct_pred):
                    accuracy = 100 * float(correct_count) / total_pred[class_label]
                    log_dict[
                        prefix + "test_accuracy_on_noisy" + str(class_label)
                    ] = accuracy
            wandb.log(log_dict)


def evaluate_test_accuracy_legacy(
    model, dataloader, train_step, epoch, args, prefix="", reference_accuracy=None
):
    if prefix != "":
        prefix = prefix + "/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
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
            if args.dataset_name == "CIFAR10":
                for label, prediction in zip(labels, predicted):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1
        acc1 = 100 * float(correct / total)
        log_dict = {
            prefix + "test_accuracy": acc1,
            "epochs": epoch,
            "train_step": train_step,
        }
        if reference_accuracy is not None:
            log_dict[prefix + "adapt_accuracy_gain"] = acc1 - reference_accuracy
        if args.dataset_name == "CIFAR10":
            for classname, correct_count in correct_pred.items():
                accuracy = 100 * float(correct_count) / total_pred[classname]
                log_dict[prefix + "test_accuracy_" + classname] = accuracy
        wandb.log(log_dict)
        return acc1


def load_imbalanced_data_loaders(root_dir, args, dataset_name=None):
    dsets, output_tfm = load_cifar_datasets(
        root_dir=root_dir,
        dataset_name=args.dataset_name if dataset_name is None else dataset_name,
        imb_type="exp",
        imb_factor=args.imb_factor,
        trainer_type="baseline",
        flip=False,
        args=args,
    )
    dataset_cifar_train = dsets["train"]
    dataset_cifar_val = dsets["val"]
    dataset_cifar_test = dsets["test"]
    dataloader_train = DataLoader(
        dataset_cifar_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dataloader_test = DataLoader(
        dataset_cifar_test,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dataloader_val = DataLoader(
        dataset_cifar_val,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return dataloader_train, dataloader_val, dataloader_test


def get_adapt_batch(adapt_mode, dataloader_val, args, dataloader_train=None):
    dataset_val = dataloader_val.dataset.dataset
    dataset_val_small = []
    shuffle = True
    adapt_steps = args.adapt_steps
    batch_size = 100
    class_nums = 10 if args.dataset_name == "CIFAR10" else 100
    if adapt_mode == "ValSet1B":
        """construct a dataloader that contains 10 samples per class from dataset_val"""
        adapt_steps = int(class_nums * 100 * args.adapt_steps / 500) # TODO: Changed
        counts_per_class = [0 for _ in range(class_nums)]
        for i in range(len(dataset_val)):
            if counts_per_class[dataset_val[i][1]] < 1:
                dataset_val_small.append(dataset_val[i])
                counts_per_class[dataset_val[i][1]] += 1
    elif adapt_mode == "ValSet5B":
        """construct a dataloader that contains 10 samples per class from dataset_val"""
        counts_per_class = [0 for _ in range(class_nums)]
        adapt_steps = int(class_nums * 100 * args.adapt_steps / 500)  # TODO: Changed
        for i in range(len(dataset_val)):
            if counts_per_class[dataset_val[i][1]] < 5:
                dataset_val_small.append(dataset_val[i])
                counts_per_class[dataset_val[i][1]] += 1
    elif adapt_mode == "ValSet10B":
        """construct a dataloader that contains 10 samples per class from dataset_val"""
        counts_per_class = [0 for _ in range(class_nums)]
        adapt_steps = int(class_nums * 100 * args.adapt_steps / 500)  # TODO: Changed
        for i in range(len(dataset_val)):
            if counts_per_class[dataset_val[i][1]] < 10:
                dataset_val_small.append(dataset_val[i])
                counts_per_class[dataset_val[i][1]] += 1
    elif adapt_mode == "ValSet20B":
        """construct a dataloader that contains 10 samples per class from dataset_val"""
        counts_per_class = [0 for _ in range(class_nums)]
        adapt_steps = int(class_nums * 100 * args.adapt_steps / 500)  # TODO: Changed
        for i in range(len(dataset_val)):
            if counts_per_class[dataset_val[i][1]] < 20:
                dataset_val_small.append(dataset_val[i])
                counts_per_class[dataset_val[i][1]] += 1
    elif adapt_mode == "ValSet50B":
        """construct a dataloader that contains 50 samples per class from dataset_val"""
        counts_per_class = [0 for _ in range(class_nums)]
        adapt_steps = int(class_nums * 100 * args.adapt_steps / 500)  # TODO: Changed
        for i in range(len(dataset_val)):
            if counts_per_class[dataset_val[i][1]] < 50:
                dataset_val_small.append(dataset_val[i])
                counts_per_class[dataset_val[i][1]] += 1
    elif adapt_mode == "ValSet70B":
        """construct a dataloader that contains 70 samples per class from dataset_val"""
        counts_per_class = [0 for _ in range(class_nums)]
        adapt_steps = int(class_nums * 100 * args.adapt_steps / 500)  # TODO: Changed
        for i in range(len(dataset_val)):
            if counts_per_class[dataset_val[i][1]] < 70:
                dataset_val_small.append(dataset_val[i])
                counts_per_class[dataset_val[i][1]] += 1
    elif adapt_mode == "ValSet100B":
        """construct a dataloader that contains 100 samples per class from dataset_val"""
        counts_per_class = [0 for _ in range(class_nums)]
        adapt_steps = int(class_nums * 100 * args.adapt_steps / 500)  # TODO: Changed
        for i in range(len(dataset_val)):
            if counts_per_class[dataset_val[i][1]] < 100:
                dataset_val_small.append(dataset_val[i])
                counts_per_class[dataset_val[i][1]] += 1
    elif adapt_mode == "ValSet500B":
        """construct a dataloader that contains 50 samples per class from dataset_val"""
        counts_per_class = [0 for _ in range(class_nums)]
        adapt_steps = int(class_nums * 100 * args.adapt_steps / 500)  # TODO: Changed
        for i in range(len(dataset_val)):
            if counts_per_class[dataset_val[i][1]] < 500:
                dataset_val_small.append(dataset_val[i])
                counts_per_class[dataset_val[i][1]] += 1
    elif adapt_mode == "ValSet1000B":
        """construct a dataloader that contains 100 samples per class from dataset_val"""
        counts_per_class = [0 for _ in range(class_nums)]
        adapt_steps = int(class_nums * 100 * args.adapt_steps / 500)  # TODO: Changed
        for i in range(len(dataset_val)):
            if counts_per_class[dataset_val[i][1]] < 1000:
                dataset_val_small.append(dataset_val[i])
                counts_per_class[dataset_val[i][1]] += 1
    elif adapt_mode == "ValSet100-P100":
        """construct a dataloader that contains 50 samples per class from dataset_val"""
        adapt_steps = int(class_nums * 100 * args.adapt_steps / 500)
        counts_per_class = [0 for _ in range(class_nums)]
        dataset_val_small_sub = [[] for _ in range(class_nums)]
        for i in range(len(dataset_val)):
            if counts_per_class[dataset_val[i][1]] < 100:
                dataset_val_small_sub[dataset_val[i][1]].append(dataset_val[i])
                counts_per_class[dataset_val[i][1]] += 1
        for j in range(class_nums):
            dataset_val_small.extend(dataset_val_small_sub[j] * 100)
        shuffle = False
    elif adapt_mode == "ValSet1000-P100":
        """construct a dataloader that contains 50 samples per class from dataset_val"""
        adapt_steps = int(class_nums * 100 * args.adapt_steps / 500)
        counts_per_class = [0 for _ in range(class_nums)]
        dataset_val_small_sub = [[] for _ in range(class_nums)]
        for i in range(len(dataset_val)):
            if counts_per_class[dataset_val[i][1]] < 1000:
                dataset_val_small_sub[dataset_val[i][1]].append(dataset_val[i])
                counts_per_class[dataset_val[i][1]] += 1
        for j in range(class_nums):
            dataset_val_small.extend(dataset_val_small_sub[j] * 100)
        shuffle = False
    elif adapt_mode == "ValSet100-P20":
        """construct a dataloader that contains 50 samples per class from dataset_val"""
        adapt_steps = int(class_nums * 100 * args.adapt_steps / 500)
        counts_per_class = [0 for _ in range(class_nums)]
        dataset_val_small_sub = [[] for _ in range(class_nums)]
        for i in range(len(dataset_val)):
            if counts_per_class[dataset_val[i][1]] < 100:
                dataset_val_small_sub[dataset_val[i][1]].append(dataset_val[i])
                counts_per_class[dataset_val[i][1]] += 1
        for j in range(class_nums):
            dataset_val_small.extend(dataset_val_small_sub[j] * 20)
        shuffle = False
    elif adapt_mode == "ValSet1000-P20":
        """construct a dataloader that contains 50 samples per class from dataset_val"""
        adapt_steps = int(class_nums * 100 * args.adapt_steps / 500)
        counts_per_class = [0 for _ in range(class_nums)]
        dataset_val_small_sub = [[] for _ in range(class_nums)]
        for i in range(len(dataset_val)):
            if counts_per_class[dataset_val[i][1]] < 1000:
                dataset_val_small_sub[dataset_val[i][1]].append(dataset_val[i])
                counts_per_class[dataset_val[i][1]] += 1
        for j in range(class_nums):
            dataset_val_small.extend(dataset_val_small_sub[j] * 20)
        shuffle = False
    elif adapt_mode == "ValSet100-P10":
        """construct a dataloader that contains 50 samples per class from dataset_val"""
        adapt_steps = int(class_nums * 100 * args.adapt_steps / 500)
        counts_per_class = [0 for _ in range(class_nums)]
        dataset_val_small_sub = [[] for _ in range(class_nums)]
        for i in range(len(dataset_val)):
            if counts_per_class[dataset_val[i][1]] < 100:
                dataset_val_small_sub[dataset_val[i][1]].append(dataset_val[i])
                counts_per_class[dataset_val[i][1]] += 1
        for j in range(class_nums):
            dataset_val_small.extend(dataset_val_small_sub[j] * 10)
        shuffle = False
    elif adapt_mode == "ValSet1000-P10":
        """construct a dataloader that contains 50 samples per class from dataset_val"""
        adapt_steps = int(class_nums * 100 * args.adapt_steps / 500)
        counts_per_class = [0 for _ in range(class_nums)]
        dataset_val_small_sub = [[] for _ in range(class_nums)]
        for i in range(len(dataset_val)):
            if counts_per_class[dataset_val[i][1]] < 1000:
                dataset_val_small_sub[dataset_val[i][1]].append(dataset_val[i])
                counts_per_class[dataset_val[i][1]] += 1
        for j in range(class_nums):
            dataset_val_small.extend(dataset_val_small_sub[j] * 10)
        shuffle = False
    elif adapt_mode == "ValSet100-P1":
        """construct a dataloader that contains 50 samples per class from dataset_val"""
        adapt_steps = int(class_nums * 100 * args.adapt_steps / 500)
        counts_per_class = [0 for _ in range(class_nums)]
        dataset_val_small_sub = [[] for _ in range(class_nums)]
        for i in range(len(dataset_val)):
            if counts_per_class[dataset_val[i][1]] < 100:
                dataset_val_small_sub[dataset_val[i][1]].append(dataset_val[i])
                counts_per_class[dataset_val[i][1]] += 1
        for j in range(class_nums):
            dataset_val_small.extend(dataset_val_small_sub[j] * 1)
        shuffle = False
    elif adapt_mode == "ValSet":
        adapt_steps = int(class_nums * 100 * args.adapt_steps / 500)  # TODO: Changed
        adapt_loader = DataLoader(
            dataloader_val.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        adapt_dict = dict(adapt_loader=adapt_loader, adapt_steps=adapt_steps)
        return adapt_dict
    elif adapt_mode == "ValSet-P1":
        adapt_steps = int(len(dataset_val) / batch_size)
        adapt_loader = DataLoader(
            dataloader_val.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        adapt_dict = dict(adapt_loader=adapt_loader, adapt_steps=adapt_steps)
        return adapt_dict
    elif adapt_mode == "TrainSet":
        adapt_loader = DataLoader(
            dataloader_train.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        adapt_dict = dict(adapt_loader=adapt_loader, adapt_steps=adapt_steps)
        return adapt_dict
    elif adapt_mode == "TrainSetImbalanced":
        dsets, output_tfm = load_cifar_datasets(
            root_dir=args.root_dir,
            dataset_name=args.dataset_name,
            imb_type="exp",
            imb_factor=0.01,
            trainer_type="baseline",
            flip=False,
        )
        dataset_1percent_train = dsets["train"]
        adapt_steps = (
            int(len(dataset_1percent_train) / batch_size) * 5
        )  # * int(args.adapt_steps / 2000)
        adapt_loader = DataLoader(
            dataset_1percent_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        adapt_dict = dict(adapt_loader=adapt_loader, adapt_steps=adapt_steps)
        return adapt_dict
    elif "ValSet100-C" in adapt_mode:
        adapt_steps = int(adapt_steps / 2)
        class_label = int(adapt_mode[-1])
        for i in range(len(dataset_val)):
            if dataset_val[i][1] == class_label:
                dataset_val_small.append(dataset_val[i])
            if len(dataset_val_small) == 100:
                break
    else:
        adapt_loader = DataLoader(
            dataloader_val.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        adapt_dict = dict(adapt_loader=adapt_loader, adapt_steps=adapt_steps)
        return adapt_dict

    class AdaptDataset(Dataset):
        """TensorDataset with support of transforms."""

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

    dataset_val_small = AdaptDataset(dataset_val_small, transform=None)
    adapt_loader = DataLoader(
        dataset_val_small, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )
    adapt_dict = dict(adapt_loader=adapt_loader, adapt_steps=adapt_steps)
    return adapt_dict


# imports from our packages
def get_mean_and_std_of_dataset(dataset):
    single_img, _ = dataset[0]
    assert torch.is_tensor(single_img)
    num_channels, dim_1, dim_2 = (
        single_img.shape[0],
        single_img.shape[1],
        single_img.shape[2],
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, num_workers=4, shuffle=False
    )
    mean = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)

    var = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])

    std = torch.sqrt(var / (len(loader.dataset) * dim_1 * dim_2))

    return mean, std


class LongTailedCIFAR(Dataset):
    cifar_num_classes_map = {"CIFAR10": 10, "CIFAR100": 100, "TinyImagenet": 200}

    def __init__(
        self,
        root_dir,
        split="train",
        dataset_name="CIFAR10",
        imb_type="exp",
        imb_factor=0.01,
        transform=None,
        target_transform=None,
        return_label=True,
        verbose=False,
        args=None,
    ):

        # check if valid inputs are given
        assert split in ["train", "val", "test"]
        assert dataset_name in ["CIFAR10", "CIFAR100", "TinyImagenet"]

        # set internal attributes and load the dataset
        self.dataset_name = dataset_name
        self.num_classes = self.cifar_num_classes_map[self.dataset_name]
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.args = args

        # load and prepare dataset
        self.load_dataset(root_dir=root_dir, base_folder=dataset_name)
        self.gen_imbalanced_data()

        self.return_label = return_label
        self.verbose = verbose

        self.validate_dataset()

    def load_dataset(self, root_dir, base_folder):
        root = os.path.join(root_dir, base_folder)
        if not os.path.isdir(root):
            os.makedirs(root)

        dataset_kwargs = {
            "root": root,
            # "train": self.split == "train",  # ["train", "val"],
            "train": self.split in ["train", "val"],
            "download": True,
            "transform": self.transform,
            "target_transform": self.target_transform,
        }

        if self.dataset_name == "CIFAR10":
            dataset = datasets.CIFAR10(**dataset_kwargs)
        elif self.dataset_name == "CIFAR100":
            dataset = datasets.CIFAR100(**dataset_kwargs)
        elif self.dataset_name == "TinyImagenet":
            dataset_kwargs["download"] = False
            dataset = TinyImagenet(**dataset_kwargs)
        else:
            raise ValueError("Given dataset name is not supported.")

        if self.split == "test":
            self.dataset = dataset
        else:
            if self.args is None:
                train_length = int(len(dataset) * 0.8)
            else:
                train_length = int(len(dataset) * self.args.train_fraction)

            num_per_class_train = int(train_length / self.num_classes)
            val_length = len(dataset) - train_length
            # train_set, val_set = torch.utils.data.random_split(dataset, [train_length, val_length],
            #                                                    generator=torch.Generator().manual_seed(42))
            train_set, val_set = self.sampleFromClass(dataset, num_per_class_train)
            if self.split == "train":
                self.dataset = train_set
            elif self.split == "val":
                self.dataset = val_set
            else:
                raise ValueError("Unknown split")

    @staticmethod
    def sampleFromClass(ds, k):
        """taken from: https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets"""
        class_counts = {}
        train_data = []
        train_label = []
        val_data = []
        val_label = []
        for data, label in ds:
            c = label
            class_counts[c] = class_counts.get(c, 0) + 1
            if class_counts[c] <= k:
                train_data.append(data[None, ...])
                train_label.append(label)
            else:
                val_data.append(data[None, ...])
                val_label.append(label)
        train_data = torch.cat(train_data)
        train_label = torch.tensor(train_label)
        val_data = torch.cat(val_data)
        val_label = torch.tensor(val_label)

        return (
            TensorDataset(train_data, train_label),
            TensorDataset(val_data, val_label),
        )

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        new_index = self.indices[index]
        img, label = self.dataset[new_index]

        if not self.return_label:
            return img

        return img, label

    def get_num_classes(self):
        return self.num_classes

    def validate_dataset(self):
        stored_return_label = self.return_label
        self.return_label = True

        self.target = []

        for index in range(len(self)):
            _, label = self[index]
            if torch.is_tensor(label):
                # assert tuple(label.shape) == (1,)
                label = label.item()

            self.target.append(label)

        self.target = np.array(self.target, dtype=np.int32)

        unique_classes, class_counts = np.unique(self.target, return_counts=True)

        if self.num_classes != len(unique_classes):
            raise ValueError(
                "Invalid powerlaw distribution, classes have gone extinct!"
            )

        if self.verbose:
            print("\n", self.split, " class labels: ", unique_classes)
            print(self.split, "corresponding class counts: ", class_counts)
            print("Transform: ", self.transform, "\n")

            print("Imbalance type: ", self.imb_type)
            print("Imbalance factor: ", self.imb_factor, "\n")

        class_weights = 1.0 / class_counts
        self.sample_weights = np.array([class_weights[t] for t in self.target])
        self.num_classes = class_counts.shape[0]

        self.return_label = stored_return_label

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        # img_max = len(self.dataset.data) / cls_num
        img_max = len(self.dataset) / cls_num
        img_num_per_cls = []
        if imb_type == "exp":
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == "step":
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self):
        if self.split in ["val", "test"]:
            self.indices = [i for i in range(len(self.dataset))]
        else:
            img_num_per_cls = self.get_img_num_per_cls(
                cls_num=self.num_classes,
                imb_type=self.imb_type,
                imb_factor=self.imb_factor,
            )

            self.img_num_per_cls = img_num_per_cls

            # targets_np = np.array(self.dataset.targets, dtype=np.int64)
            targets_np = np.array(self.dataset.tensors[1], dtype=np.int64)

            classes = np.unique(targets_np)
            self.num_per_cls_dict = dict()
            indices = []
            for the_class, the_img_num in zip(classes, img_num_per_cls):
                self.num_per_cls_dict[the_class] = the_img_num
                idx = np.where(targets_np == the_class)[0]
                idx = idx[:the_img_num]

                indices.extend(idx)

            self.indices = indices

        self.length = len(self.indices)

    def get_cls_num_list(self):
        return self.img_num_per_cls


def get_train_dataset_statistics(
    root_dir, dataset_name, imb_type, imb_factor, args=None
):
    train_dset = LongTailedCIFAR(
        root_dir=root_dir,
        split="train",
        dataset_name=dataset_name,
        transform=transforms.ToTensor(),
        target_transform=None,
        return_label=True,
        imb_type=imb_type,
        imb_factor=imb_factor,
        verbose=False,
        args=args,
    )

    mean, std = get_mean_and_std_of_dataset(dataset=train_dset)
    return mean, std


def get_cifar_transform(
    root_dir, dataset_name, imb_type, imb_factor, trainer_type, flip, args=None
):
    tfm_lst = [transforms.ToTensor()]

    mean, std = get_train_dataset_statistics(
        root_dir=root_dir,
        dataset_name=dataset_name,
        imb_type=imb_type,
        imb_factor=imb_factor,
        args=args,
    )

    if trainer_type == "baseline":
        tfm_lst.append(transforms.Normalize(mean=mean, std=std))
        output_tfm = None

    elif trainer_type == "munit":
        tfm_lst.append(transforms.Normalize(mean=(0.5,), std=(0.5,)))
        output_tfm = transforms.Normalize(mean=mean, std=std)

    else:
        raise ValueError("Given trainer type is not supported.")

    tfms = {}
    tfms["val"] = transforms.Compose(tfm_lst.copy())
    tfms["test"] = transforms.Compose(tfm_lst.copy())

    if flip:
        tfm_lst = [transforms.RandomHorizontalFlip()] + tfm_lst
        if trainer_type == "baseline":
            tfm_lst = [transforms.RandomCrop(32, padding=4)] + tfm_lst

    tfms["train"] = transforms.Compose(tfm_lst.copy())

    return tfms, output_tfm


def load_cifar_datasets(
    root_dir, dataset_name, imb_type, imb_factor, trainer_type, flip=False, args=None
):
    tfms, output_tfm = get_cifar_transform(
        root_dir=root_dir,
        dataset_name=dataset_name,
        imb_type=imb_type,
        imb_factor=imb_factor,
        trainer_type=trainer_type,
        flip=flip,
        args=args,
    )

    dataset_kwargs = {
        "root_dir": root_dir,
        "dataset_name": dataset_name,
        "imb_type": imb_type,
        "imb_factor": imb_factor,
        "target_transform": None,
    }

    splits = ["train", "val", "test"]
    dsets = {}
    for split in splits:
        dsets[split] = LongTailedCIFAR(
            split=split,
            transform=tfms[split],
            return_label=True,
            verbose=True,
            **dataset_kwargs,
            args=args
        )
        dsets[split + "_without_label"] = LongTailedCIFAR(
            split=split,
            transform=tfms[split],
            return_label=False,
            verbose=False,
            **dataset_kwargs,
            args=args
        )

    return dsets, output_tfm
