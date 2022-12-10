import json
import math
import os
import pickle
import random

import numpy as np
import torch
import torchvision
from einops.layers.torch import Rearrange
from torchvision.datasets import MNIST, EMNIST, Omniglot, CIFAR10, CIFAR100, ImageFolder
import torchvision.transforms as transforms
from torch import nn

from models.model_vae import VAE
from models.models_downstream import Classifier, RegressionModel
import lablet_generalization_benchmark.load_dataset as lgb_loader
import os
import pathlib
import shutil

from utils_refactor import cl_utils, imbalanced_data_utils


def get_job_scratch_dir(sub_directory: str = None):
    tmp_dir = os.getenv("TMP")
    # Check if env variable is set
    if tmp_dir is None:
        return None
    # Check if tmp dir exists.
    tmp_dir = pathlib.Path(tmp_dir)
    if not tmp_dir.exists():
        return None
    if sub_directory is None:
        # No sub directory is requested
        return str(tmp_dir)
    tmp_dir = tmp_dir / sub_directory
    tmp_dir.mkdir(exist_ok=True)
    return str(tmp_dir)


def copy_to_scratch(
    path: str,
    scratch_directory: str = None,
    sub_directory: str = "data",
    disable: bool = False,
) -> str:
    # Make sure we're trying to copy something that exists
    path = pathlib.Path(path)
    assert path.exists(), f"Path {str(path)} does not exist, can't copy."
    # Short cut for skipping the local copy
    if bool(int(os.getenv("DISABLE_COPY_TO_SCRATCH", 0))) or disable:
        return str(path)
    # Figure out the scratch directory
    if scratch_directory is None:
        # Make a scratch directory if required
        try:
            scratch_directory = get_job_scratch_dir(sub_directory)
        except ImportError:
            scratch_directory = os.getenv("SLURM_TMPDIR")
            if scratch_directory is not None:
                scratch_directory = os.path.join(scratch_directory, sub_directory)
                os.makedirs(scratch_directory, exist_ok=True)
    if scratch_directory is None:
        raise FileNotFoundError("Couldn't find a scratch directory.")
    scratch_directory = pathlib.Path(scratch_directory)
    target_path = scratch_directory / path.name
    # Check if target path already exists, in which case we just return now
    if target_path.exists():
        return str(target_path)
    # It doesn't exist, so we'll need to make it.
    if path.is_dir():
        # We're copying over recursively
        shutil.copytree(str(path), str(target_path))  # , dirs_exist_ok=True)
    else:
        # We're copying a single file
        shutil.copyfile(str(path), str(target_path))
    # Done.
    return str(target_path)


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, p):
        b = p * torch.log(p)
        b = -1.0 * b.sum()
        return float(b)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def save_model(args, model, train_step):
    if args.folder is not None:
        args.train_step = train_step
        filename = os.path.join(args.folder, args.identifier + "_model.pt")
        filename_json = os.path.join(args.folder, args.identifier + "_config.json")
        with open(filename, "wb") as f:
            state_dict = model.state_dict()
            torch.save(state_dict, f)
        with open(filename_json, "w") as f:
            json.dump(vars(args), f, indent=4)


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


def get_retrieval(args):
    if args.buffer_path == "none":
        return None
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder_model = load_buffer_model(args)
        retrieval_data = torch.load(args.buffer_path, map_location=device)
        for key in retrieval_data:
            retrieval_data[key] = retrieval_data[key].to(device=device)
            retrieval_data[key] = retrieval_data[key].float()

        stored_reps = retrieval_data["reps"]
        stored_labels = retrieval_data["labels"].float()

        if args.no_labels_from_reps:
            retrieval_data = stored_reps
            buffer_dims = [stored_reps.shape[-1]]
        else:
            retrieval_data = torch.cat((stored_reps, stored_labels), dim=1)
            buffer_dims = [stored_reps.shape[-1], stored_labels.shape[-1]]

        retrieval_data_dim = retrieval_data.shape[1]
        retrieval_data = torch.unsqueeze(retrieval_data, 0)
        retrieval_kwargs = {
            "attn_retrieval": args.attn_retrieval,
            "context_dim": retrieval_data_dim,
            "buffer_dims": buffer_dims,
            "sampled_buffer_size": args.sampled_buffer_size,
            "retrieval_access": args.retrieval_access,
            "embedding_from_query": args.retrieval_access
            == "knn_from_learned_embedding",
            "retrieval_query_dropout": args.retrieval_query_dropout,
            "randomize_reps": True if args.randomize_reps == "true" else False,
            "retrieval_corruption": args.retrieval_corruption,
            "topk": args.topk,
            "retrieval_last": args.retrieval_last,
            "add_distances": args.add_distances,
        }

        return {
            "retrieval_data": retrieval_data,
            "encoder_model": encoder_model,
            "retrieval_kwargs": retrieval_kwargs,
        }


def get_retrieval_omniglot(args, buffer_path):
    if buffer_path == "none":
        return None
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder_model = load_buffer_model(args)
        retrieval_data = torch.load(buffer_path, map_location=device)
        for key in retrieval_data:
            retrieval_data[key] = retrieval_data[key].to(device=device)
            retrieval_data[key] = retrieval_data[key].float()

        retrieval_data_dim = retrieval_data["reps"].shape[1]
        retrieval_kwargs = {
            "attn_retrieval": args.attn_retrieval,
            "buffer_dims": [retrieval_data_dim],
            "context_dim": retrieval_data_dim,
            "sampled_buffer_size": args.sampled_buffer_size,
            "retrieval_access": args.retrieval_access,
            "embedding_from_query": args.retrieval_access
            == "knn_from_learned_embedding",
            "retrieval_query_dropout": args.retrieval_query_dropout,
            "randomize_reps": True if args.randomize_reps == "true" else False,
            "retrieval_corruption": args.retrieval_corruption,
            "topk": args.topk,
            "retrieval_last": args.retrieval_last,
        }

        return {
            "retrieval_data": retrieval_data,
            "encoder_model": encoder_model,
            "retrieval_kwargs": retrieval_kwargs,
        }


class MultiOmniglotDataset(torchvision.datasets.VisionDataset):
    def __init__(self, root, transform=None):
        super(MultiOmniglotDataset, self).__init__(root=root)
        self.transform = transform
        images_path = os.path.join(root, "images.npy")
        targets_path = os.path.join(root, "targets.pkl")

        self.images = np.load(images_path)
        with open(targets_path, "rb") as fp:
            self.targets = pickle.load(fp)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        sample = self.images[index]
        targets = self.targets[index]
        if self.transform:
            sample = self.transform(sample)
        return sample, torch.tensor(targets)


class MultiLabelImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, multihot_targets=True, **kwargs):
        super(MultiLabelImageFolder, self).__init__(**kwargs)
        self.multihot_targets = multihot_targets

    def __len__(self):
        return super().__len__()

    def __getitem__(self, item):
        sample, target = super().__getitem__(item)
        digits = [int(i) for i in list(self.samples[item][0].split("/")[-2])]
        if self.multihot_targets:
            multi_hot_target = torch.zeros(10, dtype=torch.float)
            for digit in digits:
                multi_hot_target[digit] = 1.0
            return sample, multi_hot_target
        return sample, torch.tensor(digits)


def collate_fn(batch):
    data = [item[0] for item in batch]
    target = [{"labels": item[1]} for item in batch]
    return torch.stack(data), target


def get_omniglot_loaders(args):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Grayscale(num_output_channels=1),
            # AddGaussianNoise(mean=0, std=args.image_noise),
        ]
    )

    train_set_path = (
        args.train_dataset_paths[0]
        if isinstance(args.train_dataset_paths, list)
        else args.train_dataset_paths
    )
    dataset_train = MultiOmniglotDataset(
        root=os.path.join(args.root_dir, train_set_path, "train"), transform=transforms
    )
    dataset_test = {
        testset: MultiOmniglotDataset(
            root=os.path.join(args.root_dir, testset, "test"), transform=transforms
        )
        for testset in args.test_dataset_paths
    }

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    data_loaders_test = {
        testset: torch.utils.data.DataLoader(
            dataset_test[testset],
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
        )
        for testset in dataset_test
    }

    return dataloader_train, data_loaders_test


def get_multimnist_loaders(args):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Grayscale(num_output_channels=1),
            AddGaussianNoise(mean=0, std=args.image_noise),
        ]
    )

    train_set_path = (
        args.train_dataset_paths[0]
        if isinstance(args.train_dataset_paths, list)
        else args.train_dataset_paths
    )
    dataset_train = MultiLabelImageFolder(
        multihot_targets=args.multihot_labels,
        root=os.path.join(args.root_dir, train_set_path, "train"),
        transform=transforms,
    )
    dataset_test = {
        testset: MultiLabelImageFolder(
            multihot_targets=args.multihot_labels,
            root=os.path.join(args.root_dir, testset, "test"),
            transform=transforms,
        )
        for testset in args.test_dataset_paths
    }

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    data_loaders_test = {
        testset: torch.utils.data.DataLoader(
            dataset_test[testset],
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
        )
        for testset in dataset_test
    }

    return dataloader_train, data_loaders_test


def multi_mnist_datasets(args):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(num_output_channels=1),
        ]
    )

    train_datasets = [
        MultiLabelImageFolder(
            os.path.join(args.root_dir, dataset_dir, "train"), transform=transforms
        )
        for dataset_dir in args.train_dataset_paths
    ]
    test_datasets = [
        MultiLabelImageFolder(
            os.path.join(args.root_dir, dataset_dir, "test"), transform=transforms
        )
        for dataset_dir in args.train_dataset_paths
    ]

    return train_datasets, test_datasets


def vanilla_mnist_loaders(args, type="mnist"):
    print(type)
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    if type == "emnist":
        dataset_train = EMNIST(
            root=args.root_dir,
            split="letters",
            train=True,
            transform=transform,
            download=True,
        )
        dataset_test = EMNIST(
            root=args.root_dir,
            split="letters",
            train=False,
            transform=transform,
            download=True,
        )
    elif type == "mnist":
        dataset_train = MNIST(
            root=args.root_dir, train=True, transform=transform, download=True
        )
        dataset_test = MNIST(
            root=args.root_dir, train=False, transform=transform, download=True
        )
    elif type == "dsprites_random":
        path = os.path.join(args.root_dir, "indomain_generalization")
        dataloader_train = lgb_loader.load_dataset(
            "dsprites",
            "random",
            batch_size=args.batch_size,
            dataset_path=path,
            mode="train",
            normalize=True,
        )
        dataloader_test = lgb_loader.load_dataset(
            "dsprites",
            "random",
            batch_size=args.batch_size,
            dataset_path=path,
            mode="test",
            normalize=True,
        )
        return dataloader_test, dataloader_train
    elif type == "dsprites_composition":
        path = os.path.join(args.root_dir, "indomain_generalization")
        dataloader_train = lgb_loader.load_dataset(
            "dsprites",
            "composition",
            batch_size=args.batch_size,
            dataset_path=path,
            mode="train",
            normalize=True,
        )
        dataloader_test = lgb_loader.load_dataset(
            "dsprites",
            "composition",
            batch_size=args.batch_size,
            dataset_path=path,
            mode="test",
            normalize=True,
        )
        return dataloader_test, dataloader_train
    elif type == "dsprites_interpolation":
        path = os.path.join(args.root_dir, "indomain_generalization")
        dataloader_train = lgb_loader.load_dataset(
            "dsprites",
            "interpolation",
            batch_size=args.batch_size,
            dataset_path=path,
            mode="train",
            normalize=True,
        )
        dataloader_test = lgb_loader.load_dataset(
            "dsprites",
            "interpolation",
            batch_size=args.batch_size,
            dataset_path=path,
            mode="test",
            normalize=True,
        )
        return dataloader_test, dataloader_train
    elif type == "dsprites_extrapolation":
        path = os.path.join(args.root_dir, "indomain_generalization")
        dataloader_train = lgb_loader.load_dataset(
            "dsprites",
            "extrapolation",
            batch_size=args.batch_size,
            dataset_path=path,
            mode="train",
            normalize=True,
        )
        dataloader_test = lgb_loader.load_dataset(
            "dsprites",
            "extrapolation",
            batch_size=args.batch_size,
            dataset_path=path,
            mode="test",
            normalize=True,
        )
        return dataloader_test, dataloader_train
    elif type == "shapes3d_random":
        path = os.path.join(args.root_dir, "indomain_generalization")
        dataloader_train = lgb_loader.load_dataset(
            "shapes3d",
            "random",
            batch_size=args.batch_size,
            dataset_path=path,
            mode="train",
            normalize=True,
        )
        dataloader_test = lgb_loader.load_dataset(
            "shapes3d",
            "random",
            batch_size=args.batch_size,
            dataset_path=path,
            mode="test",
            normalize=True,
        )
        return dataloader_test, dataloader_train
    elif type == "shapes3d_composition":
        path = os.path.join(args.root_dir, "indomain_generalization")
        dataloader_train = lgb_loader.load_dataset(
            "shapes3d",
            "composition",
            batch_size=args.batch_size,
            dataset_path=path,
            mode="train",
            normalize=True,
        )
        dataloader_test = lgb_loader.load_dataset(
            "shapes3d",
            "composition",
            batch_size=args.batch_size,
            dataset_path=path,
            mode="test",
            normalize=True,
        )
        return dataloader_test, dataloader_train
    elif type == "shapes3d_interpolation":
        path = os.path.join(args.root_dir, "indomain_generalization")
        dataloader_train = lgb_loader.load_dataset(
            "shapes3d",
            "interpolation",
            batch_size=args.batch_size,
            dataset_path=path,
            mode="train",
            normalize=True,
        )
        dataloader_test = lgb_loader.load_dataset(
            "shapes3d",
            "interpolation",
            batch_size=args.batch_size,
            dataset_path=path,
            mode="test",
            normalize=True,
        )
        return dataloader_test, dataloader_train
    elif type == "shapes3d_extrapolation":
        path = os.path.join(args.root_dir, "indomain_generalization")
        dataloader_train = lgb_loader.load_dataset(
            "shapes3d",
            "extrapolation",
            batch_size=args.batch_size,
            dataset_path=path,
            mode="train",
            normalize=True,
        )
        dataloader_test = lgb_loader.load_dataset(
            "shapes3d",
            "extrapolation",
            batch_size=args.batch_size,
            dataset_path=path,
            mode="test",
            normalize=True,
        )
        return dataloader_test, dataloader_train
    elif type == "mpi3d_random":
        path = os.path.join(args.root_dir, "indomain_generalization")
        dataloader_train = lgb_loader.load_dataset(
            "mpi3d",
            "random",
            batch_size=args.batch_size,
            dataset_path=path,
            mode="train",
            normalize=True,
        )
        dataloader_test = lgb_loader.load_dataset(
            "mpi3d",
            "random",
            batch_size=args.batch_size,
            dataset_path=path,
            mode="test",
            normalize=True,
        )
        return dataloader_test, dataloader_train
    elif type == "mpi3d_composition":
        path = os.path.join(args.root_dir, "indomain_generalization")
        dataloader_train = lgb_loader.load_dataset(
            "mpi3d",
            "composition",
            batch_size=args.batch_size,
            dataset_path=path,
            mode="train",
            normalize=True,
        )
        dataloader_test = lgb_loader.load_dataset(
            "mpi3d",
            "composition",
            batch_size=args.batch_size,
            dataset_path=path,
            mode="test",
            normalize=True,
        )
        return dataloader_test, dataloader_train
    elif type == "mpi3d_interpolation":
        path = os.path.join(args.root_dir, "indomain_generalization")
        dataloader_train = lgb_loader.load_dataset(
            "mpi3d",
            "interpolation",
            batch_size=args.batch_size,
            dataset_path=path,
            mode="train",
            normalize=True,
        )
        dataloader_test = lgb_loader.load_dataset(
            "mpi3d",
            "interpolation",
            batch_size=args.batch_size,
            dataset_path=path,
            mode="test",
            normalize=True,
        )
        return dataloader_test, dataloader_train
    elif type == "mpi3d_extrapolation":
        path = os.path.join(args.root_dir, "indomain_generalization")
        dataloader_train = lgb_loader.load_dataset(
            "mpi3d",
            "extrapolation",
            batch_size=args.batch_size,
            dataset_path=path,
            mode="train",
            normalize=True,
        )
        dataloader_test = lgb_loader.load_dataset(
            "mpi3d",
            "extrapolation",
            batch_size=args.batch_size,
            dataset_path=path,
            mode="test",
            normalize=True,
        )
        return dataloader_test, dataloader_train
    elif type == "shapes3d-classes":
        path = os.path.join(args.root_dir, "indomain_generalization")
        dataloader_train = lgb_loader.load_dataset(
            "shapes3d",
            "random",
            batch_size=args.batch_size,
            dataset_path=path,
            mode="train",
            normalize=False,
        )
        dataloader_test = lgb_loader.load_dataset(
            "shapes3d",
            "random",
            batch_size=args.batch_size,
            dataset_path=path,
            mode="test",
            normalize=False,
        )
        return dataloader_test, dataloader_train
    elif type == "omniglot":
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Resize((28, 28))]
        )
        dataset_train = Omniglot(
            root=args.root_dir, background=True, transform=transform, download=True
        )
        dataset_test = Omniglot(
            root=args.root_dir, background=False, transform=transform, download=True
        )
    elif type == "omniglot_full":
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Resize((28, 28))]
        )
        dataset_train = Omniglot(
            root=args.root_dir, background=True, transform=transform, download=True
        )
        dataset_test = Omniglot(
            root=args.root_dir, background=False, transform=transform, download=True
        )
        dataset_train = torch.utils.data.ConcatDataset([dataset_train, dataset_test])
    elif type == "cifar10":
        # transform = torchvision.transforms.Compose(
        #     [torchvision.transforms.ToTensor(), torchvision.transforms.Resize((28, 28))])
        dataset_train = CIFAR10(
            root=args.root_dir, train=True, transform=transform, download=True
        )
        dataset_test = CIFAR10(
            root=args.root_dir, train=False, transform=transform, download=True
        )
    elif type == "cifar100":
        # transform = torchvision.transforms.Compose(
        #     [torchvision.transforms.ToTensor(), torchvision.transforms.Resize((28, 28))])
        dataset_train = CIFAR100(
            root=args.root_dir, train=True, transform=transform, download=True
        )
        dataset_test = CIFAR100(
            root=args.root_dir, train=False, transform=transform, download=True
        )
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    return dataloader_test, dataloader_train


def load_buffer_model(args, channels=3, resnet=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (
        "mnist" in args.dataset_name
        or "MNIST" in args.dataset_name
        or "omniglot" in args.dataset_name
    ):
        channels = 1
    model = VAE(
        z_dim=args.buffer_model_reps_dim,
        image_size=args.image_size,
        channels=channels,
        resnet=resnet,
    )
    checkpoint = torch.load(args.buffer_model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def get_downstream_models(args):
    downstream_models = dict()
    if args.weighted_values:
        input_dim = args.dim_value * args.num_queries
    else:
        input_dim = args.dim_value * args.num_queries * args.topk
    if args.bottleneck_mode == "no_bottleneck" or args.bottleneck_mode == "vq_baseline":
        input_dim = args.dim_value * args.num_queries

    if not args.concat_fetched_values and args.bottleneck_mode == "key_multi_values":
        input_dim = args.dim_value

    if args.add_positional_encoding:
        input_axis = 1  # is 1 because values are presented as a batch of vectors
        input_dim += input_axis * ((args.num_freq_bands * 2) + 1)

    if args.dataset_name == "mnist":
        downstream_models["digit"] = Classifier(input_dim=input_dim, num_classes=10)
    elif args.dataset_name == "mnist_colours":
        downstream_models["digit"] = Classifier(input_dim=input_dim, num_classes=10)
        downstream_models["bg_color"] = Classifier(input_dim=input_dim, num_classes=5)
        downstream_models["fg_color"] = Classifier(input_dim=input_dim, num_classes=5)
    elif args.dataset_name == "shapes3d-classes":
        downstream_models["floor_hue"] = Classifier(input_dim=input_dim, num_classes=10)
        downstream_models["wall_hue"] = Classifier(input_dim=input_dim, num_classes=10)
        downstream_models["object_hue"] = Classifier(
            input_dim=input_dim, num_classes=10
        )
        downstream_models["scale"] = Classifier(input_dim=input_dim, num_classes=8)
        downstream_models["shape"] = Classifier(input_dim=input_dim, num_classes=4)
        downstream_models["orientation"] = Classifier(
            input_dim=input_dim, num_classes=15
        )
    elif "shapes3d_" in args.dataset_name:
        downstream_models["floor_hue"] = RegressionModel(input_dim=input_dim)
        downstream_models["wall_hue"] = RegressionModel(input_dim=input_dim)
        downstream_models["object_hue"] = RegressionModel(input_dim=input_dim)
        downstream_models["scale"] = RegressionModel(input_dim=input_dim)
        downstream_models["shape"] = RegressionModel(input_dim=input_dim)
        downstream_models["orientation"] = RegressionModel(input_dim=input_dim)
    elif "mpi3d_" in args.dataset_name:
        downstream_models["color"] = RegressionModel(input_dim=input_dim)
        downstream_models["shape"] = RegressionModel(input_dim=input_dim)
        downstream_models["size"] = RegressionModel(input_dim=input_dim)
        downstream_models["height"] = RegressionModel(input_dim=input_dim)
        downstream_models["bg_color"] = RegressionModel(input_dim=input_dim)
        downstream_models["x_axis"] = RegressionModel(input_dim=input_dim)
        downstream_models["y_axis"] = RegressionModel(input_dim=input_dim)
    elif "dsprites_" in args.dataset_name:
        downstream_models["shape"] = RegressionModel(input_dim=input_dim)
        downstream_models["scale"] = RegressionModel(input_dim=input_dim)
        downstream_models["orientation"] = RegressionModel(input_dim=input_dim)
        downstream_models["x_position"] = RegressionModel(input_dim=input_dim)
        downstream_models["y_position"] = RegressionModel(input_dim=input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model in downstream_models:
        downstream_models[model].to(device)
    return downstream_models


def load_pretrained_model(args, resnet=False):
    pretrained_model = load_buffer_model(
        args, channels=args.input_channels, resnet=resnet
    )
    return pretrained_model


def get_init_data_loader(args):
    if args.pretrain_data == "TinyImagenet":
        from avalanche.benchmarks.datasets.dataset_utils import default_dataset_location

        local_path = str(default_dataset_location("tinyimagenet"))
        root_dir = copy_to_scratch(local_path)
        print(f"local path: {root_dir}")
        init_data_loader = cl_utils.load_TinyImagenet_loader(
            args, imagenet_root=root_dir
        )
    elif args.pretrain_data == "Imagenet":
        args.init_epochs = 1
        local_path = os.path.join("/is/cluster/ftraeuble", "imagenet-train")
        root_dir = copy_to_scratch(local_path)
        print(f"local path: {root_dir}")
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        transforms_imagenet = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        init_data_set = ImageFolder(root=root_dir, transform=transforms_imagenet)
        init_data_loader = torch.utils.data.DataLoader(
            init_data_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    elif args.pretrain_data == "Imagenet_ffcv":
        from ffcv.writer import DatasetWriter
        from ffcv.fields import RGBImageField, IntField
        from ffcv.loader import Loader, OrderOption
        from ffcv.transforms import (
            ToTensor,
            ToDevice,
            ToTorchImage,
            Cutout,
            NormalizeImage,
        )
        from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder

        args.init_epochs = 1
        local_path = os.path.join("/is/cluster/ftraeuble", "imagenet-train")
        root_dir = copy_to_scratch(local_path)
        print(f"local path: {root_dir}")
        init_data_set = ImageFolder(root=root_dir)
        write_path = root_dir + "_ds.beton"

        # Pass a type for each data field
        writer = DatasetWriter(
            write_path,
            {
                # Tune options to optimize dataset size, throughput at train-time
                "image": RGBImageField(max_resolution=256, jpeg_quality=0),
                "label": IntField(),
            },
        )

        # Write dataset
        writer.from_indexed_dataset(init_data_set)

        # Random resized crop
        decoder = RandomResizedCropRGBImageDecoder((224, 224))
        normalize = NormalizeImage(
            mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])
        )

        # Data decoding and augmentation
        image_pipeline = [
            decoder,
            normalize,
            Cutout(),
            ToTensor(),
            ToTorchImage(),
            ToDevice(0),
        ]
        label_pipeline = [IntDecoder(), ToTensor(), ToDevice(0)]

        # Pipeline for each data field
        pipelines = {"image": image_pipeline, "label": label_pipeline}
        init_data_loader = Loader(
            write_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            order=OrderOption.RANDOM,
            pipelines=pipelines,
        )
    elif args.pretrain_data == "Imagenet_ffcv_v2":
        from ffcv.writer import DatasetWriter
        from ffcv.fields import RGBImageField, IntField
        from ffcv.loader import Loader, OrderOption
        from ffcv.transforms import (
            ToTensor,
            ToDevice,
            ToTorchImage,
            Cutout,
            NormalizeImage,
            Squeeze,
        )
        from ffcv.fields.decoders import (
            IntDecoder,
            RandomResizedCropRGBImageDecoder,
            CenterCropRGBImageDecoder,
        )
        from ffcv.fields.rgb_image import CenterCropRGBImageDecoder
        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
        IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
        DEFAULT_CROP_RATIO = 224 / 256

        args.init_epochs = 1
        local_path = os.path.join("/is/cluster/ftraeuble", "imagenet-train")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scratch_directory = get_job_scratch_dir("data")
        scratch_directory = pathlib.Path(scratch_directory)
        target_path = scratch_directory / pathlib.Path(local_path).name
        target_path.mkdir(exist_ok=True)
        init_data_set = ImageFolder(root=local_path)
        write_path = target_path / "ds.beton"
        print(f"local path: {target_path}")

        # Pass a type for each data field
        writer = DatasetWriter(
            write_path,
            {
                # Tune options to optimize dataset size, throughput at train-time
                "image": RGBImageField(max_resolution=256, jpeg_quality=0),
                "label": IntField(),
            },
        )

        # Write dataset
        writer.from_indexed_dataset(init_data_set)

        res_tuple = (224, 224)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(torch.device(device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(
                IMAGENET_MEAN,
                IMAGENET_STD,
                np.float16,
            ),
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(device), non_blocking=True),
        ]

        init_data_loader = Loader(
            write_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            order=OrderOption.SEQUENTIAL,
            drop_last=False,
            pipelines={"image": image_pipeline, "label": label_pipeline},
        )
    elif args.pretrain_data == "CIFAR10":
        init_data_loader, _, _ = imbalanced_data_utils.load_imbalanced_data_loaders(
            args.root_dir, args, dataset_name="CIFAR10"
        )
    elif args.pretrain_data == "CIFAR100":
        init_data_loader, _, _ = imbalanced_data_utils.load_imbalanced_data_loaders(
            args.root_dir, args, dataset_name="CIFAR100"
        )
    else:
        raise ValueError(f"pretrain_data {args.pretrain_data} not supported")
    return init_data_loader


def get_class_nums(args):
    if args.dataset_name == "CIFAR10":
        class_nums = 10
    elif args.dataset_name == "CIFAR100":
        class_nums = 100
    elif args.dataset_name == "TinyImagenet":
        class_nums = 200
    elif args.dataset_name == "Imagenet":
        class_nums = 1000
    else:
        raise NotImplementedError("Dataset {} not supported".format(args.dataset_name))
    return class_nums


def get_decoder_module(dim_value_in, args):
    class_nums = get_class_nums(args)

    decoder_modules = []
    if args.accept_image_fmap:
        decoder_modules += [
            Rearrange("b ... d -> b d ..."),
            nn.AdaptiveAvgPool2d((1, 1)),
        ]
    if args.decoder_size == "one-layer":
        decoder_modules += [
            Rearrange("b ... -> b (...)"),
            nn.Linear(dim_value_in, 256),
            nn.ReLU(),
            nn.Linear(256, class_nums),
        ]
    elif args.decoder_size == "two-layer":
        decoder_modules += [
            Rearrange("b ... -> b (...)"),
            nn.Linear(dim_value_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, class_nums),
        ]
    else:
        raise NotImplementedError(
            "Decoder size {} not supported".format(args.decoder_size)
        )
    return decoder_modules


def get_backbone_path(bottlenecked_encoder, args, prefix="backbone_"):
    model_dir = os.path.join(args.root_dir, "kv_model_backbones")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    name = prefix
    attribute_dict = {
        "nc": bottlenecked_encoder.num_channels,
        "ncb": bottlenecked_encoder.num_codebooks,
        "nkvpcb": bottlenecked_encoder.key_value_pairs_per_codebook,
        "dk": bottlenecked_encoder.dim_keys,
        "dv": bottlenecked_encoder.dim_values,
        "decay": bottlenecked_encoder.decay,
        "eps": bottlenecked_encoder.eps,
        "threshold": bottlenecked_encoder.threshold_ema_dead_code,
        "concat_v": bottlenecked_encoder.concat_values_from_all_codebooks,
        "splitting_mode": bottlenecked_encoder.splitting_mode,
        "init_mode": bottlenecked_encoder.init_mode,
        "kmeans_iters": bottlenecked_encoder.kmeans_iters,
    }
    for key, value in attribute_dict.items():
        name += f"_{key}_{value}"
    name += "_init_data_" + args.pretrain_data
    name += "_init_epochs_" + str(args.init_epochs)
    name += "_batch_size_" + str(args.batch_size)
    name += "_pretrain_data_" + str(args.pretrain_data)
    name += "_seed_" + str(args.seed)
    name += ".pt"
    return os.path.join(model_dir, name)


def get_optimizer(model, args):
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    else:
        raise NotImplementedError("Optimizer {} not supported".format(args.optimizer))
    return optimizer


class ModelWrapper(nn.Module):
    def __init__(self, bottlenecked_encoder, decoder_modules, args):
        super(ModelWrapper, self).__init__()
        self.bottlenecked_encoder = bottlenecked_encoder
        self.bottlenecked_encoder.freeze_encoder()
        self.decoder = nn.Sequential(*decoder_modules)
        self.args = args
        if self.args.method == "ours":
            self.tuple_pos = 0  # this is the position of the returned value codes
        elif self.args.method == "kv_tune_full_decoder":
            self.tuple_pos = 0  # this is the position of the returned value codes
        elif self.args.method == "vq_tune_single_layer":
            self.tuple_pos = 1  # this is the position of the returned key codes
        elif self.args.method == "vq_tune_full_decoder":
            self.tuple_pos = 1  # this is the position of the returned key codes
        elif self.args.method == "mlp":
            self.tuple_pos = -1  # this is the position of the returned key codes
        else:
            raise NotImplementedError("Method {} not supported".format(args.method))

        self.dropout_layer = nn.Dropout3d(p=float(self.args.ff_dropout))

    def forward(self, x):
        bottleneck_tuple = self.bottlenecked_encoder(x)
        if self.args.method == "mlp":
            x = bottleneck_tuple[self.tuple_pos].clone().detach()
            x = self.decoder(x)
        else:
            x = self.dropout_layer(bottleneck_tuple[self.tuple_pos])
            x = self.decoder(x)
        return x

    def freeze_for_adaptation(self):
        if self.args.method == "ours":
            for param in self.decoder.parameters():
                param.requires_grad = False
        elif self.args.method == "vq_tune_single_layer":
            # Freeze all decoder layers except for first Linear Layer
            for param in self.decoder.parameters():
                if param.requires_grad:
                    param.requires_grad = False
            for decoder_layer in self.decoder:
                if isinstance(decoder_layer, nn.Linear):
                    for param in decoder_layer.parameters():
                        param.requires_grad = True
                    return  # Stop after first linear layer is found
        elif self.args.method == "kv_tune_full_decoder":
            return
        elif self.args.method == "vq_tune_full_decoder":
            return
        elif self.args.method == "mlp":
            return
        else:
            raise NotImplementedError(
                "Method {} not supported".format(self.args.method)
            )
