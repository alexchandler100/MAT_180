#parts of code from https://colab.research.google.com/github/reiinakano/invariant-risk-minimization/blob/master/invariant_risk_minimization_colored_mnist.ipynb#scrollTo=L4qZtXx_weBb

import os
from PIL import Image

import wandb
import numpy as np
import torch
from torchvision import datasets
import torchvision.datasets.utils as dataset_utils
from torchvision.transforms import transforms

cmnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
])


def get_adapt_modes(args):
    adapt_modes = ["train1",
                   "train2",
                   "test",
                   "all_train",
                   "uncorrelated"]

    return adapt_modes


def get_adapt_batch(adapt_mode, args):
    adapt_loader = torch.utils.data.DataLoader(ColoredMNIST(root=args.root_dir,
                                                            env=adapt_mode, transform=cmnist_transform),
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers)

    adapt_steps = 1000
    adapt_dict = dict(adapt_loader=adapt_loader,
                      adapt_steps=adapt_steps)
    return adapt_dict


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
            outputs = model(inputs)[:, 0]
            total += labels.size(0)
            pred = torch.where(torch.gt(outputs, torch.Tensor([0.0]).to(device)),
                               torch.Tensor([1.0]).to(device),
                               torch.Tensor([0.0]).to(device))  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()

        acc1 = 100 * float(correct / total)
        log_dict = {prefix + "test_accuracy": acc1,
                    "epochs": epoch,
                    "train_step": train_step}
        wandb.log(log_dict)


def load_data_loaders(args):
    test_loader = torch.utils.data.DataLoader(ColoredMNIST(root=args.root_dir,
                                                           env='test', transform=cmnist_transform),
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers)

    train1_loader = torch.utils.data.DataLoader(ColoredMNIST(root=args.root_dir,
                                                             env='train1', transform=cmnist_transform),
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers)

    train2_loader = torch.utils.data.DataLoader(ColoredMNIST(root=args.root_dir,
                                                             env='train2', transform=cmnist_transform),
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers)
    return train1_loader, train2_loader, test_loader


def color_grayscale_arr(arr, red=True):
    """Converts grayscale image to either red or green"""
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])
    if red:
        arr = np.concatenate([arr,
                              np.zeros((h, w, 2), dtype=dtype)], axis=2)
    else:
        arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                              arr,
                              np.zeros((h, w, 1), dtype=dtype)], axis=2)
    return arr


class ColoredMNIST(datasets.VisionDataset):
    """
  Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

  Args:
    root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
    env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test', or 'all_train'.
    transform (callable, optional): A function/transform that  takes in an PIL image
      and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
      target and transforms it.
  """

    def __init__(self, root='dataset', env='train1', transform=None, target_transform=None):
        super(ColoredMNIST, self).__init__(root, transform=transform,
                                           target_transform=target_transform)

        self.prepare_colored_mnist()
        if env in ['train1', 'train2', 'test', "uncorrelated"]:
            self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', env) + '.pt')
        elif env == 'all_train':
            self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', 'train1.pt')) + \
                                     torch.load(os.path.join(self.root, 'ColoredMNIST', 'train2.pt'))
        else:
            raise RuntimeError(f'{env} env unknown. Valid envs are train1, train2, test, and all_train')

    def __getitem__(self, index):
        """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
        img, target = self.data_label_tuples[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data_label_tuples)

    def prepare_colored_mnist(self):
        colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
        if os.path.exists(os.path.join(colored_mnist_dir, 'train1.pt')) \
                and os.path.exists(os.path.join(colored_mnist_dir, 'train2.pt')) \
                and os.path.exists(os.path.join(colored_mnist_dir, 'uncorrelated.pt')) \
                and os.path.exists(os.path.join(colored_mnist_dir, 'test.pt')):
            print('Colored MNIST dataset already exists')
            return

        print('Preparing Colored MNIST')
        train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

        train1_set = []
        train2_set = []
        test_set = []
        uncorrelated_set = []
        for idx, (im, label) in enumerate(train_mnist):
            if idx % 10000 == 0:
                print(f'Converting image {idx}/{len(train_mnist)}')
            im_array = np.array(im)

            # Assign a binary label y to the image based on the digit
            binary_label = 0 if label < 5 else 1

            # Flip label with 25% probability
            if np.random.uniform() < 0.25:
                binary_label = binary_label ^ 1

            # Color the image either red or green according to its possibly flipped label
            color_red = binary_label == 0

            # Flip the color with a probability e that depends on the environment
            if idx < 20000:
                # 20% in the first training environment
                if np.random.uniform() < 0.2:
                    color_red = not color_red
            elif idx < 40000:
                # 10% in the first training environment
                if np.random.uniform() < 0.1:
                    color_red = not color_red
            else:
                # 90% in the test environment
                if np.random.uniform() < 0.9:
                    color_red = not color_red

            colored_arr = color_grayscale_arr(im_array, red=color_red)

            if idx < 20000:
                train1_set.append((Image.fromarray(colored_arr), binary_label))
            elif idx < 40000:
                train2_set.append((Image.fromarray(colored_arr), binary_label))
            else:
                test_set.append((Image.fromarray(colored_arr), binary_label))

            randomly_colored_arr = color_grayscale_arr(im_array, red=np.random.uniform() < 0.5)
            uncorrelated_set.append((Image.fromarray(randomly_colored_arr), binary_label))
            # Debug
            # print('original label', type(label), label)
            # print('binary label', binary_label)
            # print('assigned color', 'red' if color_red else 'green')
            # plt.imshow(colored_arr)
            # plt.show()
            # break

        #check of dir exists
        if not os.path.exists(colored_mnist_dir):
            os.makedirs(colored_mnist_dir)
        torch.save(train1_set, os.path.join(colored_mnist_dir, 'train1.pt'))
        torch.save(train2_set, os.path.join(colored_mnist_dir, 'train2.pt'))
        torch.save(test_set, os.path.join(colored_mnist_dir, 'test.pt'))
        torch.save(uncorrelated_set, os.path.join(colored_mnist_dir, 'uncorrelated.pt'))
