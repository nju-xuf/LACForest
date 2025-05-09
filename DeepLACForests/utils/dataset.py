from torchvision import transforms, datasets
from torch.utils.data import ConcatDataset
from scipy.io import loadmat, savemat
from scipy.sparse import csc_matrix as spmtx
import numpy as np

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
svhn_mean = (0.4391, 0.4457, 0.4740)
svhn_std = (0.1990, 0.2020, 0.2034)


def read_from_mat(path, keys):
    f = loadmat(path)
    result = {}
    for key in keys:
        data_np = None
        if key not in f.keys():
            print('{} does not exists.'.format(key))
        else:
            data = f[key]
            if isinstance(data, spmtx):
                data_np = data.toarray()
            elif isinstance(data, np.ndarray):
                data_np = data
            else:
                print('Unkown Type: {}'.format(type(data)))
        result[key] = data_np
    return result


def load_transform(dataset, train):
    """
    Load transforms for training and test data

    Parameters
    ----------
    dataset: str
        The name of dataset
    train: bool
        True for training set and False for test set

    Returns
    -------
    transform_train, transform_test : object
        Transforms for training and test data.
    """
    if dataset not in ['cifar10', 'svhn', 'cifar100', 'mnist', 'fmnist', 'kuzushiji']:
        raise ValueError(
            'Dataset must be selected within kuzushiji, cifar10, cifar100, svhn, mnist and fmnist.')
    if dataset in ['mnist', 'fmnist', 'kuzushiji']:
        return transforms.ToTensor()
    else:
        transform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    ])
        return transform


def load_dataset(dataset, setting=None):
    """
    Load a dataset

    Parameters
    ----------
    dataset: str
        The name of dataset
    train: bool
        True for training set and False for test set

    Returns
    -------
    data : object
        The dataset.
    """
    configuration_path = setting['configuration_path']
    known_class_number = setting['known_class_number']
    transform_train = setting['transform_train']
    transform_test = setting['transform_test']

    if dataset not in ['cifar10', 'cifar100', 'svhn', 'mnist', 'fmnist', 'kuzushiji']:
        raise ValueError(
            'Dataset must be selected within svhn, cifar10 and cifar100.')
    if dataset == 'cifar10':
        train_set = datasets.CIFAR10(root='dataset/cifar10/data', train=True, download=True)
        test_set = datasets.CIFAR10(root='dataset/cifar10/data', train=False, download=True)
        full_set = ConcatDataset([train_set, test_set])
    elif dataset == 'svhn':
        train_set = datasets.SVHN(root='dataset/svhn/data', split='train', download=True)
        test_set = datasets.SVHN(root='dataset/svhn/data', split='test', download=True)
        full_set = ConcatDataset([train_set, test_set])
    elif dataset == 'mnist':
        train_set = datasets.MNIST(root='dataset/mnist/data', train=True, download=True)
        test_set = datasets.MNIST(root='dataset/mnist/data', train=False, download=True)
        full_set = ConcatDataset([train_set, test_set])
    elif dataset == 'fmnist':
        train_set = datasets.FashionMNIST(root='dataset/fmnist/data', train=True, download=True)
        test_set = datasets.FashionMNIST(root='dataset/fmnist/data', train=False, download=True)
        full_set = ConcatDataset([train_set, test_set])
    elif dataset == 'kuzushiji':
        train_set = datasets.KMNIST(root='dataset/kuzushiji/data', train=True, download=True)
        test_set = datasets.KMNIST(root='dataset/kuzushiji/data', train=False, download=True)
        full_set = ConcatDataset([train_set, test_set])

    # Load training data from specific partition
    data_configuration = loadmat(configuration_path)
    print('Load data partition from ' + configuration_path)
    labeled_index = data_configuration['labeled_index'][0]
    unlabeled_index = data_configuration['unlabeled_index'][0]
    test_index = data_configuration['test_index'][0]
    class_order = data_configuration['class_order'][0]
    novel_class = []
    for i in range(len(class_order)):
        if class_order[i] >= known_class_number:
            novel_class.append(i)
    labeled_data = [(transform_train(full_set[i][0]), class_order[int(full_set[i][1])]) for i in labeled_index]
    test_data = [(transform_test(full_set[i][0]), class_order[int(full_set[i][1])]) for i in test_index]
    unlabeled_data = []
    for i in unlabeled_index:
        img, _ = full_set[i]
        unlabeled_data.append((transform_train(img), known_class_number))
    return labeled_data, unlabeled_data, test_data
