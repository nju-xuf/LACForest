from utils.dataset import load_dataset, load_transform
from utils.MPE.prior_estimate import KernelPriorEstimator
from utils.trainer import train_deep_lac_forests
from utils.time import ctime
from torch.optim import SGD, lr_scheduler
from deeplacforest import DeepLACForests
from torch.utils.data import DataLoader
from datetime import datetime
import numpy as np
import argparse
import torch
import time
import os


def main():
    # set the argument-parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='svhn',
                        type=str, required=False, help="Select dataset")
    parser.add_argument("--lr", default=1e-2, type=float,
                        required=False, help="Select learning rate")
    parser.add_argument("--weight_decay", default=0.0005, type=float,
                        required=False, help="Select weight decay parameter")
    parser.add_argument("--epoch_num", default=500, type=int,
                        required=False, help="Number of epochs, default is 500")
    parser.add_argument("--ensemble_size", default=3, type=int,
                        required=False, help="Number of trees")
    parser.add_argument("--depth", default=6, type=int,
                        required=False, help="Depths of trees")
    parser.add_argument("--device", default='cuda', type=str,
                        required=False, help="Default is cuda")
    parser.add_argument("--lbatch_size", default=512, type=int,
                        required=False, help="Batch size of labeled data, default is 512")
    parser.add_argument("--ubatch_size", default=512,
                        type=int, required=False, help="Batch size of labeled data, default is 512")
    parser.add_argument("--known_class_number", default=6, type=int,
                        required=False, help="Number of known classes")
    parser.add_argument("--lambda_ce", default=1, type=float,
                        required=False, help="Lambda ce")

    # load data
    args = parser.parse_args()
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is not available.')

    # load the training and test sets
    dataset = args.dataset
    print("{} Start Loading data ".format(ctime()) + "from " + dataset + "...")
    if dataset not in ['cifar10', 'svhn', 'mnist', 'fmnist', 'kuzushiji']:
        raise ValueError(
            'Dataset must be selected within cifar10, cifar100, svhn, mnist and fmnist .')
    configuration_path = os.path.join('dataset', dataset, 'data_configuration_1.mat')
    transform_train = load_transform(dataset=args.dataset, train=True)
    transform_test = load_transform(dataset=args.dataset, train=False)
    data_setting = {
        'known_class_number': args.known_class_number,
        'configuration_path': configuration_path,
        'transform_test': transform_test,
        'transform_train': transform_train
    }

    # load the data
    labeled_dataset, unlabeled_dataset, test_dataset = load_dataset(dataset=dataset, setting=data_setting)
    labeled_dataloader = DataLoader(dataset=labeled_dataset, batch_size=args.lbatch_size, shuffle=True, drop_last=True)
    unlabeled_dataloader = DataLoader(dataset=unlabeled_dataset, batch_size=args.ubatch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.lbatch_size)

    # estimate theta by KME-based method
    print("{} Start Estimating Theta " .format(ctime()))
    repeat_times = 10
    theta_set = []
    for j in range(repeat_times):
        mpe_helper = KernelPriorEstimator()
        X_l, X_u = [], []
        for i in range(len(labeled_dataset)):
            X_l.append(np.asarray(labeled_dataset[i][0][0].view(-1)))
        for i in range(len(unlabeled_dataset)):
            X_u.append(np.asarray(unlabeled_dataset[i][0][0].view(-1)))
        X_l, X_u = np.asarray(X_l), np.asarray(X_u)
        X_l = np.asarray(X_l[np.random.choice(len(X_l), 1000, replace=False)])
        X_u = np.asarray(X_u[np.random.choice(len(X_u), 1000, replace=False)])
        theta_i = mpe_helper.estimate(np.asarray(X_l), np.asarray(X_u))
        print('Estimated theta ' + str(j) + '=' + str(theta_i))
        theta_set.append(theta_i)
    theta = np.average(theta_set)
    print('Estiamted theta = ' + str(theta))

    # Initializing model
    if args.dataset in ['cifar10', 'svhn']:
        model_arch = 'vgg16'
    elif args.dataset in ['mnist', 'fmnist', 'kuzushiji']:
        model_arch = 'cnn'
    print("{} Start building DeepLACForests " .format(ctime()))
    print('Encoder architecture: ' + model_arch + '  Depth of tree: ' + str(args.depth))
    model_setting = {'device': device,
                     'depth': args.depth,
                     'ensemble_size': args.ensemble_size,
                     'class_num': args.known_class_number,
                     'arch': model_arch
                     }
    model = DeepLACForests(setting=model_setting)

    # training model
    print("{} Start training " .format(ctime()))
    optimizer = SGD(model.parameters(), momentum=0.9,
                    lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epoch_num, eta_min=args.lr/10)
    training_setting = {'optimizer': optimizer,
                        'scheduler': scheduler,
                        'max_epoch': args.epoch_num,
                        'device': args.device,
                        'class_num': args.known_class_number,
                        'lambda_ce': args.lambda_ce,
                        'steps': max(int(len(labeled_dataset)/args.lbatch_size), int(len(unlabeled_dataset)/args.ubatch_size)),
                        'theta': theta
                        }
    train_deep_lac_forests(model, labeled_dataloader, unlabeled_dataloader, test_loader, training_setting)


if __name__ == "__main__":
    main()
