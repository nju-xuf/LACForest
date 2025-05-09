from utils.network import VGG16, CNN, MLP
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import relu
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, accuracy_score, f1_score

epsilon = 0.001


def auc_performance(labels, predictions):
    auc = roc_auc_score(labels, predictions)
    fpr, tpr, threshold = roc_curve(labels, predictions)
    fpr95 = fpr[tpr >= 0.95][0]
    return auc, fpr95


class DeepLACForests(nn.Module):
    """
    Deep Differentiable Forests for learning with augmented classes
    """

    def __init__(self, setting):
        """
        Initialize the DeepLACForests. 

        Parameters:
        setting: A dictionary with following keys:
            - 'device': str.
                The device   
            - 'depth': int. 
                Depth of each differentiable tree.  
            - 'ensemble_size': int.
                Number of differentiable trees. 
            - 'class_num': int. 
                Number of known classes. 
            - 'arch': {'vgg16', 'wideresnet'} .
                Architecture name of the feature extractor. 
        """
        super(DeepLACForests, self).__init__()
        self.model_setting = setting
        self.device = setting['device']
        self.depth = setting['depth']
        self.ensemble_size = setting['ensemble_size']
        self.class_num = setting['class_num']
        self.arch = setting['arch']
        self.total_novel_score = None
        self.total_known_class_counter = None
        self.encode_feature_dimension = None
        self.leaf_counter = None
        self.encoder = None
        self.trees = None
        self.novel_score = None
        self.novel_threshold = 0
        self.build_encoder()
        self.build_tree()

    def build_encoder(self):
        """
        Initialize the feature extractor
        """
        if self.arch not in ['vgg16', 'cnn', 'mlp']:
            raise ValueError(
                'Architecture should be selected with in vgg16, mlp or cnn.')
        if self.arch == 'vgg16':
            self.encode_feature_dimension = 128
            self.encoder = VGG16(num_classes=self.class_num).to(self.device)
        elif self.arch == 'cnn':
            self.encode_feature_dimension = 390
            self.encoder = CNN(num_classes=self.class_num).to(self.device)
        elif self.arch == 'mlp':
            hidden_dim = 128
            self.encode_feature_dimension = hidden_dim
            self.encoder = MLP(784, hidden_dim, self.class_num).to(self.device)

    def encoder_output(self, X):
        """
        Ouput the encoded features and the predictions of auxiliary predictor

        Parameters:
        X: torch.Tensor of shape (batch_size, channels, height, width)
            The input tensor.

        Return:
        tuple:
            - self.encoder.embedding(X): torch.Tensor of shape (, )
                To be continued 
            - self.encoder(X): torch.Tensor of shape (, )
        """
        return self.encoder.embedding(X), self.encoder(X)

    def build_tree(self):
        """
        Build differentiable forests 
        """
        self.trees = []
        for _ in range(self.ensemble_size):
            self.trees.append([Node(dimension=self.encode_feature_dimension, class_num=self.class_num).to(
                self.device) for _ in range(2 ** self.depth - 1)])

    def forward(self, X, encoder_prediction=True):
        """
        Output the predictions of Deep LACForests and auxiliary predictor. 

        Parameters:
        X: torch.Tensor of shape (batch_size, channels, height, width)
            The input tensor.

        Return:
        tuple:
            - prediction: torch.Tensor of shape (batch_size, self.class_num)
                Output prediction of auxiliary predictor
            - all_results (list): Output tensor of shape 
        """
        X, prediction = self.encoder(X.to(self.device))
        all_results = []
        for j in range(len(self.trees)):
            results = torch.ones(2 ** self.depth - 1, len(X)).to(self.device)
            for i in range(2 ** (self.depth-1) - 1):
                decison = self.trees[j][i](X)
                results_temp = results.clone()
                results[i*2+1] = torch.mul(results_temp[i], decison)
                results[i*2+2] = torch.mul(results_temp[i], 1-decison)
            all_results.append(results)
        return prediction, all_results

    def fit(self, ldataloader, udataloader, theta):
        """
        Fit Deep LACForests with ldataloader and udataloader

        Parameters:
        ldataloader: torch.utils.data.DataLoader. (To be continued)
            A DataLoader providing batches of labeled data.
            Each batch is expected to be a tuple containing:
                - inputs (torch.Tensor): Tensor of input data, typically of shape (batch_size, ...).
                - labels (torch.Tensor): Tensor of labels, typically of shape (batch_size, ...).
        udataloader: torch.utils.data.DataLoader. (To be continued)
            A DataLoader providing batches of labeled data.
            Each batch is expected to be a tuple containing:
        theta: float.
            The estimated proportion of augmented classes in unlabeled data.  
        record_path: str.
            The path to record the leaf information.
        """
        n, m = 0, 0
        # clear the counter for each leaf node.
        for j in range(len(self.trees)):
            for i in range(2 ** (self.depth-1) - 1, 2 ** self.depth - 1):
                self.trees[j][i].clear_counter()
        self.eval()
        with torch.no_grad():
            for _, (X, y) in enumerate(ldataloader):
                _, all_results = self(X)
                for j in range(self.ensemble_size):
                    output = torch.matmul(
                        all_results[j], F.one_hot(y, num_classes=self.class_num).float().to(self.device))
                    # update the class distribution of each leaf node
                    for i in range(2 ** (self.depth-1) - 1, 2 ** self.depth - 1):
                        self.trees[j][i].counter_add(output[i].detach().to('cpu'))
                    n += len(X)
            for _, (X, _) in enumerate(udataloader):
                _, all_results = self(X)
                for j in range(self.ensemble_size):
                    output = torch.sum(all_results[j], dim=1)
                    # update the number of unlabeled instance for each leaf node
                    for i in range(2 ** (self.depth-1) - 1, 2 ** self.depth - 1):
                        self.trees[j][i].unlabel_counter_add(output[i].detach().to('cpu'))
                    m += len(X)
        self.total_novel_score = []
        self.total_known_class_counter = []
        unlabeled_num = []
        for j in range(self.ensemble_size):
            for i in range(2 ** (self.depth-1) - 1, 2 ** self.depth - 1):
                self.total_novel_score.append(self.trees[j][i].calculate_novel_score(n, m, theta))
                unlabeled_num.append(self.trees[j][i].get_unlabeled_counter())
                self.total_known_class_counter.append(
                    self.trees[j][i].get_counter())
        unlabeled_num = torch.tensor(unlabeled_num)
        unlabeled_num = unlabeled_num / m
        self.total_novel_score = torch.tensor(self.total_novel_score)  # (ensemble_size * leaf_number)
        self.total_known_class_counter = torch.stack(
            self.total_known_class_counter)  # (ensemble_size * leaf_number, self.class_num)

    def get_depth(self):
        return self.depth

    def get_ensemble_size(self):
        return self.ensemble_size

    def predict_augmented_score(self, data_loader):
        augmented_scores = []
        with torch.no_grad():
            for _, (X, _) in enumerate(data_loader):
                _, all_results = self(X)
                leaf_dis = torch.cat([all_results[j][2 ** (self.depth-1) - 1:, ]
                                     for j in range(self.ensemble_size)], dim=0)
                novel_score = (torch.matmul(torch.transpose(leaf_dis, 0, 1),
                                            self.total_novel_score.to(self.device))).squeeze()
                for i in range(len(novel_score)):
                    augmented_scores.append(novel_score[i].detach().to('cpu'))
        return augmented_scores

    def evaluate(self, test_loader):
        """
        Evaluate Deep LACForests with test_loader. 

        Parameters:
        test_loader: torch.utils.data.DataLoader. 
            A DataLoader providing batches of test data.
            Each batch is expected to be a tuple containing:
                - X: torch.Tensor of shape (batch_size, channels, height, width).
                - y: torch.Tensor of shape (X, X)
        record_path: str.
            The path to record the leaf information.
        leaf_info: bool.
            The flag that decides whether to record the detailed information of each leaf nodes. 
        """
        novel_prediction = []
        novel_ground_truth = []
        prediction = []
        ground_truth = []
        test_counter = torch.zeros(
            (2 ** (self.depth-1) * self.ensemble_size, self.class_num+1))
        n = 0
        with torch.no_grad():
            for _, (X, y) in enumerate(test_loader):
                n += len(X)
                _, all_results = self(X)
                leaf_dis = torch.cat([all_results[j][2 ** (self.depth-1) - 1:, ]
                                     for j in range(self.ensemble_size)], dim=0)
                novel_score = (torch.matmul(torch.transpose(leaf_dis, 0, 1), self.total_novel_score.to(self.device))).squeeze()
                novel_score = novel_score / self.ensemble_size
                current_prediction = (torch.matmul(torch.transpose(
                    leaf_dis, 0, 1), self.total_known_class_counter.to(self.device))) / self.ensemble_size
                y_clamp = torch.clamp(y, max=self.class_num)
                y_one_hot = F.one_hot(
                    y_clamp, num_classes=self.class_num+1).float().to(self.device)
                test_counter = test_counter + \
                    torch.matmul(leaf_dis, y_one_hot).detach().to('cpu')
                for i in range(len(novel_score)):
                    current_novel_prediction = novel_score[i].detach().to('cpu')
                    novel_prediction.append(current_novel_prediction)
                    novel_ground_truth.append(1 if y[i] >= self.class_num else 0)
                    ground_truth.append(y_clamp[i])
                    if current_novel_prediction > (1-current_novel_prediction) * torch.max(current_prediction[i]).detach().to('cpu'):
                        prediction.append(self.class_num)
                    else:
                        prediction.append(torch.argmax(
                            current_prediction[i]).detach().to('cpu'))
        gini_index = 0
        mse = 0
        gini_index = np.zeros(self.ensemble_size)
        leaf_per_tree = 2 ** (self.depth-1)
        # report the information of each leaf node
        for i in range(len(test_counter)):
            size = torch.sum(test_counter[i])
            weight = size / n
            novel_truth = test_counter[i][self.class_num] / size
            mse += weight * (self.total_novel_score[i] - novel_truth) ** 2
            class_dis = test_counter[i] / size

            # print the gini index of each tree 
            gini_index[i // leaf_per_tree] += weight * (1 - torch.sum(class_dis ** 2))
        auc, fpr95 = auc_performance(novel_ground_truth, novel_prediction)
        accuracy = accuracy_score(ground_truth, prediction)
        macro_f1 = f1_score(ground_truth, prediction,
                            average='macro', zero_division=0)
        return auc, accuracy, macro_f1, np.average(gini_index)
    
    # validate the effectiveness of the current model on data of known classes 
    def validate(self, val_loader, record_path, leaf_info=False):
        prediction = []
        ground_truth = []
        with torch.no_grad():
            for _, (X, y) in enumerate(val_loader):
                encoder_prediction, _ = self(X)
                for i in range(len(encoder_prediction)):
                    if y[i] < self.class_num:
                        ground_truth.append(y[i])
                        prediction.append(torch.argmax(
                            encoder_prediction[i]).detach().to('cpu'))
        accuracy = accuracy_score(ground_truth, prediction)
        print(classification_report(ground_truth, prediction, zero_division=0))
        with open(record_path, 'a') as f:
            f.write('accuracy: ' + str(accuracy) + '\n')
            f.write(classification_report(
                ground_truth, prediction, zero_division=0))
        return accuracy


class Node(nn.Module):
    """
    A single decision node for a differetiable decision tree
    """

    def __init__(self, dimension, class_num):
        """
        Initialize a decision node. 

        Parameters:
        dimension: int. 
            The feature dimension of the input data. 
        class_num: int.
            Number of known classes. 
        """
        super(Node, self).__init__()
        self.decision_function = nn.Linear(dimension, 1)
        nn.init.xavier_uniform_(self.decision_function.weight)
        self.class_num = class_num
        self.unlabel_counter = None
        self.counter = None

    def forward(self, X):
        return torch.sigmoid(self.decision_function(X)).squeeze()

    def clear_counter(self):
        self.counter = torch.zeros(self.class_num)
        self.unlabel_counter = 0

    def counter_add(self, delta):
        self.counter += delta

    def unlabel_counter_add(self, delta):
        self.unlabel_counter += delta

    def get_counter(self):
        return self.counter / torch.sum(self.counter)
    
    def get_unlabeled_counter(self):
        return self.unlabel_counter
    
    def calculate_novel_score(self, n, m, theta):
        return (self.unlabel_counter - m * theta * torch.sum(self.counter) / n)/ self.unlabel_counter