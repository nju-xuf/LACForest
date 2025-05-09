from utils.io import read_from_mat, read_from_hdf
from lacforest import LACForest
import argparse
import numpy as np
import os
import scipy


def load_data_configuration(data, label, configuration_path):
    index = scipy.io.loadmat(configuration_path)
    class_order = index['class_order'][0]
    test_X = np.asarray([data[i] for i in index['test']][0])
    test_y = np.asarray([class_order[int(label[i])] for i in index['test'][0]]).astype(int)
    labeled_X = np.asarray([data[i] for i in index['labeled']][0])
    labeled_y = np.asarray([class_order[int(label[i])] for i in index['labeled'][0]]).astype(int)
    unlabeled_X = np.asarray([data[i] for i in index['unlabeled']][0])
    return labeled_X, labeled_y, unlabeled_X, test_X, test_y, np.max(labeled_y) + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='segment', type=str,
                        required=False, help="Select dataset")
    parser.add_argument("--treenumber", default='100', type=int,
                        required=False, help="Number of trees")
    parser.add_argument("--depth", default='10', type=int,
                        required=False, help="Depths of trees")
    parser.add_argument("--gamma",  default=0.01, type=float,
                        help='Hyper-parameter gamma')

   # load raw data
    args = parser.parse_args()
    dataset = args.dataset
    print('Dataset = ' + dataset)
    file_path = os.path.join('dataset', dataset, dataset + '.mat')
    try:
        raw_data = read_from_hdf(file_path, ['X', 'y'])
    except OSError:
        raw_data = read_from_mat(file_path, ['X', 'y'])
    instance = np.asarray(raw_data['X'])
    label = np.asarray(raw_data['y']).flatten()
    label -= np.min(label)

    # load data configuration
    configuration_path = os.path.join('dataset', dataset, 'data_configuration_1.mat')
    print('configuration_path = ' + configuration_path)
    X_l, y_l, X_u, X_test, y_test, known_class_number = load_data_configuration(instance, label, configuration_path)
    
    # run our LACForest approach
    model = LACForest(class_number=known_class_number, gamma=args.gamma, ensemble_size=args.treenumber, max_features="sqrt")
    model.fit(X_l, y_l, X_u)
    auc_score, macro_f1, accuracy = model.evaluate(X_test, y_test)
    print('Accuracy =' + str(accuracy))
    print('Macro F1 score =' + str(macro_f1))
    print('AUC =' + str(auc_score))
    



if __name__ == "__main__":
    main()
