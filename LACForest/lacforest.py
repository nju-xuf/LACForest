import numpy as np
from numba import jit
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve,  f1_score, accuracy_score
from utils.MPE.prior_estimate import KernelPriorEstimator

_UNDEFINED = -1
_AVALIABLE = -2

LEFT = 0
LEFT_INTERSECT = 1
RIGHT_INTERSECT = 2
RIGHT = 3


@jit(nopython=True)
def best_split_decision_giniaug(X_l, y_l, X_u, class_number, min_n_l, min_n_u, candidate_features, 
                                total_threshold_set, total_sorted_indices, total_sorted_indices_u, class_count, novel_number):
    """
    A fast implementation of searching the best split under augmented gini-index
    """
    best_score, best_feature, best_threshold = 1, -1, 0
    best_left_gini, best_right_gini = 0, 0
    for i in range(len(candidate_features)):
        feature = candidate_features[i]
        left_count = np.zeros(class_number + 1).astype('float')
        right_count = class_count.astype('float')
        n_l, n_r, n_u_l, n_u_r = 0, len(X_l), 0, len(X_u)
        threshold_set = total_threshold_set[i]
        sorted_indices = total_sorted_indices[i]
        sorted_indices_u = total_sorted_indices_u[i]
        labeled_index, unlabeled_index = 0, 0
        for current_theshold in threshold_set:
            if current_theshold < 0:
                break
            while labeled_index < len(X_l):
                l_index = sorted_indices[labeled_index]
                if X_l[l_index][int(feature)] <= current_theshold:
                    n_l = n_l + 1
                    n_r = n_r - 1
                    left_count[y_l[l_index]] += 1
                    right_count[y_l[l_index]] -= 1
                    labeled_index += 1
                else:
                    break
            while unlabeled_index < len(X_u):
                u_index = sorted_indices_u[unlabeled_index]
                if X_u[u_index][int(feature)] <= current_theshold:
                    n_u_l = n_u_l + 1
                    n_u_r = n_u_r - 1
                    unlabeled_index += 1
                else:
                    break
            if (n_u_l == 0 and n_u_r == 0) or (n_u_l == 0 and n_l == 0) or (n_u_r == 0 and n_r == 0):
                continue
            if n_u_l < min_n_u or n_u_r < min_n_u or n_l < min_n_l or n_r < min_n_l:
                continue
            left_novel_number = max(
                0, n_u_l - (n_u_l + n_u_r - novel_number) * n_l / (n_l + n_r))
            right_novel_number = max(
                0, n_u_r - (n_u_l + n_u_r - novel_number) * n_r / (n_l + n_r))
            if n_u_l == 0:
                left_novel_rate = 0
            else:
                left_novel_rate = left_novel_number / n_u_l
            if n_u_r == 0:
                right_novel_rate = 0
            else:
                right_novel_rate = right_novel_number / n_u_r

            left_count_temp, right_count_temp = np.asarray(left_count).astype(
                'float'), np.asarray(right_count).astype('float')
            left_sum, right_sum = np.sum(
                left_count_temp), np.sum(right_count_temp)
            if left_sum == 0:
                left_gini_index = 1
            else:
                left_gini_index = 1 - \
                    np.sum(((1-left_novel_rate) * left_count_temp /
                           left_sum) ** 2) - left_novel_rate ** 2
            if right_sum == 0:
                right_gini_index = 1
            else:
                right_gini_index = 1 - \
                    np.sum(((1-right_novel_rate) * right_count_temp /
                           right_sum) ** 2) - right_novel_rate ** 2
            score = (n_u_l * left_gini_index + n_u_r *
                     right_gini_index) / (n_u_l + n_u_r)
            if score < best_score:
                best_score = score
                best_feature = feature
                best_threshold = current_theshold
                best_left_gini = left_gini_index
                best_right_gini = right_gini_index
    return best_feature, best_threshold, best_score, best_left_gini, best_right_gini


@jit(nopython=True)
def best_split_decision_gini(X, y, w, class_number, candidate_features, total_threshold_set, total_sorted_indices, class_count):
    """
    A fast implementation of searching the best split under gini-index
    """
    best_score, best_feature, best_threshold = 1, 0, 0
    best_left_gini, best_right_gini = 0, 0
    for i in range(len(candidate_features)):
        feature = candidate_features[i]
        left_count = np.zeros(class_number + 1).astype('float')
        right_count = class_count.astype('float')
        threshold_set = total_threshold_set[i]
        sorted_indices = total_sorted_indices[i]
        labeled_index = 0
        for current_theshold in threshold_set:
            if current_theshold < 0:
                break
            while labeled_index < len(X):
                l_index = sorted_indices[labeled_index]
                if X[l_index][int(feature)] <= current_theshold:
                    left_count[y[l_index]] += w[l_index]
                    right_count[y[l_index]] = max(
                        0, right_count[y[l_index]] - w[l_index])
                    labeled_index += 1
                else:
                    break
            left_count_temp, right_count_temp = np.asarray(left_count).astype(
                'float'), np.asarray(right_count).astype('float')
            left_sum, right_sum = np.sum(
                left_count_temp), np.sum(right_count_temp)
            if left_sum == 0:
                left_gini_index = 1
            else:
                left_gini_index = 1 - \
                    np.sum((left_count_temp / left_sum) ** 2)
            if right_sum == 0:
                right_gini_index = 1
            else:
                right_gini_index = 1 - \
                    np.sum((right_count_temp / right_sum) ** 2)
            if left_sum + right_sum == 0:
                score = 0
            else:
                score = (left_sum * left_gini_index + right_sum *
                         right_gini_index) / (left_sum + right_sum)
            if score < best_score:
                best_score = score
                best_feature = feature
                best_threshold = current_theshold
                best_left_gini = left_gini_index
                best_right_gini = right_gini_index
    return best_feature, best_threshold, best_score, best_left_gini, best_right_gini


def transfer_novel_label(y, known_class_number):
    return (np.asarray(y) >= known_class_number).astype('int')


def evaluate_novel_detection(labels, predictions):
    auc = roc_auc_score(labels, predictions)
    fpr, tpr, threshold = roc_curve(labels, predictions)
    fpr95 = fpr[tpr >= 0.95][0]
    return auc, fpr95


def aug_gini_index(y_l, class_number, novel_rate):
    known_dis = np.bincount(y_l, minlength=class_number)
    known_dis = known_dis / np.sum(known_dis) * (1-novel_rate)
    return 1 - np.sum(known_dis ** 2) - novel_rate ** 2


def gini_index(y_l, class_number):
    known_dis = np.bincount(y_l, minlength=class_number)
    known_dis = known_dis / np.sum(known_dis)
    return 1 - np.sum(known_dis ** 2)

class LACForest():
    def __init__(self, class_number, ensemble_size, max_features, gamma):
        self.class_number = class_number
        self.ensemble_size = ensemble_size
        self.max_features = max_features
        self.n_categories_ = None
        self.trees = []
        self.novel_threshold = None
        self.gamma = gamma
        self.theta = None
        for _ in range(ensemble_size):
            self.trees.append(LACTree(class_number=class_number, gamma=gamma, max_features=self.max_features))

    def fit(self, X_l, y_l, X_u):
        print('Start estimating theta. This may take a long time.')
        mpe_helper = KernelPriorEstimator()
        self.theta = 1 - mpe_helper.estimate(np.asarray(X_l), np.asarray(X_u))
        print('Estimated theta = ' + str(self.theta))
        print('Build the forest:')
        novel_number = len(X_u) * self.theta
        for i in range(self.ensemble_size):
            self.trees[i].initialize(X_l=X_l, y_l=y_l, X_u=X_u, theta=self.theta)
        print('Step 1: Construct shallow forest')
        for i in tqdm(range(self.ensemble_size)):
            self.trees[i].expand()
        print('Stage 2: Refine forest with pseudo-labeled augmented instances')
        augmented_score = self.predict_novel_score(X_u)
        sorted_index = np.argsort(-augmented_score)
        selections = sorted_index[:int(novel_number)]
        augmented_label = np.zeros(len(X_u))
        for i in selections:
            augmented_label[i] = 1
        for i in tqdm(range(self.ensemble_size)):
            self.trees[i].refine(augmented_label)

    def evaluate(self, X_test, y_test):
        novel_score = self.predict_novel_score(X_test)
        novel_label = transfer_novel_label(y_test, self.class_number)
        auc_score, _ = evaluate_novel_detection(novel_label, novel_score)
        y_pred_label = self.predict(X_test)
        y_test = np.clip(y_test, 0, self.class_number)
        macro_f1 = f1_score(y_pred=y_pred_label,
                            y_true=y_test, average='macro', zero_division=0)
        accuracy = accuracy_score(y_pred=y_pred_label, y_true=y_test)
        return auc_score, macro_f1, accuracy

    def predict_novel_score(self, X):
        results = []
        for i in range(self.ensemble_size):
            results.append(self.trees[i].predict_novel_score(X))
        return np.average(results, axis=0)

    def predict(self, X, prob=False):
        temp = []
        for i in range(self.ensemble_size):
            temp.append(self.trees[i].predict(X))
        temp = np.average(temp, axis=0)
        prediction = []
        for i in range(len(temp)):
            prediction.append(np.argmax(temp[i]))
        return prediction


class LACTree():
    def __init__(
        self,
        class_number,
        min_samples_leaf=1,
        max_features="sqrt",
        random_seed=None,
        gamma=0.1,
        criterion='giniaug',
        if_record_loss=False
    ):
        self.class_number = class_number
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.if_record_loss = if_record_loss
        self.gamma = gamma
        self.n_categories_ = None
        self.criterion = criterion
        self.random_seed = random_seed
        self.root_ = None
        self.min_samples_leaf_labeled = None
        self.min_samples_leaf_unlabeled = None
        self.n_features_ = None
        self.novel_number = None
        self.theta = None

    def initialize(self, X_l, y_l, X_u, theta):
        """
        initialize the LACTree predictor
        """
        self.theta = theta
        self.n_features_ = X_l.shape[1]
        if self.max_features == "sqrt":
            self.max_features_ = int(np.sqrt(self.n_features_))
        elif self.max_features == "log2":
            self.max_features_ = int(np.log2(self.n_features_))
        elif self.max_features is None:
            self.max_features_ = self.n_features_
        else:
            self.max_features_ = self.max_features
        self.root_ = Node(X_l=X_l, y_l=y_l, X_u=X_u, class_number=self.class_number, labeled_index=[i for i in range(len(X_l))], 
                          unlabeled_index=[i for i in range(len(X_u))], theta=theta, depth=0, gamma=self.gamma)

    def expand(self):
        self.root_.expand(max_features=self.max_features_)

    def refine(self, augmented_label):
        self.root_.refine(max_features=self.max_features_, augmented_label=augmented_label)

    def test(self, X, y):
        self.root_.test_init()
        for i in range(len(X)):
            self.root_.test(X[i], y[i])

    def print_leaf_info(self):
        self.root_.print_leaf_info()

    def predict(self, X):
        results = []
        for sample in X:
            results.append(self.root_.predict(sample))
        return np.asarray(results)

    def predict_novel_score(self, X):
        results = []
        for sample in X:
            results.append(self.root_.predict_augmented_score(sample))
        return np.asarray(results)

    def update_novel_rate(self, p_aug_set):
        self.root_.update_novel_rate(p_aug_set)


class Node:
    def __init__(self, X_l, y_l, X_u, class_number, labeled_index, unlabeled_index, theta, depth, gamma):
        self.X_l, self.y_l, self.X_u = X_l, y_l, X_u
        self.theta = theta
        self.depth = depth
        self.labeled_index = labeled_index
        self.unlabeled_index = unlabeled_index
        self.n_features_ = X_l.shape[1]
        self.class_number = class_number
        self.gamma = gamma
        self.feature = _UNDEFINED
        self.threshold = _UNDEFINED
        self.left_child = _UNDEFINED
        self.right_child = _UNDEFINED
        self.status = _AVALIABLE
        self.refine_flag = False
        self.test_dis = None
        self.ori_labels = np.bincount(
            [self.y_l[i] for i in self.labeled_index], minlength=self.class_number)
        self.novel_number = max(
            0, len(unlabeled_index) - (1-theta) * len(X_u) * len(labeled_index) / len(X_l))
        if len(unlabeled_index) == 0:
            self.novel_rate = 0
        else:
            self.novel_rate = self.novel_number / len(unlabeled_index)
        if np.sum(self.ori_labels) > 0:
            self.labels = self.ori_labels / np.sum(self.ori_labels)
        else:
            self.labels = self.ori_labels
        self.final_labels = None

    def predict_augmented_score(self, input_instance):
        if self.feature == _UNDEFINED:
            return self.novel_rate
        comparison = input_instance[self.feature] <= self.threshold
        if comparison:
            return self.left_child.predict_augmented_score(input_instance)
        else:
            return self.right_child.predict_augmented_score(input_instance)

    def predict(self, sample):
        if self.feature == _UNDEFINED:
            return np.concatenate((self.labels * (1 - self.novel_rate), np.array([self.novel_rate])))
        comparison = sample[self.feature] <= self.threshold
        if comparison:
            return self.left_child.predict(sample)
        else:
            return self.right_child.predict(sample)

    def expand(self, max_features):
        if self.feature == _UNDEFINED:
            # terminate this process if certain condition is satisfied
            if len(self.labeled_index) < 2 * self.gamma * len(self.X_l) or len(self.unlabeled_index) < 2 * self.gamma * len(self.X_u):
                return
            # expand current node
            X_l, y_l, X_u = np.asarray([self.X_l[i] for i in self.labeled_index]), np.asarray(
                [self.y_l[i] for i in self.labeled_index]), np.asarray([self.X_u[i] for i in self.unlabeled_index])
            candidate_features = np.random.permutation(self.n_features_)[
                :max_features]
            total_threshold_set, total_sorted_indices, total_sorted_indices_u = [], [], []
            for feature in candidate_features:
                threshold_set = np.unique(np.concatenate(
                    (X_l[:, int(feature)], X_u[:, int(feature)])))
                sorted_indices = np.argsort(X_l[:, int(feature)])
                sorted_indices_u = np.argsort(X_u[:, int(feature)])
                total_sorted_indices_u.append(sorted_indices_u)
                total_sorted_indices.append(sorted_indices)
                total_threshold_set.append(threshold_set)
            max_length = max(len(arr) for arr in total_threshold_set)
            result = - np.ones((len(total_threshold_set), max_length))
            for i, arr in enumerate(total_threshold_set):
                result[i, :len(arr)] = arr
            class_count = np.bincount(y_l, minlength=self.class_number + 1)
            min_n_l, min_n_u = self.gamma * \
                len(self.X_l), self.gamma * len(self.X_u)
            feature, threshold, _, _, _ = best_split_decision_giniaug(X_l=X_l, X_u=X_u, y_l=y_l, class_number=self.class_number,
                                                                    min_n_l=min_n_l, min_n_u=min_n_u, candidate_features=candidate_features, total_sorted_indices=np.asarray(total_sorted_indices), total_sorted_indices_u=np.asarray(total_sorted_indices_u), class_count=class_count, total_threshold_set=result, novel_number=self.novel_number)
            if feature >= 0:
                left_labeled_index, right_labeled_index, left_unlabeled_index, right_unlabeled_index = self.__split_left_right(
                    X_l, X_u, feature, threshold)
                left_child = Node(X_l=self.X_l, y_l=self.y_l, X_u=self.X_u, class_number=self.class_number, theta=self.theta, depth=self.depth + 1, labeled_index=[
                    self.labeled_index[i] for i in left_labeled_index], unlabeled_index=[self.unlabeled_index[i] for i in left_unlabeled_index], gamma=self.gamma)
                right_child = Node(X_l=self.X_l, y_l=self.y_l, X_u=self.X_u, class_number=self.class_number, theta=self.theta, depth=self.depth + 1, labeled_index=[
                    self.labeled_index[i] for i in right_labeled_index], unlabeled_index=[self.unlabeled_index[i] for i in right_unlabeled_index], gamma=self.gamma)
                # update the information of current node
                self.feature, self.threshold, self.left_child, self.right_child = feature, threshold, left_child, right_child
        if self.left_child == _UNDEFINED:
            return
        self.left_child.expand(max_features=max_features)
        self.right_child.expand(max_features=max_features)

    def refine(self, max_features, augmented_label):
        if self.feature == _UNDEFINED:
            # update the information of current node
            self.update_final_labels(augmented_label)
            if self.check_terminate_condition(augmented_label):
                return
            X_l, y_l, X_u, y_u = [self.X_l[i] for i in self.labeled_index], [self.y_l[i] for i in self.labeled_index], [
                self.X_u[i] for i in self.unlabeled_index], [self.class_number for _ in range(len(self.unlabeled_index))]
            X_temp, y_temp = X_l + X_u, y_l + y_u
            w = [1 for _ in range(len(X_l))] + [augmented_label[i]
                                                for i in self.unlabeled_index]
            X_temp, y_temp, w = np.asarray(
                X_temp), np.asarray(y_temp), np.asarray(w)
            candidate_features = np.random.permutation(self.n_features_)[
                :max_features]
            total_threshold_set, total_sorted_indices = [], []
            for feature in candidate_features:
                threshold_set = np.unique(X_temp[:, feature])
                sorted_indices = np.argsort(X_temp[:, int(feature)])
                total_sorted_indices.append(sorted_indices)
                total_threshold_set.append(threshold_set)
            max_length = max(len(arr) for arr in total_threshold_set)
            result = - np.ones((len(total_threshold_set), max_length))
            for i, arr in enumerate(total_threshold_set):
                result[i, :len(arr)] = arr
            class_count = np.zeros(self.class_number + 1)
            for i in range(len(X_temp)):
                class_count[int(y_temp[i])] += w[i]
            feature, threshold, _, _, _ = best_split_decision_gini(X=X_temp, y=y_temp, w=w, class_number=self.class_number,
                                                                    candidate_features=candidate_features, total_sorted_indices=np.asarray(total_sorted_indices), total_threshold_set=result, class_count=class_count)
            # split the node and update
            left_labeled_index, right_labeled_index, left_unlabeled_index, right_unlabeled_index = self.__split_left_right(
                X_l, X_u, feature, threshold)
            left_child = Node(X_l=self.X_l, y_l=self.y_l, X_u=self.X_u, class_number=self.class_number, theta=self.theta, depth=self.depth + 1, labeled_index=[
                self.labeled_index[i] for i in left_labeled_index], unlabeled_index=[self.unlabeled_index[i] for i in left_unlabeled_index], gamma=self.gamma)
            right_child = Node(X_l=self.X_l, y_l=self.y_l, X_u=self.X_u, class_number=self.class_number, theta=self.theta, depth=self.depth + 1, labeled_index=[
                self.labeled_index[i] for i in right_labeled_index], unlabeled_index=[self.unlabeled_index[i] for i in right_unlabeled_index], gamma=self.gamma)
            self.feature, self.threshold, self.left_child, self.right_child = feature, threshold, left_child, right_child
        if self.status == _AVALIABLE:
            self.left_child.refine(max_features, augmented_label)
            self.right_child.refine(max_features, augmented_label)

    def __split_left_right(self, X_l, X_u, feature, threshold):
        left_labeled_index, right_labeled_index, left_unlabeled_index, right_unlabeled_index = [], [], [], []
        for i in range(len(X_l)):
            if X_l[i][feature] <= threshold:
                left_labeled_index.append(i)
            else:
                right_labeled_index.append(i)
        for i in range(len(X_u)):
            if X_u[i][feature] <= threshold:
                left_unlabeled_index.append(i)
            else:
                right_unlabeled_index.append(i)
        return left_labeled_index, right_labeled_index, left_unlabeled_index, right_unlabeled_index

    def update_final_labels(self, augmented_prob):
        self.refine_flag = True
        if len(self.unlabeled_index) != 0:
            self.novel_rate = np.sum(np.asarray(
                [augmented_prob[i] for i in self.unlabeled_index])) / len(self.unlabeled_index)
        else:
            self.novel_rate = 0
        self.final_labels = np.bincount(
            [self.y_l[i] for i in self.labeled_index], minlength=self.class_number+1)
        self.final_labels[-1] = np.sum(np.asarray([augmented_prob[i]
                                       for i in self.unlabeled_index]))

    def check_terminate_condition(self, augmented_prob):
        if len(self.labeled_index) == 0 or (np.count_nonzero(self.labels) == 1 and np.sum([augmented_prob[i] for i in self.unlabeled_index]) == 0) or (len(self.labeled_index) + len(self.unlabeled_index) <= 10):
            return True
        else:
            return False

    def test_init(self):
        if self.feature == _UNDEFINED:
            self.test_dis = np.zeros(self.class_number + 1)
            return
        self.left_child.test_init()
        self.right_child.test_init()

    def test(self, x, y):
        if self.feature == _UNDEFINED:
            self.test_dis[int(y)] += 1
            return
        comparison = x[self.feature] <= self.threshold
        if comparison:
            return self.left_child.test(x, y)
        else:
            return self.right_child.test(x, y)