from utils.time import ctime
from torch.nn import CrossEntropyLoss
from torch import relu
import torch.nn.functional as F
import torch
import sys
# smooth parameter
epsilon = 0.001

def gini_loss(pl, y, pu, theta):
    m = pu.shape[1]
    n = pl.shape[1]
    pl_sum = torch.sum(pl, dim=1)
    pu_sum = torch.sum(pu, dim=1)
    p_sum = pu_sum / torch.sum(pu_sum)
    p_known_dis = F.normalize(torch.mm(pl, y.float()), p=1, dim=1)
    p_new = (pu_sum - m * theta * pl_sum / n) / (pu_sum + epsilon)
    p_new = relu(p_new)
    p_known = 1 - p_new
    p_known_dis = p_known_dis * p_known[:, None]
    p_complete_dis = torch.cat((p_known_dis, p_new.unsqueeze(1)), dim=1)
    gini_loss = torch.sum(p_sum * (1 - torch.sum(p_complete_dis ** 2, dim=1)))
    return gini_loss



def train_deep_lac_forests(model, ldataloader, udataloader, testloader, setting):
    """
    Train deep LACForests with labeled data and unlabeled data. 

    Parameters:
    model: DeepLACForests
        The deep LACForests to be trained. 
    ldataloader: torch.utils.data.DataLoader. (To be continued)
        A DataLoader providing batches of labeled data.
        Each batch is expected to be a tuple containing:
            - inputs (torch.Tensor): Tensor of input data, typically of shape (batch_size, ...).
            - labels (torch.Tensor): Tensor of labels, typically of shape (batch_size, ...).
    udataloader: torch.utils.data.DataLoader. (To be continued)
        A DataLoader providing batches of labeled data.
        Each batch is expected to be a tuple containing:
    testloader: torch.utils.data.DataLoader.
        A Dataloader providing batches of test data. 
    valloader: torch.utils.data.DataLoader.
        A Dataloader providing batches of labeled data. 
    setting: A dictionary with training settigns
    """

    optimizer, scheduler, max_epoch, steps = setting['optimizer'], setting[
        'scheduler'], setting['max_epoch'], setting['steps']
    device, class_num  = setting['device'], setting['class_num']
    lambda_ce = setting['lambda_ce']
    ce = CrossEntropyLoss().to(device)
    theta = setting['theta']
    labeled_iter = iter(ldataloader)
    unlabeled_iter = iter(udataloader)
    model.eval()
    model.fit(ldataloader, udataloader, theta)
    model.evaluate(testloader)
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(max_epoch):
            print("{} Start Epoch ".format(ctime()) +
                  str(epoch + 1) + '/' + str(max_epoch))
            print('Learning rate=' + str(optimizer.param_groups[0]['lr']))
            model.train()
            for batch_idx in range(steps):
                try:
                    X_l, y = next(labeled_iter)
                except BaseException:
                    labeled_iter = iter(ldataloader)
                    X_l, y = next(labeled_iter)
                try:
                    X_u, _ = next(unlabeled_iter)
                except BaseException:
                    unlabeled_iter = iter(udataloader)
                    X_u, _ = next(unlabeled_iter)
                # concatenate the input instances
                l_size, u_size = len(X_l), len(X_u)
                input = torch.cat([X_l, X_u], dim=0).to(device)
                depth, ensemble_size = model.get_depth(), model.get_ensemble_size()
                logits, all_results = model(input)
                p = torch.cat([all_results[j][2 ** (depth-1) - 1:, ]
                               for j in range(ensemble_size)], dim=0)
                logits_l = logits[:l_size]
                y = y.to(device)
                lce = ce(logits_l, y)
                p_l, p_u = p[:, :l_size], p[:, l_size:]
                lgn = gini_loss(p_l, F.one_hot(y, class_num).to(device), p_u, theta)
                loss = lambda_ce * lce  + lgn
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sys.stdout.write('\rEpoch {} | Batch {} | Loss {:.4f} | Lce {:.4f} | Lgn {:.4f} '.format(
                    epoch + 1, batch_idx, loss, lce, lgn))
                sys.stdout.flush()
            scheduler.step()
            print('\nStart evaluation on test data:')
            model.eval()
            model.fit(ldataloader, udataloader, theta)
            auc, accuracy, f1_score, gini = model.evaluate(testloader)
            print('Accuracy =' + str(accuracy))
            print('Macro F1 score =' + str(f1_score))
            print('AUC =' + str(auc))
        
    return model
