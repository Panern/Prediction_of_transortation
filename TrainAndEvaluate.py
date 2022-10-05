from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Lossses import sce_loss, Regloss, BarlowTwins_loss
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt
from Models import XGBOOST

def eval_func(y1, y2) :
    # N = len(y1)
    r2 = r2_score(y1, y2)
    MAE = mean_absolute_error(y1, y2)
    MSE = mean_squared_error(y1, y2)
    RMSE = sqrt(MSE)

    return r2, MAE, RMSE


'''
    This is for training and evaluation function for DL.
'''

def train(
        net: torch.nn.Module,
        train_loader: DataLoader,
        parameters: Dict[str, float],
        dtype: torch.dtype,
        device: torch.device,
        ) -> nn.Module :
    """
    Train CNN on provided data set.

    Args:
        net: initialized neural network
        train_loader: DataLoader containing training set
        parameters: dictionary containing parameters to be passed to the optimizer.
            - lr: default (0.001)
            - momentum: default (0.0)
            - weight_decay: default (0.0)
            - num_epochs: default (1)
        dtype: torch dtype
        device: torch device
    Returns:
        nn.Module: trained CNN.
    """
    # Initialize network
    net.to(dtype=dtype, device=device)  # pyre-ignore [28]
    net.train()
    # Define loss and optimizer
    criterion = nn.NLLLoss(reduction="sum")
    optimizer = optim.SGD(
            net.parameters(),
            lr=parameters.get("lr", 0.001),
            momentum=parameters.get("momentum", 0.0),
            weight_decay=parameters.get("weight_decay", 0.0),
            )
    scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(parameters.get("step_size", 10)),
            gamma=parameters.get("gamma", 1.0),  # default is no learning rate decay
            )

    num_epochs = parameters.get("num_epochs", 1)

    # Train Network
    for epochs in range(num_epochs) :
        H1 = []
        H2 = []
        Y_pl = []
        Y_pr = []
        for step, (train_x, train_y) in enumerate(train_loader) :
            train_x = train_x.cuda()
            train_y = train_y.cuda()

            optimizer.zero_grad()

            h1, h2, x_bar1, x_bar2, y_pl, y_pr = net(train_x)
            # if step == 0 :
            #     print(y_hat, train_y[:, 0])
            lss_reg_pl = Regloss(y_pl, train_y[:, 0])
            lss_reg_pr = Regloss(y_pr, train_y[:, 1])

            lss_rec1 = sce_loss(x_bar1, train_x)
            lss_rec2 = sce_loss(x_bar2, train_x)

            lss_bl = BarlowTwins_loss(h1, h2, bn_size=5)
            lss = lss_bl + (lss_rec1 + lss_rec2) + (lss_reg_pl + lss_reg_pr)

            lss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad() :
                H1.extend(h1.cpu().numpy().tolist())
                H2.extend(h2.cpu().numpy().tolist())
                Y_pl.extend(train_y[:, 0].cpu().numpy().tolist())
                Y_pr.extend(train_y[:, 1].cpu().numpy().tolist())

        with torch.no_grad() :
            XG1, T1 = XGBOOST(H1, Y_pl)
            XG2, T2 = XGBOOST(H2, Y_pr)

    return net, XG1, XG2


def evaluate(
        net: nn.Module, data_loader: DataLoader, XG1, XG2
        ) -> float :
    """
    Compute classification accuracy on provided dataset.

    Args:
        net: trained model
        data_loader: DataLoader containing the evaluation set
        dtype: torch dtype
        device: torch device
    Returns:
        float: classification accuracy
    """
    net.eval()

    with torch.no_grad() :
        # for inputs, labels in data_loader :
        #     # move data to proper dtype and device
        #     inputs = inputs.to(dtype=dtype, device=device)
        #     labels = labels.to(device=device)
        #     outputs = net(inputs)
        #     _, predicted = torch.max(outputs.data, 1)
        #     total += labels.size(0)
        #     correct += (predicted == labels).sum().item()
        for x_test, y_test in data_loader :
            x_test = x_test.cuda()
            h1, h2, _, _, _, _ = net(x_test)

            y_test_bar = XG1.predict(h1.cpu().numpy())
            y_test_hat = XG2.predict(h2.cpu().numpy())

            R2_pl, MAE_pl, RMSE_pl = eval_func(y_test_bar, y_test[:, 0])
            R2_pr, MAE_pr, RMSE_pr = eval_func(y_test_hat, y_test[:, 1])
            # print("R2_pr: ", R2_pr, "MAE_pr: ", MAE_pr, "RMSE_pr: ", RMSE_pr)
    # ac = correct / total

    return R2_pr #, MAE_pl, RMSE_pl, R2_pr, MAE_pr, RMSE_pr




