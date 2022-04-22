from sklearn.svm import SVR
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import torch.utils.data as Data
from SVRR import feature_selection

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  #


def getdata() :
    x = np.loadtxt("./data/train_features.txt", delimiter=",")
    y = np.loadtxt("./data/train_labels.txt", delimiter=',')
    x_pd = np.loadtxt("./data/test_features.txt", delimiter=',')

    return x, y, x_pd


class AER(nn.Module) :

    def __init__(self, in_dims=16) :
        super(AER, self).__init__()
        self.Encoder = nn.Sequential(
                nn.Linear(in_dims, 128),
                nn.BatchNorm1d(128, momentum=0.5),
                nn.LeakyReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, 512),
                nn.BatchNorm1d(512, momentum=0.5),
                nn.LeakyReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512, momentum=0.5),
                nn.LeakyReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(512, 128),
                nn.BatchNorm1d(128, momentum=0.5),
                nn.LeakyReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, 64),
                )
        self.Encoder_pd = nn.Sequential(
                nn.Linear(64, 128),
                nn.BatchNorm1d(128, momentum=0.5),
                nn.LeakyReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, 512),
                nn.BatchNorm1d(512, momentum=0.5),
                nn.LeakyReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512, momentum=0.5),
                nn.LeakyReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(512, 128),
                nn.BatchNorm1d(128, momentum=0.5),
                nn.LeakyReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, 64),
                )

        self.Decoder = nn.Sequential(
                nn.Linear(64, 128),
                nn.BatchNorm1d(128, momentum=0.5),
                nn.LeakyReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, 512),
                nn.BatchNorm1d(512, momentum=0.5),
                nn.LeakyReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512, momentum=0.5),
                nn.LeakyReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(512, 128),
                nn.BatchNorm1d(128, momentum=0.5),
                nn.LeakyReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128, momentum=0.5),
                nn.LeakyReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, in_dims),
                )

    def forward(self, x) :
        h = self.Encoder(x)
        y = torch.sum(h, dim=1)

        hat = self.Encoder_pd(h)

        y_hat = torch.sum(hat, dim=1)
        z2 = self.Decoder(hat)

        return y_hat, y, z2

        pass


def Normalized(x_train, x_test, y) :
    pass


def Training(
        x_train, y_train, x_test, y_test, learn_rate=1e-3, weight_decay=1e-2, epochs=200, epochs_pd=50, Batch_size=128,
        model_num=1
        ) :
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("PLZ use GPU machine for not demaging your PC")
    # print(device)
    x_train = torch.from_numpy(x_train)
    x_train = x_train.to(torch.float32)

    y_train = torch.from_numpy(y_train)

    y_train = y_train.to(torch.float32)
    # print(x_train.size(), y_train.size())
    train_data = Data.TensorDataset(x_train, y_train)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=Batch_size, shuffle=True, num_workers=0)

    x_test = torch.from_numpy(x_test)
    x_test = x_test.to(torch.float32)

    Net = AER(x_train.size(1))
    Net.to(device)

    x_test = x_test.cuda()

    Optimizer = Adam(Net.parameters(), lr=learn_rate, weight_decay=weight_decay)
    loss1 = nn.MSELoss()
    loss2 = nn.L1Loss()

    try :
        Net.load_state_dict(torch.load('./model/AER_epochs=100_{}.pth'.format(model_num)))
    except :
        for epoch in range(epochs) :
            epoch_loss = 0
            for step, (train_x, train_y) in enumerate(train_loader) :
                train_x = train_x.cuda()
                train_y = train_y.cuda()

                y_hat, y_bar, z1 = Net(train_x)
                lss = loss1(y_bar, train_y[:, 0]) + loss2(
                        train_x, z1
                        ) + loss1(y_hat, train_y[:, 1])  # pred-loss + re-loss
                # lss =  loss2(
                #         train_x, z
                #         ) +   loss1(y_bar, train_y[:, 1])  # pred-loss +

                Optimizer.zero_grad()
                lss.backward()
                Optimizer.step()

                epoch_loss += lss.item()

            # for epoch_pd in range(epochs_pd) :
            #     for step, (train_x, train_y) in enumerate(train_loader):
            #
            #         train_x = train_x.cuda()
            #         train_y = train_y.cuda()
            #
            #         y_hat, _ = Net(train_x, mode='pd')
            #         lss_pd = loss1(y_hat, train_y[:, 1])  #scale-loss
            #
            #         Optimizer.zero_grad()
            #         lss_pd.backward()
            #         Optimizer.step()

            # epoch_loss += lss_pd.item()

            with torch.no_grad() :
                y_test_hat, y_test_bar, _, = Net(x_test)
                y_test_bar = y_test_bar.cpu().numpy()
                y_test_hat = y_test_hat.cpu().numpy()

                R2_scorePl = r2_score(y_test_bar, y_test[:, 0])
                R2_scorePr = r2_score(y_test_hat, y_test[:, 1])

            if epoch == (epochs - 1) :
                torch.save(Net.state_dict(), './model/AER_epochs={}_{}.pth'.format(epoch + 1, model_num))
                return R2_scorePl, R2_scorePr
            if epoch % 10 == 9 :
                print(
                        "Epoch====>{} Loss====>{} R2scorePl====>{} R2scorePr====>{}".format(
                                epoch, epoch_loss, R2_scorePl, R2_scorePr
                                )
                        )
    finally :
        print("____________________Training Done!!______________")
        with torch.no_grad() :
            y_test_hat, y_test_bar, _, = Net(x_test)
            y_test_bar = y_test_bar.cpu().numpy()
            y_test_hat = y_test_hat.cpu().numpy()

            R2_scorePl = r2_score(y_test_bar, y_test[:, 0])
            R2_scorePr = r2_score(y_test_hat, y_test[:, 1])

            return R2_scorePl, R2_scorePr


def Predict(x_pd, best_model=1) :
    model = AER(x_test.shape(1))
    model.load_state_dict(torch.load('./model/AER_epochs=100_{}.pth'.format(best_model)))
    x_pd = torch.tensor(x_pd).cuda()
    train_loader = Data.DataLoader(dataset=train_data, batch_size=Batch_size, shuffle=True, num_workers=0)

    pass


if __name__ == "__main__" :
    x, y, x_pd = getdata()

    # train 10 times, and get its Avg.
    R2Pr_list = []
    R2Pl_list = []

    for sd in range(10) :

        x_train, x_test, y_train, y_test = train_test_split(x, y[0], train_size=0.7, random_state=sd + 1)
        _, _, y_train1, y_test1 = train_test_split(x, y[1], train_size=0.7, random_state=sd + 1)
        y_train = np.vstack((y_train, y_train1)).T
        y_test = np.vstack((y_test, y_test1)).T
        R2Pr, R2Pl = Training(
            x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, epochs=100, model_num=sd + 1
            )
        R2Pl_list.append(R2Pl)
        R2Pr_list.append(R2Pr)

    num_model = np.argmax(R2Pl_list)
    print("Best E2score for Pl:{}".format(R2Pl[num_model]))
    print("Best E2score for Pr:{}".format(R2Pl[num_model]))
    Predict(x_pd, num_model.item())

    # training_model(X1, X2, model1, model2, y, x_pd)
