from sklearn.svm import SVR
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Adam, SGD
import torch.utils.data as Data

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.utils.tutorials.cnn_utils import train, evaluate




# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  #




def getdata():
    x = np.loadtxt("./data/train_features.txt", delimiter=",")
    y = np.loadtxt("./data/train_labels.txt", delimiter=',')
    x_pd = np.loadtxt("./data/test_features.txt", delimiter=',')

    return x, y, x_pd

class AER(nn.Module):

    def __init__(self, in_dims=16):
        super(AER, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Linear(in_dims, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.Decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, in_dims),
        )

    def forward(self, x):
        x = self.Encoder(x)
        y = torch.sum(x, dim=1)
        z = self.Decoder(x)

        return y, z
        pass


def Normalized(x_train, x_test, y):
    pass



def Training(x_train, y_train, x_test, y_test, learn_rate=1e-3, weight_decay=1e-2, epochs=500, Batch_size=128):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    x_train = torch.from_numpy(x_train)
    x_train = x_train.to(torch.float32)
    # x_train.cuda()
    y_train = torch.from_numpy(y_train)
    y_train = y_train.to(torch.float32)
    # y_train.cuda()

    train_data = Data.TensorDataset(x_train, y_train)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=Batch_size, shuffle=True, num_workers=4)

    x_test = torch.from_numpy(x_test)
    x_test = x_test.to(torch.float32)
    # y_test = torch.from_numpy(y_test)
    # y_test = y_test.to(torch.float32)
    # test_data = Data.TensorDataset(x_test, y_test)
    # test_loader = Data.DataLoader(dataset=test_data, batch_size=Batch_size, shuffle=True, num_workers=4)

    Net = AER(x_train.size(1))
    Net.to(device)
    # train_loader.cuda()
    x_test = x_test.cuda()

    Optimizer = Adam(Net.parameters(), lr=learn_rate, weight_decay=weight_decay)
    loss1 = nn.MSELoss()
    loss2 = nn.L1Loss()

    for epoch in range(epochs):
        epoch_loss = 0
        for step, (train_x, train_y) in enumerate(train_loader):
            train_x = train_x.cuda()
            train_y = train_y.cuda()
            y_bar, z = Net(train_x)
            lss = loss1(y_bar, train_y) + 1000 * loss2(train_x, z)
            Optimizer.zero_grad()
            lss.backward()
            Optimizer.step()

            epoch_loss += lss.item()

        with torch.no_grad():
            y_test_bar, _ = Net(x_test)
            y_test_bar = y_test_bar.cpu().numpy()
            R2_score = r2_score(y_test_bar, y_test)
        if epoch % 100 == 99:
            torch.save(Net.state_dict(), './model/AER_epoch={}.pth')
        print("Epoch====>{} Loss====>{} R2score====>{}".format(epoch, epoch_loss, R2_score))
        epoch_loss = 0


def Predict(x_test):
    pass



if __name__ == "__main__":
    x, y, x_pd = getdata()
    # X1, X2, model1, model2 = feature_selection(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y[1], test_size=0.9)
    Training(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

    # training_model(X1, X2, model1, model2, y, x_pd)




