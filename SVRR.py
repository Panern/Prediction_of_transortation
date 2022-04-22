from sklearn.svm import SVR
import numpy as np
from  sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Adam, SGD
import torch.utils.data as Data

#解决中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False #

x = np.loadtxt("./data/train_features_new.txt", delimiter=",")
y = np.loadtxt("./data/train_labels_new.txt", delimiter=',')
x_pd = np.loadtxt("./data/test_features.txt", delimiter=',')
feature_name = np.load("./data/features_name_new.npy")
C_list = [0.1, 1, 10, 100, 1000, 1e4, 1e5]
Kernel = ["poly", "rbf", "sigmoid"]
Train_rate = [0.6, 0.7, 0.8, 0.9]
Num_look = 20


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
            nn.Linear(128,512),
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

def training_model(x1, x2, model1, model2, y, x_test):
    for kl in Kernel:
        print("___________________________________________________________________________________________________")
        for rate in Train_rate:
            for c in C_list:
                # for C in C_list:

                JHC1 = SVR(kernel="{}".format(kl), C=c)
                JHC2 = SVR(kernel="{}".format(kl))



                # x_pd1 = model1.transform(x_test)
                # x_pd2 = model2.transform(x_test)

                x_tran1, x_test1, y_train1, y_test1 = train_test_split(x1, y[0], test_size= 1 - rate)
                x_tran2, x_test2, y_train2, y_test2 = train_test_split(x2, y[1], test_size= 1 - rate)


                JHC1.fit(x_tran1, y_train1)
                JHC2.fit(x_tran2, y_train2)


                y_bar1 = JHC1.predict(x_test1)
                y_bar2 = JHC2.predict(x_test2)

                print("Kernel====>{} Train rate=====>{} Loss with planed prices=====>{}. Loss with practical prices=====>{}.".format(kl, rate,  r2_score(y_test1, y_bar1), r2_score(y_test2, y_bar2)))


                # y_pd1 = JHC1.predict(x_pd1)
                # y_pd2 = JHC2.predict(x_pd2)

                # np.random.seed(1)
                #
                # # len_pd = len(y_pd2)
                # Y1 = np.random.choice(len_pd, Num_look)
                # Y1 += 1
                #
                # Y1.sort()
                #
                # plt.figure()  # 设置画布的尺寸
                # plt.xticks(np.arange(1, Num_look+1), Y1, rotation=45)
                # plt.plot(np.arange(1, Num_look+1), y_pd1[Y1], c='red', label="指导价格")
                # plt.plot(np.arange(1, Num_look+1), y_pd2[Y1], c='green', label="实际价格")
                # plt.legend(loc=0)
                #
                # plt.savefig("./re/Kernerl={}_rate={}.png".format(kl, rate), dpi=400)

        print("___________________________________________________________________________________________________")




def Training(x_train, y_train, x_test, y_test, learn_rate=1e-3, weight_decay=1e-2, epochs=100, Batch_size=128):

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
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
            lss = loss1(y_bar, train_y) + 1000*loss2(train_x, z)
            Optimizer.zero_grad()
            lss.backward()
            Optimizer.step()

            epoch_loss += lss.item()


        with torch.no_grad():
            y_test_bar, _ = Net(x_test)
            y_test_bar = y_test_bar.cpu().numpy()
            R2_score = r2_score(y_test_bar, y_test)

        print("Epoch====>{} Loss====>{} R2score====>{}".format(epoch, epoch_loss, R2_score))
        epoch_loss = 0


def Predict(x_test):
    pass


def feature_selection(x_, y_):
    print(len(feature_name))
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.feature_selection import SelectFromModel
    print("____________________Begin features selection______________________________________________________________")
    print(x_.shape[1]+1)
    Selector1 = ExtraTreesRegressor(n_estimators=50, criterion='squared_error')
    Selector1 = Selector1.fit(x_, y_[0])
    plt.figure(figsize=(8, 8))
    plt.ylabel("Importances of features")
    plt.xticks(np.arange(1, x_.shape[1] + 1), feature_name, rotation=45)
    Im = np.array(Selector1.feature_importances_)
    print("When we focus on the Planned Prices, the order of importances of features is:\n", [(ftt,np.fabs(imp)) for ftt, imp in zip(feature_name[np.argsort(-Im)], np.sort(-Im))])
    plt.bar(np.arange(1, x_.shape[1]+1), Im,  color='red', label="Planned")
    plt.legend()
    plt.savefig("./re/Importance_Planned.png", dpi=400)
    plt.show()

    model1 = SelectFromModel(Selector1,  prefit=True)
    X1 = model1.transform(x_)


    Selector2 = ExtraTreesRegressor(n_estimators=50, criterion='squared_error')
    Selector2 = Selector2.fit(x_, y_[1])
    plt.figure(figsize=(8, 8))
    plt.ylabel("Importances of features")
    plt.xticks(np.arange(1, x_.shape[1]+1), feature_name, rotation=45, fontsize=15)
    Im = np.array(Selector2.feature_importances_)
    print("When we focus on the Practical Prices, the order of importances of features is:\n", [(ftt,np.fabs(imp)) for ftt, imp in zip(feature_name[np.argsort(-Im)], np.sort(-Im))])
    plt.bar(np.arange(1, x_.shape[1] + 1), Im, color='red', label="Practical")
    plt.legend()
    plt.savefig("./re/Importance_Practical.png", dpi=400)
    plt.show()

    model2 = SelectFromModel(Selector2, prefit=True)
    X2 = model2.transform(x_)

    print("____________________Completed feature selection______________________________________________________________")

    return X1, X2, model1, model2


if __name__ == "__main__":
    X1, X2, model1, model2 = feature_selection(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y[1], test_size=0.7)
    # Training(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

    training_model(X1, X2, model1, model2, y, x_pd)




