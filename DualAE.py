import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  #



class DualR(nn.Module) :

    def __init__(self, in_dims=15, nz=128) :
        super(DualR, self).__init__()
        self.Encoder1 = nn.Sequential(
                nn.Linear(in_dims, 128),
                nn.BatchNorm1d(128, momentum=0.5),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                # nn.Linear(128, 512),
                # nn.BatchNorm1d(512, momentum=0.5),
                #  nn.ReLU(),
                # nn.Dropout(p=0.5),
                nn.Linear(128, 512),
                nn.BatchNorm1d(512, momentum=0.5),
                nn.ReLU(),
                nn.Dropout(p=0.5),

                nn.Linear(512, 128),
                nn.BatchNorm1d(128, momentum=0.5),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, nz),
                nn.ReLU(),
                )
        self.Encoder2 = nn.Sequential(
                nn.Linear(in_dims, 128),
                nn.BatchNorm1d(128, momentum=0.5),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, 512),
                nn.BatchNorm1d(512, momentum=0.5),
                nn.ReLU(),
                nn.Dropout(p=0.5),

                nn.Linear(512, 128),
                nn.BatchNorm1d(128, momentum=0.5),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, nz),
                nn.ReLU(),
                )

        self.Decoder1 = nn.Sequential(
                nn.Linear(nz, 128),
                nn.BatchNorm1d(128, momentum=0.5),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, 512),
                nn.BatchNorm1d(512, momentum=0.5),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(512, 128),
                nn.BatchNorm1d(128, momentum=0.5),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, in_dims),
                nn.ReLU()
                )

        self.Decoder2 = nn.Sequential(
                nn.Linear(nz, 128),
                nn.BatchNorm1d(128, momentum=0.5),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, 512),
                nn.BatchNorm1d(512, momentum=0.5),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(512, 128),
                nn.BatchNorm1d(128, momentum=0.5),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, in_dims),
                nn.ReLU()
                )

    def forward(self, x) :
        h1 = self.Encoder1(x)
        h2 = self.Encoder2(x)
        # h = F.relu(h)
        y_pr = torch.sum(h2, dim=1)

        x_bar1 = self.Decoder1(h1)
        x_bar2 = self.Decoder2(h2)
        # hat = F.relu(hat)
        y_pl = torch.sum(h1, dim=1)
        # z2 = self.Decoder(hat)


        return h1, h2, x_bar1, x_bar2, y_pl, y_pr

    def predict(self, x) :
        with torch.no_grad() :
            h1 = self.Encoder1(x)
            h2 = self.Encoder2(x)
            # h = F.relu(h)
            # y_pr = torch.sum(h1, dim=1)
            # y_pl = torch.sum(h2, dim=1)

        return h1, h2


