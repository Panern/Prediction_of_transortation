
from sklearn.model_selection import train_test_split
import numpy as np
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import torch.utils.data as Data
from TrainAndEvaluate import train, evaluate, eval_func
import torch
import joblib
import scipy.sparse as sp
from ax.service.managed_loop import optimize
from torch.optim import Adam
from Lossses import sce_loss, Regloss, BarlowTwins_loss
import torch.nn.functional as F
import random
import warnings
import logging

from torch.backends import cudnn

warnings.filterwarnings("ignore")

class BoOper() :

    def __init__(self, random_seed=12345, Ex_name="Trpp", model=None, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) :
        '''
        Args:
            :type
                "DL" or "ML"
            :datafunc
                data and lables for training and testing

            :num_BoOpt:
                iteration number of BayesianOptimization
            :random_seed:
                random seed for reproduce result
            :Ex_name
                name for Bo and writting logs
        '''


        self.re = None


        self.seed = random_seed
        self.Ex_name = Ex_name

        self.model = model
        self.device = device
        self.Predictor1 = None
        self.Predictor2 = None
        self.Emergence_predictor = None

    def setup_seed(self) :
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        cudnn.deterministic = True

    def loader_get(self,x_train, x_test, y_train, y_test, Batch_size=128) :
       
       x_train = torch.from_numpy(x_train)
       x_train = x_train.to(torch.float32)
      
       y_train = torch.from_numpy(y_train)
       y_train = y_train.to(torch.float32)

       # print(x_train.size(), y_train.size())

       train_data = Data.TensorDataset(x_train, y_train)
       train_loader = Data.DataLoader(dataset=train_data, batch_size=Batch_size, shuffle=True, num_workers=0)

       N_test = max(x_test.shape[0], x_test.shape[1])
       x_test = torch.from_numpy(x_test)
       x_test = x_test.to(torch.float32)
       y_test = torch.from_numpy(y_test)
       y_test = y_test.to(torch.float32)
       
       
       test_data = Data.TensorDataset(x_test, y_test)
       test_loader = Data.DataLoader(dataset=test_data, batch_size=N_test, shuffle=True, num_workers=0)

       self.train_loader = train_loader
       self.test_loader = test_loader

    def BoRun(self, num_BoOpt=10, parameters=None) :
        '''
        :return:
            best_parameters, values, experiment
        '''

        self.num_BoOpt = num_BoOpt
        dtype = torch.float
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.setup_seed()


        def train_evaluate(parameters) :
            net = DualR(nz=5)
            net, Xg1, Xg2 = train(
                    net=net, train_loader=self.train_loader, \
                    parameters=parameters, dtype=dtype, device=device
                    )
            re = evaluate(
                    net=net,
                    data_loader=self.test_loader,
                    XG1=Xg1,
                    XG2=Xg2,
                    )
            return re

        best_parameters, values, experiment, model = optimize(
                parameters=parameters,
                evaluation_function=train_evaluate,
                objective_name='R2',
                total_trials=self.num_BoOpt,
                random_seed=self.seed,
                experiment_name=self.Ex_name,
                )
        data = experiment.fetch_data()
        df = data.df
        best_arm_name = df.arm_name[df['mean'] == df['mean'].max()].values[0]
        best_arm = experiment.arms_by_name[best_arm_name]
        net, Xg1, Xg2 = train(
                net=self.model, train_loader=self.train_loader, \
                parameters=best_arm.parameters, dtype=dtype, device=device
                )

        self.model = net
        self.Predictor1, self.Predictor2 = Xg1, Xg2
        self.optimal_paras = best_parameters
        self.re = values
        self.experiment = experiment

        # self.model = model
        # print(type(self.model))
        
    def show(self) :
        print("Ex:{} obtains best result: {} with parameters: {} ".format(self.Ex_name, self.re, self.optimal_paras))
        
    def training(self,learn_rate=1e-3, weight_decay=1e-2, epochs=500):

        from Models import XGBOOST
        self.setup_seed()
        self.re = [[0, 0, 0],[0, 0, 0]]
        self.optimal_paras = {'lr': learn_rate, 'weight_decay': weight_decay, 'epoch_pl': 0, 'epoch_pr': 0}
        self.model.to(self.device)

        train_loader, test_loader = self.train_loader, self.test_loader

        optimizer = Adam(self.model.parameters(), lr=learn_rate, weight_decay=weight_decay)

        for epoch in range(epochs) :
            epoch_loss = 0
            lss_pd = 0
            H1 = []
            H2 = []
            Y_pl = []
            Y_pr = []
            for step, (train_x, train_y) in enumerate(train_loader) :
                train_x = train_x.cuda()
                train_y = train_y.cuda()

                optimizer.zero_grad()

                h1, h2, x_bar1, x_bar2, y_pl, y_pr = self.model(train_x)
                # if step == 0 :
                #     print(y_hat, train_y[:, 0])
                lss_reg_pl = Regloss(y_pl, train_y[:, 0])
                lss_reg_pr = Regloss(y_pr, train_y[:, 1])

                lss_rec1 = sce_loss(x_bar1, train_x)
                lss_rec2 = sce_loss(x_bar2, train_x)

                lss_bl = BarlowTwins_loss(h1, h2, bn_size=5)
                lss =  lss_bl +  (lss_rec1 + lss_rec2) + (lss_reg_pl + lss_reg_pr)
                lss_pd += lss.item()
                lss.backward()
                optimizer.step()

                with torch.no_grad() :
                    H1.extend(h1.cpu().numpy().tolist())
                    H2.extend(h2.cpu().numpy().tolist())
                    Y_pl.extend(train_y[:,0].cpu().numpy().tolist())
                    Y_pr.extend(train_y[:,1].cpu().numpy().tolist())

            epoch_loss += lss_pd

            with torch.no_grad() :

                XG1, T1 = XGBOOST(H1, Y_pl)
                XG2, T2 = XGBOOST(H2, Y_pr)

                for x_test, y_test in test_loader :
                    x_test = x_test.cuda()
                    h1, h2,  _, _ ,  _, _ = self.model(x_test)

                    y_test_bar = XG1.predict(h1.cpu().numpy())
                    y_test_hat = XG2.predict(h2.cpu().numpy())

                    R2_pl, MAE_pl, RMSE_pl = eval_func(y_test_bar, y_test[:, 0])
                    R2_pr, MAE_pr, RMSE_pr = eval_func(y_test_hat, y_test[:, 1])

                if R2_pl > self.re[0][0]:
                    self.re[0] = R2_pl, MAE_pl, RMSE_pl
                    self.Predictor1 = XG1
                    self.optimal_paras['epoch_pl'] = epoch
                if R2_pr > self.re[1][0]:
                    self.re[1] = R2_pr, MAE_pr, RMSE_pr
                    self.Predictor2 = XG2
                    self.optimal_paras['epoch_pr'] = epoch

                if R2_pl >= 0.9996 or R2_pr >= 0.9996:
                    break

            if epoch % 10 == 9 :
                print(
                        "Epoch====>{} Loss====>{} R2_Pl====>{} MAE_pl====>{}, RMSE_pl====>{} R2_Pr====>{} MAE_pr====>{}, RMSE_pr====>{}".format(
                                epoch+1, epoch_loss, R2_pl, MAE_pl, RMSE_pl, R2_pr, MAE_pr, RMSE_pl
                                )
                        )

    def emergency_training(self, em, y):
        from sklearn.ensemble import GradientBoostingClassifier
        x_train, x_test, y_train, y_test = train_test_split(y, em, test_size=0.2, random_state=self.seed)
        clf = GradientBoostingClassifier()
        rf = clf.fit(x_train, y_train.ravel())
        print("Emergency traing result：")
        print("On training set：", rf.score(x_train, y_train))
        print("On validation set：", rf.score(x_test, y_test))
        self.Emergence_predictor = rf



def getdata() :
    x = np.loadtxt("./data/train_features.txt", delimiter=",")
    # print(x.shape)
    y = np.loadtxt("./data/train_labels.txt", delimiter=',')
    x_pd = np.loadtxt("./data/test_features.txt", delimiter=',')

    return x, y, x_pd



def Sigmoid(x, alpha=1e3) :
    return 1/(1+np.exp(-x/alpha))

def Inverse_sigmoid(y, alpha=1e3) :
    return alpha * np.log(y/(1-y))



if __name__ == "__main__" :

    from sklearn.preprocessing import MinMaxScaler
    from DualAE import DualR
    x, y, x_pd = getdata()
    train_rate = 0.8
    Train = True
    See_list = [i for i in range(100, 120)]
    # x_pd = Sigmoid(x_pd)
    x_pd = torch.from_numpy(x_pd).float()
    prices_plan =[]
    prices_prictical = []
    emergency = []
    prices_plan_test = []
    prices_prictical_test = []
    emergency_test = []
    if Train :
        # train 10 times, and get its Avg.
        R2Pr_list = []
        R2Pl_list = []

        for sd in range(1) :
            Optizimer1 = BoOper(model=DualR(nz=5))
            # SC = MinMaxScaler()
            # print(y[0])
            x_train, x_test, y_train, y_test = train_test_split(x, y[0], train_size=train_rate, random_state=sd + 1)

            # y_train = SC.fit_transform(y_train.reshape(1, -1))
            y_train = y_train.reshape(1, -1)
            # x_train, x_test = Sigmoid(x_train), Sigmoid(x_test)
            # y_train = Sigmoid(y_train)
            # y_test = SC.fit_transform(y_test.reshape(1, -1))
            y_test = y_test.reshape(1, -1)
            # y_test = Sigmoid(y_test)
            _, _, y_train1, y_test1 = train_test_split(x, y[1], train_size=train_rate, random_state=sd + 1)
            # y_train1 = SC.fit_transform(y_train1.reshape(1, -1))
            y_train1 = y_train1.reshape(1, -1)
            # y_train1 = Sigmoid(y_train1)
            # y_test1 = SC.fit_transform(y_test1.reshape(1, -1))
            y_test1 = y_test1.reshape(1, -1)
            # y_test1 = Sigmoid(y_test1)
            y_train = np.vstack((y_train, y_train1)).T
            y_test = np.vstack((y_test, y_test1)).T

            Optizimer1.loader_get(x_train, x_test, y_train, y_test)
            # Optizimer1.training(learn_rate=1e-2, weight_decay=1e-4, epochs=100)
            Optizimer1.BoRun(num_BoOpt=10, parameters=[
                        {"name" : "lr", "type" : "range", "bounds" : [1e-6, 0.99], "log_scale" : True},
                        {"name" : "momentum", "type" : "range", "bounds" : [0.0, 1.0]},
                        {"name" : "num_epochs", "type" : "range", "bounds" : [1, 100]},
                                    ])

            Optizimer1.show()
            torch.save(Optizimer1.model, "./model/DualPPR.pkl")
            joblib.dump(Optizimer1.Predictor1, "./model/Predictor1.pkl")
            joblib.dump(Optizimer1.Predictor2, "./model/Predictor2.pkl")

            x_pd = x_pd.cuda()
            h1, h2 = Optizimer1.model.predict(x_pd)
            h1, h2 = h1.cpu().numpy(), h2.cpu().numpy()
            Pr_pl, Pr_pr = Optizimer1.Predictor1.predict(h1), Optizimer1.Predictor2.predict(h2)
            # Pr_pl, Pr_pr = torch.unsqueeze(Pr_pl, 0), torch.unsqueeze(Pr_pr, 0)
            # Pr_pl, Pr_pr = Pr_pl.cpu().numpy(),  Pr_pr.cpu().numpy()
            # Pr_pl, Pr_pr = Inverse_sigmoid(Pr_pl), Inverse_sigmoid(Pr_pr)
            # Pr_pl, Pr_pr = SC.inverse_transform(Pr_pl), SC.inverse_transform(Pr_pr)
            Optizimer1.emergency_training(x[:, 1], y.T)
            joblib.dump(Optizimer1.Emergence_predictor, "./model/Emergence_predictor.pkl")
            EM = Optizimer1.Emergence_predictor.predict(np.array([Pr_pl, Pr_pr]).T)
            emergency.append(EM)
            print(Pr_pl[:20], Pr_pr[:20])
            prices_plan.append(Pr_pl)
            prices_prictical.append(Pr_pr)
            x_test = torch.from_numpy(x_test).float()
            x_test = x_test.cuda()
            h1_test, h2_test = Optizimer1.model.predict(x_test)
            h1_test, h2_test = h1_test.cpu().numpy(), h2_test.cpu().numpy()
            Pr_pl_test, Pr_pr_test = Optizimer1.Predictor1.predict(h1_test), Optizimer1.Predictor2.predict(h2_test)
            # Pr_pl, Pr_pr = torch.unsqueeze(Pr_pl, 0), torch.unsqueeze(Pr_pr, 0)
            # Pr_pl, Pr_pr = Pr_pl.cpu().numpy(),  Pr_pr.cpu().numpy()
            # Pr_pl, Pr_pr = Inverse_sigmoid(Pr_pl), Inverse_sigmoid(Pr_pr)
            # Pr_pl, Pr_pr = SC.inverse_transform(Pr_pl), SC.inverse_transform(Pr_pr)
            # Optizimer1.emergency_training(x[:, 1], y.T)
            EM_test = Optizimer1.Emergence_predictor.predict(np.array([Pr_pl_test, Pr_pr_test]).T)
            emergency_test.append(EM_test)
            # print(Pr_pl[:20], Pr_pr[:20])
            prices_plan_test.append(Pr_pl_test)
            prices_prictical_test.append(Pr_pr_test)

            # R2Pl_list.append(Optizimer1.re[0][0])
            # R2Pr_list.append(Optizimer1.re[1][0])

        # num_model = np.argmax(np.array(R2Pl_list))
        # # print(num_model)
        # print("Best R2score for Pl:{}".format(R2Pl_list[num_model]))
        # print("Best R2score for Pr:{}".format(R2Pr_list[num_model]))
        # prices, scales = Pr_pl[See_list], (Pr_pr / Pr_pl)[See_list]
        # print("We can see the predicted prices\scales 10 items, index of which are ")
        # print(See_list)
        # for j in See_list :
        #     print(
        #         "No. {} item======>Price:{}========>Suggested Scale-value:{}".format(
        #             j, round(prices[0][j], 2), scales[0][j]
        #             )
        #         )

        np.save("./re/prices_plan_DualNN.npy", np.array(prices_plan))
        np.save("./re/prices_prictical.npy", np.array(prices_prictical))
        np.save("./re/emergency.npy", np.array(emergency))

        np.save("./re/prices_plan_DualNN_test.npy", np.array(prices_plan_test))
        np.save("./re/prices_prictical_test.npy", np.array(prices_prictical_test))
        np.save("./re/emergency_test.npy", np.array(emergency_test))
    else :
        prices, scales = Predict(x_pd, 1)
        print(prices.shape)
        print("We can see the predicted prices\scales 10 items, index of which are ")
        print(See_list)
        for j in See_list :
            print(
                "No. {} item======>Price:{}========>Suggested Scale-value:{}".format(
                    j, round(prices[0][j], 2), scales[0][j]
                    )
                )

        # training_model(X1, X2, model1, model2, y, x_pd)