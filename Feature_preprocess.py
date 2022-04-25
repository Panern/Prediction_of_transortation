import pandas as pd
import xlwt
import xlrd
import numpy as np
import datetime

def Observe_data():
    train_data = pd.read_excel("./data/Train.xlsx", sheet_name=0, header=0)
    test_data = pd.read_excel("./data/Test.xlsx", sheet_name=0, header=0)
    processed_data = pd.read_excel("./data/Processed.xlsx", sheet_name=4)

    Train_N_instances = len(train_data)
    Train_N_features = len(train_data.keys())

    New_N_instaces = len(processed_data)
    New_N_features = len(processed_data.keys())

    Test_N_instances = len(test_data)
    Test_N_features = len(test_data.keys())

    print("Training dataset has {} instances and {} attributes".format(Train_N_instances, Train_N_features))
    print("Test dataset has {} instances and {} attribute".format(Test_N_instances, Test_N_features))
    Shared_features = set(train_data.keys()).intersection(set(test_data.keys()))
    Shared_features_done = set(processed_data.keys()).intersection(set(test_data.keys()))
    # print(New_N_features)
    # print(processed_data.keys())
    # print(test_data.keys())
    # print(len(set(train_data.keys()).intersection(set(test_data.keys()))))
    print(len(Shared_features_done))
    #
    #
    # def Preprocess_data():
    #     # workbook = xlrd.open_workbook("./data/Train_1.xlsx")
    train_data = pd.read_excel("./data/Processed.xlsx", sheet_name=4, header=0)



    book = xlwt.Workbook()

    # sheet = book.add_sheet("shared_features")
    features_name = []
    # for idx, fts in enumerate(Shared_features_done):
    #     features_name.append(fts)
    #     sheet.write(0, idx, fts)
    #     len_ft = len(train_data[fts])
    #     
    #     for row in range(len_ft):
    #         content = str(train_data[fts][row])
    #         sheet.write(row+1, idx, content)
    # 
    # 
    # book.save("./data/Train_4.xls")
    features = []
    Time = [[],[]]
    labels = []
    labels.append(np.array(train_data["线路指导价（不含税）"]).tolist())
    labels.append(np.array(train_data["线路总成本"]).tolist())

    test_Time = [[], []]
    test_features = []
    print(len(labels[0]), len(labels[1]))
    for idx, fts in enumerate(Shared_features_done):
        if "时间" in fts:
            Time[0].append(fts)
            Time[1].append(np.array(train_data[fts], dtype=str).tolist())
            test_Time[0].append(fts)
            test_Time[1].append(np.array(test_data[fts], dtype=str).tolist())
        else:
            features_name.append(fts)
            features.append(np.array(train_data[fts]).tolist())
            test_features.append(np.array(test_data[fts]).tolist())
        
    # print(features, len(features))
    # print(features_name)
    # dd1 = features_name.index("车辆长度")
    # print(type(features[dd1][1]))




    for idxx, feature in enumerate(features):
        if isinstance(feature[0], str):
            if "N" in feature:
                value = (np.unique(feature).tolist()).remove("N")
                for idx, vl in enumerate(feature):
                    if vl == "N":
                        features[idxx][idx] = 0
                    else:
                        features[idxx][idx] = value.index(vl)
            else:
                value = np.unique(feature).tolist()
                for idx, vl in enumerate(feature):
                    features[idxx][idx] = value.index(vl)

        else:
            continue

    for idxx, feature in enumerate(test_features):
        if isinstance(feature[0], str):
            if "N" in feature:
                value = (np.unique(feature).tolist()).remove("N")
                for idx, vl in enumerate(feature):
                    if vl == "N":
                        test_features[idxx][idx] = 0
                    else:
                        test_features[idxx][idx] = value.index(vl)
            else:
                value = np.unique(feature).tolist()
                for idx, vl in enumerate(feature):
                    test_features[idxx][idx] = value.index(vl)

        else:
            continue

    test_features_time = [["计划装货时间", "计划运输时间"], [[],[]]]
    features_time = [["计划装货时间", "计划运输时间"], [[],[]]]
    idx_ta, idx_tstop, idx_tstart = Time[0].index('计划到达时间'), Time[0].index('计划靠车时间'), Time[0].index('计划发车时间')

    test_idx_ta, test_idx_tstop, test_idx_tstart = test_Time[0].index('计划到达时间'), test_Time[0].index('计划靠车时间'), test_Time[0].index('计划发车时间')
    # for i in range(10):
    #     print(Time[1][idx_ta][i])

    for t_arriving, t_starting, t_stopping in zip(Time[1][idx_ta], Time[1][idx_tstart], Time[1][idx_tstop]):
        t_arriving = datetime.datetime.strptime(t_arriving, "%Y-%m-%dT%H:%M:%S.000000000")

        t_starting = datetime.datetime.strptime(t_starting, "%Y-%m-%dT%H:%M:%S.000000000")
        t_stopping = datetime.datetime.strptime(t_stopping, "%Y-%m-%dT%H:%M:%S.000000000")
        features_time[1][1].append((t_arriving - t_starting).total_seconds())
        features_time[1][0].append((t_starting - t_stopping).total_seconds())

        pass

    for t_arriving, t_starting, t_stopping in zip(test_Time[1][idx_ta], test_Time[1][idx_tstart], test_Time[1][idx_tstop]):
        t_arriving = datetime.datetime.strptime(t_arriving, "%Y-%m-%dT%H:%M:%S.000000000")

        t_starting = datetime.datetime.strptime(t_starting, "%Y-%m-%dT%H:%M:%S.000000000")
        t_stopping = datetime.datetime.strptime(t_stopping, "%Y-%m-%dT%H:%M:%S.000000000")
        test_features_time[1][1].append((t_arriving - t_starting).total_seconds())
        test_features_time[1][0].append((t_starting - t_stopping).total_seconds())

        pass
    features_name.extend(features_time[0])
    features.append(features_time[1][1])
    features.append(features_time[1][0])
    # dd = features_name.index("需求类型2")
    # print(features[dd][1:10])


    test_features.append(test_features_time[1][1])
    test_features.append(test_features_time[1][0])


    labels = np.array(labels)
    features = np.array(features)
    test_features = np.array(test_features)
    del_idx = []
    for idx, (ls1, ls2) in enumerate(zip(labels[0], labels[1])):
        if (ls1 == "N") or (ls2 == "N"):
            del_idx.append(idx)

    Labels = []

    print(test_features.shape)
    features = features.T
    test_features = test_features.T
    Labels.append(np.delete(labels[0], del_idx, axis=0))
    Labels.append(np.delete(labels[1], del_idx, axis=0))
    Ft = np.delete(features, del_idx, axis=0)

    labels = np.array(Labels)
    features = np.array(Ft)
    print(len(labels[0]), len(labels[1]))
    print(len(features))
    labels = np.array([list(map(float, labels[0])), list(map(float, labels[1]))])
    print(features.shape, test_features.shape)
    print(features_name)
    np.save("./data/features_name", features_name)
    np.savetxt("./data/train_features.txt", features, delimiter=',')
    np.savetxt("./data/train_labels.txt", labels, delimiter=',')
    np.savetxt("./data/test_features.txt", test_features, delimiter=',')
    #print(features, len(features))

# Preprocess_data()
def New_process():
    train_data = pd.read_excel("./data/Train.xlsx", sheet_name=0, header=0)
    test_data = pd.read_excel("./data/Test.xlsx", sheet_name=0, header=0)
    processed_data = pd.read_excel("./data/Processed.xlsx", sheet_name=4)

    Train_N_instances = len(train_data)
    Train_N_features = len(train_data.keys())

    New_N_instaces = len(processed_data)
    New_N_features = len(processed_data.keys())

    Test_N_instances = len(test_data)
    Test_N_features = len(test_data.keys())

    print("Training dataset has {} instances and {} attributes".format(Train_N_instances, Train_N_features))
    print("Test dataset has {} instances and {} attribute".format(Test_N_instances, Test_N_features))
    Shared_features = set(train_data.keys()).intersection(set(test_data.keys()))
    Shared_features_done = set(processed_data.keys()).intersection(set(test_data.keys()))
    train_features = train_data.keys()
    train_features = processed_data.keys()
    # print(New_N_features)
    # print(processed_data.keys())
    # print(test_data.keys())
    # print(len(set(train_data.keys()).intersection(set(test_data.keys()))))
    # print(len(Shared_features_done))
    #
    #
    # def Preprocess_data():
    #     # workbook = xlrd.open_workbook("./data/Train_1.xlsx")
    # train_data = pd.read_excel("./data/Processed.xlsx", sheet_name=4, header=0)

    book = xlwt.Workbook()

    # sheet = book.add_sheet("shared_features")
    features_name = []
    # for idx, fts in enumerate(Shared_features_done):
    #     features_name.append(fts)
    #     sheet.write(0, idx, fts)
    #     len_ft = len(train_data[fts])
    #
    #     for row in range(len_ft):
    #         content = str(train_data[fts][row])
    #         sheet.write(row+1, idx, content)
    #
    #
    # book.save("./data/Train_4.xls")
    features = []
    Time = [[], []]
    labels = []
    labels.append(np.array(train_data["线路指导价（不含税）"]).tolist())
    labels.append(np.array(train_data["线路总成本"]).tolist())
    print("There are {} features.".format(len(train_features)))
    # test_Time = [[], []]
    # test_features = []
    print(len(labels[0]), len(labels[1]))
    ct = 0
    for idx, fts in enumerate(Shared_features_done):
        if ("线路指导价" not in fts) and ("线路总成本" not in fts):
            if "时间" in fts and ("666" not in fts):
                Time[0].append(fts)
                Time[1].append(np.array(train_data[fts], dtype=str).tolist())
                print(fts)
                # test_Time[0].append(fts)
                # test_Time[1].append(np.array(test_data[fts], dtype=str).tolist())
            elif "666" in fts:
                continue
            else:
                ct += 1
                # print(fts)
                features_name.append(fts)
                features.append(np.array(train_data[fts]).tolist())
                # test_features.append(np.array(test_data[fts]).tolist())

        else:
            print(fts)
    print(np.array(features).shape, ct)
    # print(features_name)
    # dd1 = features_name.index("车辆长度")
    # print(type(features[dd1][1]))

    for idxx, feature in enumerate(features):
        if isinstance(feature[0], str):
            if "N" in feature:
                value = (np.unique(feature).tolist()).remove("N")
                for idx, vl in enumerate(feature):
                    if vl == "N":
                        try:
                            features[idxx][idx] = value[0]
                        except:
                            value_ = np.delete(np.unique(feature), 0).tolist()
                            features[idxx][idx] = value_[0]
                    else:
                        try:
                            features[idxx][idx] = value.index(vl)
                        except:
                            value_ = np.delete(np.unique(feature), 0).tolist()
                            features[idxx][idx] = value_.index(vl)

            else:
                value = np.unique(feature).tolist()
                # print(isinstance(value, None))
                for idx, vl in enumerate(feature):
                    features[idxx][idx] = value.index(vl)


        else:
            continue

    # for idxx, feature in enumerate(test_features):
    #     if isinstance(feature[0], str):
    #         if "N" in feature:
    #             value = (np.unique(feature).tolist()).remove("N")
    #             for idx, vl in enumerate(feature):
    #                 if vl == "N":
    #                     test_features[idxx][idx] = 0
    #                 else:
    #                     test_features[idxx][idx] = value.index(vl)
    #         else:
    #             value = np.unique(feature).tolist()
    #             for idx, vl in enumerate(feature):
    #                 test_features[idxx][idx] = value.index(vl)

        # else:
        #     continue

    # test_features_time = [["计划装货时间", "计划运输时间"], [[], []]]
    features_time = [["计划装货时间", "计划运输时间"], [[], []]]
    idx_ta, idx_tstop, idx_tstart = Time[0].index('计划到达时间'), Time[0].index('计划靠车时间'), Time[0].index('计划发车时间')

    # test_idx_ta, test_idx_tstop, test_idx_tstart = test_Time[0].index('计划到达时间'), test_Time[0].index('计划靠车时间'), \
    #                                                test_Time[0].index('计划发车时间')
    # for i in range(10):
    #     print(Time[1][idx_ta][i])

    for t_arriving, t_starting, t_stopping in zip(Time[1][idx_ta], Time[1][idx_tstart], Time[1][idx_tstop]):
        t_arriving = datetime.datetime.strptime(t_arriving, "%Y-%m-%dT%H:%M:%S.000000000")

        t_starting = datetime.datetime.strptime(t_starting, "%Y-%m-%dT%H:%M:%S.000000000")
        t_stopping = datetime.datetime.strptime(t_stopping, "%Y-%m-%dT%H:%M:%S.000000000")
        features_time[1][1].append((t_arriving - t_starting).total_seconds())
        features_time[1][0].append((t_starting - t_stopping).total_seconds())

        pass
    # idx_ta, idx_tstop, idx_tstart = Time[0].index('实际到车时间'), Time[0].index('实际靠车时间'), Time[0].index('实际发车时间')

        # test_idx_ta, test_idx_tstop, test_idx_tstart = test_Time[0].index('计划到达时间'), test_Time[0].index('计划靠车时间'), \
        #                                                test_Time[0].index('计划发车时间')
        # for i in range(10):
        #     print(Time[1][idx_ta][i])
    #
    # for t_arriving, t_starting, t_stopping in zip(Time[1][idx_ta], Time[1][idx_tstart], Time[1][idx_tstop]):
    #     t_arriving = datetime.datetime.strptime(t_arriving, "%Y-%m-%dT%H:%M:%S.000000000")
    #
    #     t_starting = datetime.datetime.strptime(t_starting, "%Y-%m-%dT%H:%M:%S.000000000")
    #     t_stopping = datetime.datetime.strptime(t_stopping, "%Y-%m-%dT%H:%M:%S.000000000")
    #     features_time[1][2].append((t_arriving - t_starting).total_seconds())
    #     features_time[1][3].append((t_starting - t_stopping).total_seconds())

    # idx_ta, idx_tstop, idx_tstart = Time[0].index('交易成功时间'), Time[0].index('交易结束时间'), Time[0].index('实际结束时间')

        # test_idx_ta, test_idx_tstop, test_idx_tstart = test_Time[0].index('计划到达时间'), test_Time[0].index('计划靠车时间'), \
        #                                                test_Time[0].index('计划发车时间')
        # for i in range(10):
        #     print(Time[1][idx_ta][i])

    # for t_arriving, t_starting, t_stopping in zip(Time[1][idx_ta], Time[1][idx_tstart], Time[1][idx_tstop]):
    #     t_arriving = datetime.datetime.strptime(t_arriving, "%Y-%m-%dT%H:%M:%S")
    #
    #     t_starting = datetime.datetime.strptime(t_starting, "%Y-%m-%dT%H:%M:%S")
    #
    #     features_time[1][5].append((t_starting - t_arriving).total_seconds())
    #     features_time[1][4].append((t_stopping - t_arriving).total_seconds())
    # for t_arriving, t_starting, t_stopping in zip(test_Time[1][idx_ta], test_Time[1][idx_tstart],
    #                                               test_Time[1][idx_tstop]):
    #     t_arriving = datetime.datetime.strptime(t_arriving, "%Y-%m-%dT%H:%M:%S.000000000")
    #
    #     t_starting = datetime.datetime.strptime(t_starting, "%Y-%m-%dT%H:%M:%S.000000000")
    #     t_stopping = datetime.datetime.strptime(t_stopping, "%Y-%m-%dT%H:%M:%S.000000000")
    #     test_features_time[1][1].append((t_arriving - t_starting).total_seconds())
    #     test_features_time[1][0].append((t_starting - t_stopping).total_seconds())

        pass
    features_name.extend(features_time[0])
    # features_name.extend(features_time[1])
    features.append(features_time[1][1])
    features.append(features_time[1][0])
    # features.append(features_time[1][2])
    # features.append(features_time[1][3])
    # dd = features_name.index("需求类型2")
    # print(features[dd][1:10])

    # test_features.append(test_features_time[1][1])
    # test_features.append(test_features_time[1][0])

    labels = np.array(labels)
    features = np.array(features)
    # test_features = np.array(test_features)
    del_idx = []
    for idx, (ls1, ls2) in enumerate(zip(labels[0], labels[1])):
        if (ls1 == "N") or (ls2 == "N"):
            del_idx.append(idx)

    Labels = []

    # print(test_features.shape)
    features = features.T
    print(labels.shape)
    # test_features = test_features.T
    Labels.append(np.delete(labels[0], del_idx, axis=0))
    Labels.append(np.delete(labels[1], del_idx, axis=0))
    Ft = np.delete(features, del_idx, axis=0)

    labels = np.array(Labels)
    features = np.array(Ft)
    print(features.shape)
    print(len(labels[0]), len(labels[1]))
    print(len(features))
    labels = np.array([list(map(float, labels[0])), list(map(float, labels[1]))])
    print(features_name)
    np.save("./data/features_name_new", features_name)
    np.savetxt("./data/train_features_new.txt", features, delimiter=',', fmt='%s')
    np.savetxt("./data/train_labels_new.txt", labels, delimiter=',')
    # np.savetxt("./data/test_features_new.txt", test_features, delimiter=',')
    # print(features, len(features))


# New_process()
Observe_data()