import sys
import warnings
sys.path.append("../")
warnings.filterwarnings("ignore")
from util.HeadTransfer import HeadTransfer
import random
import pandas as pd
import numpy as np
import networkx as nx
import collections
import torch
from random import shuffle
from sklearn.model_selection import  StratifiedKFold
from model.GraphSage import GraphSage_Net
import torch
from torchmetrics import AUROC
from torchmetrics.classification import BinaryAccuracy,BinarySpecificity,BinaryPrecision
from util.Sensitivity import sensitivity

import copy

RPI_list=["RPI369"]
for RPI in RPI_list:
    txt_name=RPI+'HeadTransfer_GraphSage.txt'
    txt = open('../result/'+txt_name, mode = 'w')
    _rate = [0.01, 0.05, 0.1, 0.2, 0.3]
    edge_list = [5, 10, 15]
    file_name=["NPInter2"]
    for file in file_name:
        for rate in _rate:
            for edge_num in edge_list:
                random.seed(123)
                data=HeadTransfer(adj_path="../data/"+RPI+".xlsx",rna_feature_path="../data/lncRNA_3_mer/"+RPI+"/lncRNA_3_mer.txt",protein_feature_path="../data/protein_2_mer/"+RPI+"/protein_2_mer.txt",inter2_adj_path="../data/"+file+".xlsx",inter2_rna_feature_path=r"../data/lncRNA_3_mer/"+file+"/lncRNA_3_mer.txt",inter2_protein_feature_path=r"../data/protein_2_mer/"+file+"/protein_2_mer.txt")
                for times in range(5):
                    txt.writelines(str(rate)+' '+str(times+1)+str(edge_num)+"\n")
                    data.rebuild()
                    data.build_transfer_data(rate=rate,edge_num=edge_num,add_adj=False)
                    adj,train_edge_index,train_edge_label,test_edge_index,test_edge_label=data.get_transfer_data()
                    tensor_feature_matrix=copy.deepcopy(data.tensor_feature_matrix)
                    
                    #进行k_fold交叉验证
                    skf = StratifiedKFold(n_splits=5, shuffle=False)
                    index_temp=np.array(data.ts_train_edge_index)
                    label_temp=np.array(data.ts_train_edge_label)
                    eva_value=0
                    ave_eva = 0
                    for train_index, val_index in skf.split(index_temp,label_temp):
                        train_x,val_x=index_temp[train_index],index_temp[val_index]
                        train_y,val_y=label_temp[train_index],label_temp[val_index]
                        train_x,train_y=list(train_x),list(train_y)
                        val_x,val_y=list(val_x),list(val_y)
                        for index in range(len(train_x)):
                            if train_y[index]==1:
                                train_x.append((train_x[index][1],train_x[index][0]))
                                train_y.append(1)
                        val_adj=list()
                        for index in range(len(train_x)):
                            if train_y[index]==1:
                                val_adj.append(train_x[index])

                        val_adj=torch.tensor(val_adj).to(torch.long).T.cuda()
                        train_x=torch.tensor(train_x).to(torch.long).T.cuda()
                        train_y=torch.tensor(train_y).to(torch.long).cuda()
                        val_x=torch.tensor(val_x).to(torch.long).T.cuda()
                        val_y=torch.tensor(val_y).to(torch.long).cuda()
                        
                        model = GraphSage_Net(len(tensor_feature_matrix[0]), 64, 16).cuda()
                        model.train()
                        opt = torch.optim.Adam(params=model.parameters(), lr=0.01)
                        loss_fn = torch.nn.BCELoss().cuda()


                        epoch = 800


                        for i in range(epoch):
                            y_hat = model(tensor_feature_matrix, val_adj, train_x)
                            loss = loss_fn(y_hat.to(torch.float32), train_y.to(torch.float32))
                            opt.zero_grad()
                            loss.backward()
                            opt.step()
                        model.eval()
                        Auc = AUROC(task="binary")

                        with torch.no_grad():
                            y_hat = model(tensor_feature_matrix, val_adj, val_x)
                            val_y = val_y.cpu()
                            y_hat = y_hat.cpu()
                            Auc_value = Auc(y_hat, val_y).item()
                            Acc_value = 0
                            for j in range(1, 10):
                                i = float(j / 10)
                                Acc = BinaryAccuracy(i)
                                if Acc(y_hat, val_y).item() > Acc_value:
                                    Acc_value = Acc(y_hat, val_y).item()
                            ave_eva += Acc_value * 70 + Auc_value * 30
                            if eva_value < (Acc_value * 70 + Auc_value * 30):
                                eva_value = Acc_value * 70 + Auc_value * 30
                                torch.save(model, "../model/HeadTransfer_best.pt")  # 用来评估和读取

                    model = torch.load("../model/HeadTransfer_best.pt").cuda()
                    model.train()
                    opt = torch.optim.Adam(params=model.parameters(), lr=0.01)
                    loss_fn = torch.nn.BCELoss().cuda()

                    adj=torch.tensor(adj).to(torch.long).T.cuda()
                    train_edge_index=torch.tensor(train_edge_index).to(torch.long).T.cuda()
                    train_edge_label=torch.tensor(train_edge_label).to(torch.long).cuda()
                    test_edge_index=torch.tensor(test_edge_index).to(torch.long).T.cuda()
                    test_edge_label=torch.tensor(test_edge_label).to(torch.long).cuda()

                    epoch = 600


                    for i in range(epoch):
                        y_hat = model(tensor_feature_matrix, adj, train_edge_index)
                        loss = loss_fn(y_hat.to(torch.float32), train_edge_label.to(torch.float32))
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                    model.eval()

                    Auc = AUROC(task="binary")

                    with torch.no_grad():
                        y_hat = model(tensor_feature_matrix, adj, test_edge_index)
                        test_edge_label = test_edge_label.cpu()
                        y_hat = y_hat.cpu()
                        txt.writelines(f"{epoch}轮训练\nAUC:{Auc(y_hat, test_edge_label)}\n")
                        # print(f"{epoch}轮训练\nAUC:{Auc(y_hat,test_edge_label)}")
                        _list = list()
                        max_value = 0
                        max_index = 0
                        for j in range(1, 10):
                            _l = list()
                            i = float(j / 10)
                            temp = y_hat.numpy().copy()
                            temp[temp > i] = 1
                            temp[temp <= i] = 0
                            Acc = BinaryAccuracy(i)
                            Spc = BinarySpecificity(i)
                            Pcl = BinaryPrecision(i)
                            if max_value < Acc(y_hat, test_edge_label).item():
                                max_value = Acc(y_hat, test_edge_label).item()
                                max_index = j - 1
                            _l.append(f"阈值为{i}时,Accuracy:{Acc(y_hat, test_edge_label) * 100}%\n")
                            _l.append(f"阈值为{i}时,Specificity:{Spc(y_hat, test_edge_label) * 100}%\n")
                            _l.append(f"阈值为{i}时,Precision:{Pcl(y_hat, test_edge_label) * 100}%\n")
                            _l.append(f"阈值为{i}时,Sensitivity:{sensitivity(temp, test_edge_label) * 100}%\n\n")
                            _list.append(_l)
                        for j in _list:
                            for i in j:
                                txt.writelines(i)
    txt.close()