import random
import pandas as pd
import numpy as np
import networkx as nx
import collections
import torch
from random import shuffle
import warnings
from sklearn.model_selection import  StratifiedKFold
from GraphSage import GraphSage_Net
import torch
from torchmetrics import AUROC
from torchmetrics.classification import BinaryAccuracy,BinarySpecificity,BinaryPrecision
from Sensitivity import sensitivity
from transfer import transfer
import copy
warnings.filterwarnings("ignore")

RPI_list=["RPI369"]
for RPI in RPI_list:
    txt_name=RPI+'迁移结果GraphSage.txt'
    txt = open(txt_name, mode = 'w')
    _rate = [0.01, 0.05, 0.1, 0.2, 0.3]
    edge_list = [5, 10, 15]
    file_name=["NPInter2"]
    for file in file_name:
        for rate in _rate:
            for edge_num in edge_list:
                random.seed(123)
                data=transfer(adj_path="C:\\Users\\Administrator\\Desktop\\data\\source_database_data\\"+RPI+".xlsx",rna_feature_path="C:\\Users\\Administrator\\Desktop\\data\\lncRNA_3_mer\\"+RPI+"\\lncRNA_3_mer.txt",protein_feature_path="C:\\Users\\Administrator\\Desktop\\data\\protein_2_mer\\"+RPI+"\\protein_2_mer.txt",inter2_adj_path="C:\\Users\\Administrator\\Desktop\\data\\source_database_data\\"+file+".xlsx",inter2_rna_feature_path=r"C:\Users\Administrator\Desktop\data\lncRNA_3_mer\\"+file+"\\lncRNA_3_mer.txt",inter2_protein_feature_path=r"C:\Users\Administrator\Desktop\data\protein_2_mer\\"+file+"\\protein_2_mer.txt")
                for times in range(5):
                    txt.writelines(str(rate)+' '+str(times+1)+str(edge_num)+"\n")
                    data.rebuild()
                    data.build_transfer_data(rate=rate,edge_num=edge_num,add_adj=False)
                    adj,train_edge_index,train_edge_label,test_edge_index,test_edge_label=data.get_transfer_data()
                    tensor_feature_matrix=copy.deepcopy(data.tensor_feature_matrix)
                    
                    """
                    以下是train-test数据格式的转换
                    """
                    adj=torch.tensor(adj).to(torch.long).T.cuda()
                    train_edge_index=torch.tensor(train_edge_index).to(torch.long).T.cuda()
                    train_edge_label=torch.tensor(train_edge_label).to(torch.long).cuda()
                    test_edge_index=torch.tensor(test_edge_index).to(torch.long).T.cuda()
                    test_edge_label=torch.tensor(test_edge_label).to(torch.long).cuda()
                    
                    epoch=1200
                    
                    model=GraphSage_Net(len(tensor_feature_matrix[0]),64,16).cuda()
                    model.train()
                    opt = torch.optim.Adam(params=model.parameters(), lr=0.01)
                    loss_fn = torch.nn.BCELoss().cuda()
                    
                    
                    for i in range(epoch):
                        y_hat=model(tensor_feature_matrix, adj,train_edge_index)
                        loss=loss_fn(y_hat.to(torch.float32),train_edge_label.to(torch.float32))
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                    model.eval()

                    Auc=AUROC(task="binary")



                    with torch.no_grad():
                        y_hat=model(tensor_feature_matrix, adj,test_edge_index)
                        test_edge_label=test_edge_label.cpu()
                        y_hat=y_hat.cpu()
                        txt.writelines(f"AUC,{Auc(y_hat,test_edge_label)}\n")
                        print({Auc(y_hat,test_edge_label)})
                        #print(f"{epoch}轮训练\nAUC:{Auc(y_hat,test_edge_label)}")
                        _list=list()
                        max_value=0
                        max_index=0
                        for j in range(1,10):
                            _l=list()
                            i=float(j/10)
                            temp=y_hat.numpy().copy()
                            temp[temp>i]=1
                            temp[temp<=i]=0
                            Acc = BinaryAccuracy(i)
                            Spc = BinarySpecificity(i)
                            Pcl = BinaryPrecision(i)
                            if max_value<Acc(y_hat,test_edge_label).item():
                                max_value=Acc(y_hat,test_edge_label).item()
                                max_index=j-1
                            _l.append(f"value,{i}\n")
                            _l.append(f"Accuracy,{Acc(y_hat,test_edge_label)}\n")
                            _l.append(f"Specificity,{Spc(y_hat,test_edge_label)}\n")
                            _l.append(f"Precision,{Pcl(y_hat,test_edge_label)}\n")
                            _l.append(f"Sensitivity,{sensitivity(temp,test_edge_label)}\n\n")
                            _list.append(_l)
                        for i in _list[max_index]:
                            txt.writelines(i)

    txt.close()