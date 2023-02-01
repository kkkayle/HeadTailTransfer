import pandas as pd
import numpy as np
import random
import torch
import collections
import copy
class HeadToTailTransfer():
    """
    构造函数:根据相应的文件路径 读取数据到变量中
    """
    def __init__(self,adj_path,rna_feature_path,portein_feature_path) -> None:
        random.seed(123)
        self.adj_path,self.rna_feature_path,self.portein_feature_path=adj_path,rna_feature_path,portein_feature_path
        self.adj_data=pd.read_excel(adj_path,sheet_name=0)
        self.rna_feature_data=pd.read_csv(rna_feature_path)
        self.portein_feature_data=pd.read_csv(portein_feature_path)
    def rebuild(self):
        self.adj_data=pd.read_excel(self.adj_path,sheet_name=0)
        self.rna_feature_data=pd.read_csv(self.rna_feature_path)
        self.portein_feature_data=pd.read_csv(self.portein_feature_path)
        self.encoder()
        self.build_feature_matrix()
        self.split_edge()
        self.build_no_transfer_data()
    """
    根据度对节点进行划分
    将前21个节点划分到high_degree中
    将其余节点划分到low_degree中
    并将其对应的边进行划分(优先级:两点中有一点在high_degree中即划给high_edge)
    """
    def split_edge(self):
        degree_list=list(zip(self.degree_dict.keys(),self.degree_dict.values()))
        from functools import cmp_to_key
        def cmp(a,b):
            if a[1]>b[1]:
                return 1
            elif a[1]<b[1]:
                return -1
            else:
                return 0
        degree_list.sort(key=cmp_to_key(cmp),reverse=True)
        for i in range(len(degree_list)):
            degree_list[i]=degree_list[i][0]
        self.high_degree=degree_list[:21]
        self.low_degree=degree_list[21:] 
        self.high_edge,self.low_edge=list(),list()
        for index,row in self.adj_data.iterrows():
            if row[0] in self.high_degree or row[1] in self.high_degree:
                self.high_edge.append((row[0],row[1],row[2]))
            else:
                self.low_edge.append((row[0],row[1],row[2]))
        
    """
    对节点进行编码并保存 并且收集每个节点的度
    name->code
    row_index:
    0->rna_name
    1->portein_name
    2->label
    编码结果储存在self.code_dict中
    每个节点的度储存在self.degree_dict中
    """
    def encoder(self):
        self.code_dict,self.degree_dict=dict(),collections.defaultdict(int)
        for index,row in self.adj_data.iterrows():
            if row[0] not in self.code_dict:
                self.code_dict[row[0]]=len(self.code_dict)
            if row[1] not in self.code_dict:
                self.code_dict[row[1]]=len(self.code_dict)
            self.adj_data.iloc[index,0]=self.code_dict[row[0]]    
            self.adj_data.iloc[index,1]=self.code_dict[row[1]]
            if row[2]==1:
                self.degree_dict[self.code_dict[row[0]]]+=1
                self.degree_dict[self.code_dict[row[1]]]+=1
    """
    读取数据并通过编码字典(code_dict)的长度和特征向量的维度来构建
    一个合并后的rna-蛋白质特征矩阵
    最后转换为tensor储存到self.tensor_feature_matrix中
    """
    def build_feature_matrix(self):
        rna_feature_length=len(self.rna_feature_data.iloc[1][0].split())
        protein_feature_length=len(self.portein_feature_data.iloc[1][0].split())
        feature_matrix=pd.DataFrame(0,index=range(len(self.code_dict)), columns=(range(rna_feature_length+protein_feature_length)))
        index=0
        while index<len(self.rna_feature_data):#RNA特征矩阵
            key=self.rna_feature_data.iloc[index][0][1:]
            value=self.rna_feature_data.iloc[index+1][0].split()
            if key in self.code_dict:
                feature_matrix.iloc[self.code_dict[key],:rna_feature_length]=pd.Series(value)
            index+=2

        index=0
        while index<len(self.portein_feature_data):#Protein特征矩阵
            key=self.portein_feature_data.iloc[index][0][1:]
            value=self.portein_feature_data.iloc[index+1][0].split()
            if key in self.code_dict:
                feature_matrix.iloc[self.code_dict[key],rna_feature_length:]=pd.Series(value)
            index+=2
        self.tensor_feature_matrix=torch.from_numpy(feature_matrix.values.astype(float)).to(torch.float32).cuda()
    
    """
    构造未进行 数据迁移(Head->Tail)的数据集
    """
    
    def build_no_transfer_data(self):
        self.test_edge_index,self.test_edge_label,self.train_edge_index,self.train_edge_label=[],[],[],[]
        random.shuffle(self.low_edge)
        temp=self.low_edge[:int(len(self.low_edge)*(1/7))]
        for i in temp:
            self.test_edge_index.append((i[0],i[1]))
            self.test_edge_label.append(i[2])
        temp=self.low_edge[int(len(self.low_edge)*(1/7)):]
        for i in temp:
            self.train_edge_index.append((i[0],i[1]))
            self.train_edge_label.append(i[2])
    
    """
    根据传入的edge_index和edge_label构建回边
    """
    def build_back_edge(self,_edge_index,_edge_label,flag=False):
        edge_index=copy.deepcopy(_edge_index)
        edge_label=copy.deepcopy(_edge_label)
        for i in range(len(edge_index)):
            if flag==False:
                if edge_label[i]==1:
                    edge_index.append((edge_index[i][1],edge_index[i][0]))
                    edge_label.append(1)
            else:
                edge_index.append((edge_index[i][1],edge_index[i][0]))
                edge_label.append(edge_label[i])
        return edge_index,edge_label
    """
    根据传入的edge_index和edge_label构建adj_matrix
    """
    def build_adj_matrix(self,edge_index,edge_label):
        adj=[]
        for i in range(len(edge_index)):
            if edge_label[i]==1:
                adj.append(edge_index[i])
        return adj
    
    """
    返回没有进行数据迁移的数据内容    
    """
    def get_no_transfer_data(self):
        train_edge_index,train_edge_label=self.build_back_edge(self.train_edge_index,self.train_edge_label)
        adj=self.build_adj_matrix(train_edge_index,train_edge_label)
        return adj,train_edge_index,train_edge_label,self.test_edge_index,self.test_edge_label
    """
    构建进行数据迁移(Head->Tail)的数据内容
    """
    def build_transfer_data(self,rate):
        self.ts_train_edge_index,self.ts_train_edge_label=copy.deepcopy(self.train_edge_index),copy.deepcopy(self.train_edge_label)
        random.shuffle(self.high_edge)
        #迁移学习
        temp=self.high_edge[:int(len(self.high_edge)*rate)]
        for i in temp:
            self.ts_train_edge_index.append((i[0],i[1]))
            self.ts_train_edge_label.append(i[2])
    """
    返回进行数据迁移的数据内容    
    """
    def get_transfer_data(self):
        train_edge_index,train_edge_label=self.build_back_edge(self.ts_train_edge_index,self.ts_train_edge_label)
        adj=self.build_adj_matrix(train_edge_index,train_edge_label)
        return adj,train_edge_index,train_edge_label,self.test_edge_index,self.test_edge_label
    
if __name__=="__main__":
    pass