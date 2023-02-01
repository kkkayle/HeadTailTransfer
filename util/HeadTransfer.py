import pandas as pd
import numpy as np
import random
import torch
import collections
import copy
class HeadTransfer():
    """
    构造函数:保存文件路径并读取数据
    """
    def __init__(self,adj_path,rna_feature_path,protein_feature_path,inter2_adj_path="",inter2_rna_feature_path="",inter2_protein_feature_path="") -> None:
        random.seed(123)
        self.adj_path,self.rna_feature_path,self.protein_feature_path=adj_path,rna_feature_path,protein_feature_path
        self.inter2_adj_path,self.inter2_rna_feature_path,self.inter2_protein_feature_path=inter2_adj_path,inter2_rna_feature_path,inter2_protein_feature_path
        self.read_data()
    """
    重新构建数据 可以进行多轮实验
    """
    def rebuild(self):
        self.read_data()
        self.encoder()
        self.build_feature_matrix()
        if self.inter2_adj_path !="":
            self.split_degree()
        self.build_no_transfer_data()
    """
    根据相应的文件路径 读取数据到变量中
    """
    def read_data(self):
        self.adj_data=pd.read_excel(self.adj_path,sheet_name=0)
        self.rna_feature_data=pd.read_csv(self.rna_feature_path)
        self.protein_feature_data=pd.read_csv(self.protein_feature_path)
        if self.inter2_adj_path !="":
            self.inter2_adj_data=pd.read_excel(self.inter2_adj_path,sheet_name=0)
            self.inter2_rna_feature_data=pd.read_csv(self.inter2_rna_feature_path)
            self.inter2_protein_feature_data=pd.read_csv(self.inter2_protein_feature_path)

        

    def split_degree(self):
        self.degree_list=list(zip(self.degree_dict.keys(),self.degree_dict.values()))
        from functools import cmp_to_key
        def cmp(a,b):
            if a[1]>b[1]:
                return 1
            elif a[1]<b[1]:
                return -1
            else:
                return 0
        self.degree_list.sort(key=cmp_to_key(cmp),reverse=True)
        for i in range(len(self.degree_list)):
            self.degree_list[i]=self.degree_list[i][0]
        
        
    """
    对节点进行编码并保存 并且收集每个节点的度
    name->code
    row_index:
    0->rna_name
    1->protein_name
    2->label
    编码结果储存在self.code_dict中
    每个节点的度储存在self.degree_dict中
    """
    def encoder(self):
        self.code_dict,self.degree_dict=dict(),collections.defaultdict(int)
        #对原数据进行编码
        for index,row in self.adj_data.iterrows():
            if row[0] not in self.code_dict:
                self.code_dict[row[0]]=len(self.code_dict)
            if row[1] not in self.code_dict:
                self.code_dict[row[1]]=len(self.code_dict)
            self.adj_data.iloc[index,0]=self.code_dict[row[0]]    
            self.adj_data.iloc[index,1]=self.code_dict[row[1]]
        #对迁移的数据进行编码和度的统计
        if self.inter2_adj_path !="":
            for index,row in self.inter2_adj_data.iterrows():
                if row[0] not in self.code_dict:
                    self.code_dict[row[0]]=len(self.code_dict)
                if row[1] not in self.code_dict:
                    self.code_dict[row[1]]=len(self.code_dict)
                self.inter2_adj_data.iloc[index,0]=self.code_dict[row[0]]    
                self.inter2_adj_data.iloc[index,1]=self.code_dict[row[1]]
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
        protein_feature_length=len(self.protein_feature_data.iloc[1][0].split())
        feature_matrix=pd.DataFrame(0,index=range(len(self.code_dict)), columns=(range(rna_feature_length+protein_feature_length)))
        index=0
        while index<len(self.rna_feature_data):#RNA特征矩阵
            key=self.rna_feature_data.iloc[index][0][1:]
            value=self.rna_feature_data.iloc[index+1][0].split()
            if key in self.code_dict:
                feature_matrix.iloc[self.code_dict[key],:rna_feature_length]=pd.Series(value)
            index+=2

        index=0
        while index<len(self.protein_feature_data):#Protein特征矩阵
            key=self.protein_feature_data.iloc[index][0][1:]
            value=self.protein_feature_data.iloc[index+1][0].split()
            if key in self.code_dict:
                feature_matrix.iloc[self.code_dict[key],rna_feature_length:]=pd.Series(value)
            index+=2

        if self.inter2_adj_path !="":
            index=0
            while index<len(self.inter2_rna_feature_data):#RNA特征矩阵
                key=self.inter2_rna_feature_data.iloc[index][0][1:]
                value=self.inter2_rna_feature_data.iloc[index+1][0].split()
                if key in self.code_dict:
                    feature_matrix.iloc[self.code_dict[key],:rna_feature_length]=pd.Series(value)
                index+=2

            index=0
            while index<len(self.inter2_protein_feature_data):#Protein特征矩阵
                key=self.inter2_protein_feature_data.iloc[index][0][1:]
                value=self.inter2_protein_feature_data.iloc[index+1][0].split()
                if key in self.code_dict:
                    feature_matrix.iloc[self.code_dict[key],rna_feature_length:]=pd.Series(value)
                index+=2

        self.tensor_feature_matrix=torch.from_numpy(feature_matrix.values.astype(float)).to(torch.float32).cuda()

    """
    构造了没有进行 数据迁移(Head->Tail)的数据集
    """
    
    def build_no_transfer_data(self):
        self.test_edge_index,self.test_edge_label,self.train_edge_index,self.train_edge_label=[],[],[],[]
        data=np.array(self.adj_data).tolist()
        random.shuffle(data)
        #test
        self.test_edge_index=list()
        self.test_edge_label=list()
        temp=data[:int(len(data)*0.2)]
        for i in temp:
            self.test_edge_index.append((i[0],i[1]))
            self.test_edge_label.append(i[2])
        self.train_edge_index=list()
        self.train_edge_label=list()    
        temp=data[int(len(data)*0.2):]
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
    def build_transfer_data(self,rate,edge_num,add_adj=False):
        num_dict=collections.defaultdict(int)
        self.add_adj=add_adj
        self.ts_train_edge_index,self.ts_train_edge_label=copy.deepcopy(self.train_edge_index),copy.deepcopy(self.train_edge_label)
        degree_list=self.degree_list[:int(len(self.degree_list)*rate)]
        for index,row in self.inter2_adj_data.iterrows():
            if row[0] in degree_list or row[1] in degree_list:
                if num_dict[row[0]]<=edge_num and num_dict[row[1]]<=edge_num:
                    self.ts_train_edge_index.append((row[0],row[1]))
                    self.ts_train_edge_label.append(row[2])
                    num_dict[row[0]]+=1
                    num_dict[row[1]]+=1
    """
    返回进行数据迁移的数据内容    
    """
    def get_transfer_data(self):
        if self.add_adj ==True:
            train_edge_index,train_edge_label=self.build_back_edge(self.ts_train_edge_index,self.ts_train_edge_label)
            adj=self.build_adj_matrix(train_edge_index,train_edge_label)
            return adj,train_edge_index,train_edge_label,self.test_edge_index,self.test_edge_label
        else:
            train_edge_index,train_edge_label=self.build_back_edge(self.ts_train_edge_index,self.ts_train_edge_label)
            temp1,temp2=self.build_back_edge(self.train_edge_index,self.train_edge_label)
            adj=self.build_adj_matrix(temp1,temp2)
            return adj,train_edge_index,train_edge_label,self.test_edge_index,self.test_edge_label
        
    
if __name__=="__main__":
        pass
        