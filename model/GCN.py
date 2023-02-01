from torch_geometric.nn import GCNConv
from torch import nn

class GCN_Net(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(GCN_Net, self).__init__()
        self.GCN1=GCNConv(input_dim,hidden_dim)
        self.GCN2=GCNConv(hidden_dim,output_dim)
    def forward(self,Features,A,E):
        Features=self.GCN1(Features,A)    
        Features=nn.functional.relu(Features) 
        Features=nn.functional.dropout(Features,training=self.training)
        Features=self.GCN2(Features,A)   
        src=Features[E[0]]
        dst=Features[E[1]]
        result=(src*dst).sum(dim=-1)
        result=nn.functional.sigmoid(result)
        return result
    