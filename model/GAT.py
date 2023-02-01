from torch import nn
from torch_geometric.nn import GATConv

class GAT_Net(nn.Module):
    def __init__(self, input_dim,hidden_dim,output_dim, heads=3):
        super(GAT_Net, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim*heads, output_dim)

    def forward(self, Features,A,E):
        Features = self.gat1(Features, A)
        Features = nn.functional.relu(Features)
        Features = nn.functional.dropout(Features, training=self.training)
        Features = self.gat2(Features, A)
        src=Features[E[0]]
        dst=Features[E[1]]
        result=(src*dst).sum(dim=-1)
        result=nn.functional.sigmoid(result)
        return result