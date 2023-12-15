import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
#from algorithms.CM_HGCN.aggregator import LocalAggregator, GlobalAggregator,GNN,LocalAggregator_mix





def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CombineGraph(Module):
    def __init__(self, num_nodes, n_category, category, lr, batch_size, l2, dim, seed):
        super(CombineGraph, self).__init__()
        self.seed = seed
        init_seed(self.seed)
        
        self.batch_size = batch_size
 #      self.num_node = num_node
        self.num_total = num_nodes
        self.dim = dim # hidden size
        self.dropout_local = 0
        self.dropout_global = 0
        self.hop = 1
        self.sample_num = 12
        self.alpha = 0.2
        self.n_category = n_category
        self.category = category
        self.lr = lr
        self.l2 = l2
        self.lr_dc = 0.1
        self.lr_dc_step = 3
        
        
        # Aggregator
        self.local_agg_1 = LocalAggregator(self.dim, self.alpha, self.seed)
       
        self.gnn = GNN(self.dim)
        
        self.local_agg_mix_1 = LocalAggregator(self.dim, self.alpha, self.seed)
   
        # Item representation & Position representation
        self.embedding = nn.Embedding(self.num_total, self.dim)
        self.pos = nn.Embedding(200, self.dim)
        

        # Parameters_1
        self.w_1 = nn.Parameter(torch.Tensor(3 * self.dim, 2*self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(2*self.dim, 1))
        self.glu1 = nn.Linear(2*self.dim, 2*self.dim)
        self.glu2 = nn.Linear(2*self.dim, 2*self.dim, bias=False)
   #     self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)
        

        
 
       # self.aaa = Parameter(torch.Tensor(1))
        self.bbb = Parameter(torch.Tensor(1))
        self.ccc = Parameter(torch.Tensor(1))

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay= self.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size= self.lr_dc_step, gamma= self.lr_dc)
        self.reset_parameters()
    
        item = []
        for x in range(1, self.num_total + 1 - n_category):
            item += [category[x]]
        item = np.asarray(item)  
        self.item =  trans_to_cuda(torch.Tensor(item).long())

    def reset_parameters(self):
        init_seed(self.seed)
        
        
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden1, hidden2, hidden1_mix, hidden2_mix, mask):
        init_seed(self.seed)
        
              
        hidden1 = hidden1 + hidden1_mix * self.bbb
        hidden2 = hidden2 + hidden2_mix * self.ccc
        hidden = torch.cat([hidden1, hidden2],-1)
        
        mask = mask.float().unsqueeze(-1)
        batch_size = hidden1.shape[0]
        len = hidden1.shape[1]
        
        pos_emb = self.pos.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        # Error is here for eCommerce dataset.....
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)

        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)

        b = self.embedding.weight[1:self.num_total-self.n_category+1]  # n_nodes x latent_size
        item_category = self.embedding(self.item)     #n*d
        
        t = torch.cat([b,item_category],-1)
        scores = torch.matmul(select, t.transpose(1, 0))

        return scores

    def forward(self, inputs, adj, mask_item, item, items_ID, adj_ID, total_items, total_adj):
        init_seed(self.seed)
        
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        
        hidden1 = self.embedding(inputs)
        hidden2 = self.embedding(items_ID)
        hidden_mix = self.embedding(total_items)
        
        # local
        hidden1 = self.local_agg_1(hidden1, adj, mask_item)
  #      hidden2 = self.local_agg_2(hidden2, adj_ID, mask_item)
        hidden2 = self.gnn(adj_ID,hidden2)
        
        hidden_mix = self.local_agg_mix_1(hidden_mix, total_adj, mask_item)
    #    hidden_mix = self.local_agg_mix_2(hidden_mix, total_adj, mask_item)
    #    hidden_mix = self.local_agg_mix_3(hidden_mix, total_adj, mask_item)

        # combine
        hidden1 = F.dropout(hidden1, self.dropout_local, training=self.training)
        hidden2 = F.dropout(hidden2, self.dropout_local, training=self.training)
        hidden_mix = F.dropout(hidden_mix, self.dropout_local, training=self.training)

        return hidden1, hidden2, hidden_mix
    
    
class LocalAggregator(nn.Module):
    def __init__(self, dim, alpha, seed):
        super(LocalAggregator, self).__init__()
        
        self.seed = seed
        init_seed(self.seed)
        
        self.dim = dim
        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, hidden, adj, mask_item=None):
        init_seed(self.seed)
        
        
        h = hidden
        batch_size = h.shape[0]
        N = h.shape[1]

        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim)
                   * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)
       # a_input = torch.cat([h.repeat(1, 1, N).view(batch_size, N * N, self.dim), h.repeat(1, N, 1)],-1).view(batch_size, N, N, 2*self.dim)

        e_0 = torch.matmul(a_input, self.a_0)
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)
        e_3 = torch.matmul(a_input, self.a_3)

        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(batch_size, N, N)

        mask = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = torch.softmax(alpha, dim=-1)

        output = torch.matmul(alpha, h)
        return output    
    
class GNN(Module):
    def __init__(self, hidden_size, step=1, seed = 2000):
        super(GNN, self).__init__()
        
        
        self.seed = seed
        init_seed(self.seed)
        
        
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        
        self.linear_in1= nn.Linear(self.gate_size, self.gate_size, bias=True)
        self.linear_in2= nn.Linear(self.gate_size, self.gate_size, bias=True)

    def GNNCell(self, A, hidden):
        init_seed(self.seed)
        
        
        
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        
        test = torch.cat([inputs,hidden], 2)
        test1 = self.linear_in1(test)
        test2 = self.linear_in2(test)
        
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        
        gi = gi+test1
        gh = gh+test2
        
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        init_seed(self.seed)
        
        
        
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden    
    
    
    
    
def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable
    
    
def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable    
    




