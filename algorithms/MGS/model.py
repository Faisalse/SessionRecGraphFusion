import datetime
import numpy as np
from tqdm import tqdm
from algorithms.MGS.aggregator import *
from torch.nn import Module
import torch.nn.functional as F

# device conf
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

class CombineGraph(Module):
    def __init__(self, num_node, lr, l2, embedding, dropout):
        super(CombineGraph, self).__init__()
        self.num_node = num_node
        self.dim = embedding
        self.hop = 1
        self.mu = 0.01
        self.alpha = 0.2
        self.dropout_attribute = dropout
        self.lr = lr
        self.l2 = l2
        self.lr_dc = 0.1
        self.lr_dc_step = 3
        self.dropout_score = 0.2
        self.temp = 0.1

        # Aggregator
        self.attribute_agg = AttributeAggregator(self.dim, self.alpha, self.dropout_attribute)
        self.local_agg = nn.ModuleList()
        self.mirror_agg = nn.ModuleList()
        for i in range(self.hop):
            agg = LocalAggregator(self.dim, self.alpha)
            self.local_agg.append(agg)
            agg = MirrorAggregator(self.dim)
            self.mirror_agg.append(agg)

        # high way net
        self.highway = nn.Linear(self.dim * 2, self.dim, bias=False)

        # embeddings
        self.embedding = nn.Embedding(num_node, self.dim)
        self.pos_embedding = nn.Embedding(200, self.dim)

        # Parameters
        self.w = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim, bias=False)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.glu3 = nn.Linear(self.dim, self.dim)
        self.gate = nn.Linear(self.dim * 2, self.dim, bias=False)

        # loss function
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_dc_step, gamma=self.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_score(self, hidden, pos_emb, h_mirror, h_local, mask, item_weight):
        hm = h_mirror
        hl = h_local.unsqueeze(1).repeat(1, hidden.size(1), 1)
        hp = hidden + pos_emb
        nh = torch.sigmoid(self.glu1(hp) + self.glu2(hm) + self.glu3(hl))
        beta = torch.matmul(nh, self.w)
        beta = beta * mask
        zg = torch.sum(beta * hp, 1)
        gf = torch.sigmoid(self.gate(torch.cat([zg, h_local], dim=-1))) * self.mu
        zh = gf * h_local + (1 - gf) * zg
        zh = F.dropout(zh, self.dropout_score, self.training)
        scores = torch.matmul(zh, item_weight.transpose(1, 0))

        return scores

    def similarity_loss(self, hf, hf_SSL, simi_mask):
        h1 = hf
        h2 = hf_SSL
        h1 = h1.unsqueeze(2).repeat(1, 1, h1.size(1), 1)
        h2 = h2.unsqueeze(1).repeat(1, h2.size(1), 1, 1)
        hf_similarity = torch.sum(torch.mul(h1, h2), dim=3) / self.temp
        loss = -torch.log(torch.softmax(hf_similarity, dim=2) + 1e-8)
        simi_mask = simi_mask == 1
        loss = torch.sum(loss * simi_mask, dim=2)
        loss = torch.sum(loss, dim=1)
        loss = torch.mean(loss)

        return loss

    def compute_score_and_ssl_loss(self, h, h_local, h_mirror, mask, hf_SSL1, hf_SSL2, simi_mask):
        mask = mask.float().unsqueeze(-1)
        batch_size = h.shape[0]
        len = h.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        b = self.embedding.weight[1:]

        simi_loss = self.similarity_loss(hf_SSL1, hf_SSL2, simi_mask)
        scores = self.compute_score(h, pos_emb, h_mirror, h_local, mask, b)

        return simi_loss, scores

    def forward(self, inputs, adj, last_item_mask, as_items, as_items_SSL, simi_mask):
        # preprocess
        mask_item = inputs != 0
        attribute_num = len(as_items)
        h = self.embedding(inputs)
        h_as = []
        h_as_SSL = []
        as_mask = []
        as_mask_SSL = []
        for k in range(attribute_num):
            nei = as_items[k]
            nei_SSL = as_items_SSL[k]
            nei_emb = self.embedding(nei)
            nei_emb_SSL = self.embedding(nei_SSL)
            h_as.append(nei_emb)
            h_as_SSL.append(nei_emb_SSL)
            as_mask.append(as_items[k] != 0)
            as_mask_SSL.append(as_items_SSL[k] != 0)

        # attribute
        hf_1, hf_2, hf = self.attribute_agg(h, h_as, as_mask, h_as_SSL, as_mask_SSL)
        # GNN
        x = h
        mirror_nodes = hf
        for i in range(self.hop):
            # aggregate neighbor info
            x = self.local_agg[i](x, adj, mask_item)
            x, mirror_nodes = self.mirror_agg[i](mirror_nodes, x, mask_item)

        # highway
        g = torch.sigmoid(self.highway(torch.cat([h, x], dim=2)))
        x_dot = g * h + (1 - g) * x

        # hidden
        hidden = x_dot

        # local representation
        h_local = torch.masked_select(x_dot, last_item_mask.unsqueeze(2).repeat(1, 1, x_dot.size(2))).reshape(mask_item.size(0), -1)

        # mirror
        h_mirror = mirror_nodes

        # calculate score
        simi_loss, scores = self.compute_score_and_ssl_loss(hidden, h_local, h_mirror, mask_item, hf_1, hf_2, simi_mask)

        return simi_loss, scores


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.to(device)
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable




def adjust_learning_rate(optimizer, decay_rate, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * decay_rate
    lr * decay_rate


