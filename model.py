# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 01:02:27 2021

@author: HP
"""

from torch.nn import Module
from layer import GraphNet
from torch.nn.functional import dropout, elu, log_softmax
import torch

class GAT(Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_ratio, nheads, alpha):
        super(GAT, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_ratio = dropout_ratio
        self.n_heads = nheads
        self.alpha = alpha
        
        self.attention = [GraphNet(self.input_dim, self.hidden_dim, self.dropout_ratio, self.alpha, concat = True) for _ in range(nheads)]
        for i, att in enumerate(self.attention):
            self.add_module('attention_{}'.format(i), att)
        self.outer_attention = GraphNet(self.n_heads*self.hidden_dim, self.output_dim, self.dropout_ratio, self.alpha, concat = False)
        
    def forward(self, X, adj):
        
        output = dropout(X, self.dropout_ratio, training = self.training)
        output = torch.cat( [att(output, adj) for att in self.attention], dim=1)
        output = dropout(output, self.dropout_ratio, training = self.training)
        output = elu(self.outer_attention(output, adj))
        output = log_softmax(output, dim=1)
        return output