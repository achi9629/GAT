# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 18:06:35 2021

@author: HP
"""
import math
import torch


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6./(tensor.size(-1) + tensor.size(-2)))
        tensor.data.uniform_(-stdv, stdv)
        
def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
    
def accuracy(y, y_hat):
    y_hat = y_hat.max(1)[1]  
    sum = y.eq(y_hat).sum()
    acc = sum/y.shape[0]
    return acc*100