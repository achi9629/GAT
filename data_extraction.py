# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 15:11:29 2021

@author: HP
"""
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, diags, eye
import torch

#One hot vector for labels    
def one_hot_vectors(labels):
    classes = np.unique(labels)
    num_classes = len(classes)
    identity = np.eye(num_classes)
    class_dictionary = {j:identity[i] for i,j in enumerate(classes)}
    return np.array(list(map(class_dictionary.get, labels)))


#Normalize features
def normalize_f(feat): 
    rows_sum = np.array(feat.sum(1))
    inverse = np.power(rows_sum, -1).flatten()
    inverse[np.isinf(inverse)] = 0.
    inverse_matrix = diags(inverse)
    feat = np.dot(inverse_matrix, feat)
    return feat


# Normalize adjacency matrix
def normalize_e(adj):
    
    rows_sum = np.array(adj.sum(1))
    inverse = np.power(rows_sum, -0.5).flatten()
    inverse[np.isinf(inverse)] = 0.
    inverse_matrix = diags(inverse)
    adj = np.dot(np.dot(inverse_matrix, adj), inverse_matrix)
    return adj


# #Sparse matrix to sparse tensor
# def sp_mx_sp_tensor(f):
#     sparse_mx = f.tocoo().astype(np.float32)
#     indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#     values = torch.from_numpy(sparse_mx.data)
#     shape = torch.Size(sparse_mx.shape)
#     matrix = torch.sparse.FloatTensor(indices, values, shape)  
#     return matrix

    
#Extracting features, labels, edges
def load_data(path = 'data/cora/', dataset = 'cora'):
    
    print(f'{dataset} dataset doading...')
    
    #Loading cora.content data
    dataset_features = np.genfromtxt(f'{path}{dataset}.content',
                                     dtype= np.dtype(str))
    
    #Extracting Features
    features = csr_matrix(dataset_features[:, 1:-1], dtype = np.float32)

    
    #Extracting Labels
    labels = one_hot_vectors(dataset_features[:,-1])
    
    #Mapping all the paper ID's with their index number
    idx = np.array(dataset_features[:,0], dtype = np.int32)
    idx_map = {j:i for i,j in enumerate(idx)}
    
    #Loading cora.cites data
    edges_without_index = np.genfromtxt(f'{path}{dataset}.cites', 
                                        dtype = np.int32)
    

    #Mapping edges with idx
    edges = np.array(list(map(idx_map.get, edges_without_index.flatten())), 
                     dtype = np.int32).reshape(edges_without_index.shape)
    
    adjacency = coo_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])),
                           shape = (labels.shape[0], labels.shape[0]), 
                           dtype = np.float32)
    
    #Making symmetric adjacency matrix
    adjacency = adjacency + adjacency.T.multiply(adjacency.T > adjacency)
    
    #Normalize Features and Adjacency Matrix
    features = normalize_f(features)
    adjacency = normalize_e(adjacency + eye(adjacency.shape[0]))
    
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adjacency = torch.FloatTensor(np.array(adjacency.todense()))

    
    train_set_idx = torch.tensor(range(140))
    val_set_idx = torch.tensor(range(200, 500))
    test_set_idx = torch.tensor(range(500, 1500))
    print('Dataset Loaded!!')
    
    return adjacency, features, labels, train_set_idx, val_set_idx, test_set_idx


