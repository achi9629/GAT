from torch.nn import Module
from torch.nn.functional import leaky_relu, softmax, dropout, elu
from torch.nn.parameter import Parameter
import torch
from utils import glorot, zeros
from torch.nn import LeakyReLU

class GraphNet(Module):
    
    def __init__(self, input_dim, output_dim, dropout_ratio, alpha, concat = False):
        
        super(GraphNet, self).__init__()
        
        self.input_dim = input_dim
        self.output = output_dim
        self.dropout_ratio = dropout_ratio
        self.alpha = alpha
        self.concat = concat
        
        self.relu = LeakyReLU(self.alpha)
        
        self.weights = Parameter(torch.FloatTensor(input_dim, output_dim))
            
        self.reset_parameters()
        
        self.attention = Parameter(torch.FloatTensor(2*output_dim, 1))
        
    #Initialize weights and bias
    def reset_parameters(self):
        glorot(self.weights)
        # zeros(self.bias)
        
        
        
    def forward(self, h, adj):
        Wh = h@self.weights
        
        e = self._mechanism(Wh)
        
        zero_vec = -9e15*torch.ones_like(e)
        
        attention = torch.where(adj > 0, e, zero_vec)
        attention = softmax(attention, dim = 1)
        attention = dropout(attention, self.dropout_ratio, training = self.training)
        h_prime = attention@Wh
        
        if self.concat == True:
            return elu(h_prime)
        else:
            return h_prime
        
    def _mechanism(self, Wh):
        
        Wh1 = Wh@self.attention[:self.output,:]
        Wh2 = Wh@self.attention[self.output:,:]
        
        e = Wh1 + Wh2.T
        
        return self.relu(e)