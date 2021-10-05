import argparse
import numpy as np
from torch.nn.functional import nll_loss
from torch.optim import Adam
from data_extraction import load_data
from utils import accuracy
from model import GAT
from tqdm import tqdm
import time
import copy

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

args = parser.parse_args()

#Data paths
path = 'data/citeseer/'
dataset = 'citeseer'

#Start timestamp
t0 = time.time() 

#Loading data
adjacency, features, labels, train_idx, val_idx, test_idx = load_data()

#Hyperparameters
input_dim = features.shape[1]
hidden_dim = args.hidden
output_dim = len(np.unique(labels))
learning_rate = args.lr
dropout_ratio = args.dropout
decay = args.weight_decay
epochs = args.epochs
nheads = args.nb_heads
alpha = args.alpha


#Model
model = GAT(input_dim, hidden_dim, output_dim, dropout_ratio, nheads, alpha)

#Loss and Optimizer
optimizer = Adam(model.parameters(), lr = learning_rate, weight_decay = decay)
 

def train():
    
    best_model_wts = copy.deepcopy(model.state_dict()) 
    best_acc = 0.
    epoch_no = 0
    
    for epoch in tqdm(range(epochs)):
        
        #Model to training mode
        model.train()
        
        # Forward pass
        y_hat = model(features, adjacency)
        train_loss = nll_loss(y_hat[train_idx], labels[train_idx])
        
        #Backward and optimize
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        #Training accuracy
        train_acc = accuracy(labels[train_idx], y_hat[train_idx])
        
        #Model to evaluation mode
        model.eval()
        
        #Validation loss and accuracy
        val_loss = nll_loss(y_hat[val_idx], labels[val_idx])
        val_acc = accuracy(labels[val_idx], y_hat[val_idx])
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epoch_no = epoch
            
        print(f'\nEpochs :{epoch+1}/{epochs}, '
              f'Train Loss :{train_loss:.4f}, '
              f'Train_acc :{train_acc:.4f}, '
              f'Val Loss :{val_loss:.4f}, '
              f'Val acc :{val_acc:.4f}')
            
    return best_acc, epoch_no, best_model_wts
    
  
#Training
best_acc, epoch_no, best_model_wts = train()
model.load_state_dict(best_model_wts)

print(f'Best val acc :{best_acc}, Epoch :{epoch_no}')

# #Saving Model
# FILE = "saved_model/model1.pth"
# torch.save(model.state_dict(), FILE)
# print('Model Saved!!')
output = model(features, adjacency)
val_loss = nll_loss(output[test_idx], labels[test_idx])
val_acc = accuracy(labels[test_idx], output[test_idx])
print(f'Test Loss : {val_loss}, Test acc : {val_acc}')

#End timestamp
print(f'Time Taken :{time.time() - t0}')