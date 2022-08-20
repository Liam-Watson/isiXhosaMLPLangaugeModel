import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


'''Feed forward NGram neural network'''
class NGamFFNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGamFFNN, self).__init__()
        # origSize = context_size * embedding_dim
        
        self.embeddingLayer = nn.Embedding(vocab_size, embedding_dim) # Embedding layer
        self.fc1 = nn.Linear(context_size * embedding_dim, 128) # First fully connected layer
        self.fc4 = nn.Linear(128, vocab_size) # Output layer

        self.bn1 = nn.BatchNorm1d(128)
        # self.bn2 = nn.BatchNorm1d(int(origSize*3))
        self.dropout = nn.Dropout(0.3)

    def forward(self, inputs):
        embeds = self.embeddingLayer(inputs).view((inputs.shape[0],-1))#.view((inputs.shape[0], -1)) #  Get embeddings for each word in the context
        x = F.relu(self.bn1(self.fc1(embeds))) # Pass through first fully connected layer
        x = self.dropout(x)
        x = self.fc4(x) # Pass through output layer
        probs = F.log_softmax(x, dim=1) # Get probabilities 
        # probs = x
        return probs