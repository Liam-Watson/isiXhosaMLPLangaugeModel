from preprocessing import Preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import NGamFFNN
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import time as t

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Check if GPU is available
print("RUNNING ON: ", device)

LEARNING_RATE = 0.01 # Set model learning rate
NUM_EPOCHS = 500 # Number of epochs to train for
BATCH_SIZE = 300 # Batch size for training
context_size = 2 #NGram size
embedding_dim = 20 #Embedding dimension
show_figs = True # Show figures - if true display loss figures

# setup pyplot figures
if show_figs:
    loss_tracker = []
    loss_val_tracker = []
    lossFig = plt.figure(1)
    lossAx = lossFig.add_subplot(1,1,1)

trainPath = "Assignment2Datasets/LMDatasets/nchlt_text.xh.train" # Path to training data
validPath = "Assignment2Datasets/LMDatasets/nchlt_text.xh.valid" # Path to validation data
testPath = "Assignment2Datasets/LMDatasets/nchlt_text.xh.test" # Path to test data  

print("Reading data...")
dataManager = Preprocessing(trainPath, validPath, testPath, context_size) # Create data manager object
print("Data read.")
print("Creating ngrams...")
trainDict = dataManager.getTrainDict() # Get training dictionary
validDict = dataManager.getValidDict() # Get validation dictionary
testDict = dataManager.getTestDict() # Get test dictionary

trainTwoGrams = dataManager.getTrainTwoGrams() # Get training ngrams
validTwoGrams = dataManager.getValidTwoGrams() # Get validation ngrams
testTwoGrams = dataManager.getTestTwoGrams() # Get test ngrams

vocab = list(trainDict.keys()) # Get vocabulary for training dataset
vocab_val = list(validDict.keys()) # Get vocabulary for validation dataset
vocab_test = list(testDict.keys()) # Get vocabulary for test dataset

word_idx = {word: i for i, word in enumerate(vocab)} # Define dict for word -> vocab index mapping for training data
word_idx_val = {word: i for i, word in enumerate(vocab_val)} # Define dict for word -> vocab index mapping for validation data
word_idx_test = {word: i for i, word in enumerate(vocab_test)} # Define dict for word -> vocab index mapping for test data

trainDataLoader = DataLoader(trainTwoGrams, batch_size=BATCH_SIZE, shuffle=True) # Create data loader for training data
validDataLoader = DataLoader(validTwoGrams, batch_size=BATCH_SIZE, shuffle=True) # Create data loader for validation data
testDataLoader = DataLoader(testTwoGrams, batch_size=1, shuffle=True) # Create dataloader for test data

print("Done processing data.")

print("Creating model...")
model = NGamFFNN(len(vocab), embedding_dim, context_size) # init model
model.to(device) # Move model to GPU if available

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE) # init optimizer to Stochastic gradient descent
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # init optimizer to Adam
# optimizer = optim.Adagrad(model.parameters(), lr=LEARNING_RATE) # init optimizer to Adagrad
# optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE) # init optimizer to RMSprop

loss_function = nn.NLLLoss() # init loss function for Negative Log Likelihood

'''batchWordToIdx converts a batch of context and target tuples to their index representation in the vocabulary'''
def batchWordToIdx(context, target, word_idx):
    tuple1, tuple2 = context[0], context[1] # Get context tuples
    inptIdx = [] # Init input index list
    targetIdx = [] # Init target index list
    for x in range(len(tuple1)):
        inptIdx.append([word_idx[tuple1[x]], word_idx[tuple2[x]]]) # Convert context tuples to index representation
        targetIdx.append(word_idx[target[x]]) # Convert target to index representation
    return inptIdx, targetIdx

print("Training model...")
# Train model
for epoch in range(NUM_EPOCHS):
    total_loss = 0 # Init total loss for epoch
    print("Epoch: ", epoch)
    st = t.time() # Start timer for epoch
    model.train() # Set model to training mode
    for batch_idx, dataItem  in enumerate((trainDataLoader)):
        context, target = dataItem  #extract context and target from dataloader item 
        inptIdx, targetIdx = batchWordToIdx(context, target, word_idx) # Convert batch to index representation
        context_idxs = torch.tensor(inptIdx, dtype=torch.long) 
        targets = torch.tensor(targetIdx, dtype=torch.long)
        context_idxs = context_idxs.to(device)

        model.zero_grad() #Zero gradients to prevent back prop through accumulated gradients

        probs = model(context_idxs) #Get prediction probabilities via forward pass
        
        loss = loss_function(probs, targets.to(device)) # Calculate loss
        loss.backward() # Backpropagate loss
        optimizer.step() 

        total_loss += loss.item() # Add loss to total loss for epoch
        
        # Print progress
        if batch_idx % 100 == 0:
            with torch.no_grad():
                ppl_train = torch.exp(F.cross_entropy((probs), targets.to(device))).item() # Calculate perplexity on training data   
                accCounter = 0 # Init accuracy counter
                probsMaxIndex = torch.max(probs, dim=1)[1] # Get index of max probability for each prediction
                for s in range(BATCH_SIZE):
                    if target[s] == vocab[probsMaxIndex[s]]:
                        accCounter += 1 # Increment accuracy counter if prediction is correct

                # print(F.cross_entropy((probs), targets.to(device)))
                # print(torch.exp(F.cross_entropy((probs), targets.to(device))))
                # print(torch.exp(F.cross_entropy((probs), targets.to(device))).item() )
                print("EPOCH: ", epoch, "| batch:", batch_idx, "/" , len(trainDataLoader) ,  "| Train Loss:", loss.item(), "| Train PPL:", ppl_train, "| Train Accuracy:", accCounter/BATCH_SIZE)   
                if show_figs:
                        loss_tracker.append(loss.item()) # Add loss to loss tracker for figure display
                        lossAx.clear()
                        lossAx.plot(loss_tracker, label="Loss Train")
                        lossAx.legend()
                        lossFig.canvas.draw_idle()
                        lossFig.canvas.flush_events()
                        lossFig.show()
    model.eval() # Set model to evaluation mode for validation data
    total_loss_val = 0 # Init total loss for validation
    et = t.time() # End timer for epoch
    elapsed = et - st # Calculate elapsed time for epoch
    ppl = 0 # Init perplexity for for validation

    for batch_idx_val, dataItem_val  in enumerate((validDataLoader)):
        context, target = dataItem_val   
        inptIdx, targetIdx = batchWordToIdx(context, target, word_idx_val) # Convert batch to index representation
        context_idxs = torch.tensor(inptIdx, dtype=torch.long) # Convert context to tensor
        targets = torch.tensor(targetIdx, dtype=torch.long) # Convert target to tensor
        context_idxs = context_idxs.to(device) # Move context to GPU if available
        
        probs_val = model(context_idxs) #Get prediction probabilities via forward pass
        accCounter = 0 # Init accuracy counter
        probsMaxIndex = torch.max(probs_val, dim=1)[1] # Get index of max probability for each prediction
        for s in range(len(target)):
            # print(len(target))
            # print(target[s])
            # print(probsMaxIndex[s])
            # print(vocab[probsMaxIndex[s]])
            if target[s] == vocab[probsMaxIndex[s]]:
                accCounter += 1 # Increment accuracy counter if prediction is correct
                # print(target[s], vocab[probsMaxIndex[s]])
        ppl = torch.exp(F.cross_entropy((probs_val), targets.to(device))).item()  # Calculate perplexity on validation data
        # print(ppl)
        # print("| Valid Accuracy:", accCounter/len(validDataLoader))
        total_loss_val += loss_function(probs_val, targets.to(device)).item() # Add loss to total loss for epoch
    
    print("END OF EPOCH", epoch, "| time:", elapsed, "| Valid Loss:", total_loss_val/len(validDataLoader), "| Valid PPL:", ppl)

torch.save(model.state_dict(), str(NUM_EPOCHS) + "_" + str(LEARNING_RATE)+"_bn.pth") # Save model


'''Test model on test data'''
model.eval() # Set model to evaluation mode for test data
ppl_test = 0 # Init perplexity for test data
total_loss_test = 0 # Init total loss for test data
for batch_idx, dataItem  in enumerate(tqdm(testDataLoader)):
        context, target = dataItem # Extract context and target from dataloader item
        inptIdx, targetIdx = batchWordToIdx(context, target, word_idx) # Convert batch to index representation
        context_idxs = torch.tensor(inptIdx, dtype=torch.long) # Convert context to tensor
        targets = torch.tensor(targetIdx, dtype=torch.long) # Convert target to tensor
        context_idxs = context_idxs.to(device) # Move context to GPU if available

        probs = model(context_idxs) #Get prediction probabilities via forward pass
        
        loss = loss_function(probs, targets.to(device)) # Calculate loss

        total_loss_test += loss.item() # Add loss to total loss for epoch

print("END OF EPOCH", epoch, "| time:", elapsed, "| Valid Loss:", total_loss_test/len(testDataLoader), "| Valid PPL:", ppl/len(testDataLoader))
