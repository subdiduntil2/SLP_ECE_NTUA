import os
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder 
import torch
from torch.utils.data import DataLoader
from nltk import word_tokenize
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score
from training import torch_train_val_split
from config import EMB_PATH, MAX_LENGTH, DATASET
from dataloading import SentenceDataset
from models import BaselineDNN,LSTM
from attention import SimpleSelfAttentionModel, MultiHeadAttentionModel, TransformerEncoderModel
from training import train_dataset, eval_dataset
from early_stopper import EarlyStopper
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.100d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 100

EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 20
# options: "MR", "Semeval2017A"s

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# load the raw data
length_array = []
def getLengths():
    length_array = []
    X_train, y_train, X_test, y_test = load_Semeval2017A()
    for example in X_train:
        length_array.append(len(word_tokenize(example)))
    X_train, y_train, X_test, y_test = load_MR()
    for example in X_train:
        length_array.append(len(word_tokenize(example)))
    mean_length = int(np.mean(np.array(length_array)))
    max_length = np.max(np.array(length_array))
    length_final = int((mean_length+max_length)/2)
    print("mean length for datasets is:",mean_length)
    print()
    print("max length for datasets is" , DATASET, "is:",max_length)
    print()
    print("length final is: ",length_final)
    return mean_length, max_length, length_final
    
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

mean_length_, max_length_,length_final_= getLengths()

# convert data labels from strings to integers
le = LabelEncoder()
y_train = le.fit_transform(y_train)  # EX1
y_test = le.fit_transform(y_test)  # EX1
n_classes = le.classes_.size  # EX1 - LabelEncoder.classes_.size

print("\n### 10 first labels-numbers match: \n")
for number in y_train[10:20]:
    print(le.classes_[number], number)
print()
    
# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)
print("initialized train & test sets")
for i in range(5):
    print(train_set.__getitem__(i))
    
print()
print("\n### 10 first examples of tokenization: ###\n")
for i, example in enumerate(train_set.data[:10]):
    print(f"{i}: {example}\n")
    
# print()
# print("\n### 5 first examples of word encoding ###\n")
# for i, item in enumerate(train_set.data[:5]):
#     print(f"{i}: initial: {item}\n")
#     print(f"\n")
#     (example, label, length) = train_set[i]
#     print(f"   example: {example}")
#     print(f"   label: {label}")
#     print(f"   length: {length}\n")

# EX4 - Define our PyTorch-based DataLoader
# train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)  # EX7
test_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,num_workers=4)  # EX7
train_loader,val_loader = torch_train_val_split(train_set, batch_train=BATCH_SIZE, batch_eval=BATCH_SIZE)

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################
model_0 = BaselineDNN(output_size=n_classes, embeddings=embeddings, trainable_emb=EMB_TRAINABLE) #EX8 (from part1)
model_1 = LSTM(output_size=n_classes,embeddings=embeddings,hidden_size=EMB_DIM,trainable_emb=EMB_TRAINABLE,bidirectional=False)
model_2 = LSTM(output_size=n_classes,embeddings=embeddings,hidden_size=EMB_DIM,trainable_emb=EMB_TRAINABLE,bidirectional=True)
model_3 = SimpleSelfAttentionModel(output_size=n_classes,embeddings=embeddings,max_length=MAX_LENGTH)
model_4 = MultiHeadAttentionModel(output_size=n_classes,embeddings=embeddings,max_length=MAX_LENGTH)
model_5 = TransformerEncoderModel(output_size=n_classes,embeddings=embeddings,max_length=MAX_LENGTH)
model = model_5

# move the mode weight to cpu or gpu
model.to(DEVICE)

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
if (n_classes == 2):
    criterion = torch.nn.BCEWithLogitsLoss()  # EX8
else:
    criterion = torch.nn.CrossEntropyLoss()
parameters = model.parameters()  # EX8
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters, lr=0.001)  # EX8

#############################################################################
# Training Pipeline
#############################################################################
train_loss_all=[]
valid_loss_all=[]
test_loss_all=[]
#for acc
train_acc_all=[]
valid_acc_all=[]
test_acc_all=[]
#for recall
train_recall_all=[]
valid_recall_all=[]
test_recall_all=[]
#for f1-score
train_f1_all=[]
valid_f1_all=[]
test_f1_all=[]

save_path = f'{DATASET}_{model.__class__.__name__}.pth'
early_stopper = EarlyStopper(model, save_path, patience=5) 
for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,model,criterion)
    
     # evaluate the performance of the model, on both data sets
    valid_loss, (y_valid_gold, y_valid_pred) = eval_dataset(val_loader,model,criterion) 
    
    test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader,model,criterion)

    train_loss_all.append(train_loss)
    valid_loss_all.append(valid_loss)
    test_loss_all.append(test_loss)
    # Convert preds and golds in a list.
    y_train_true = np.concatenate( y_train_gold, axis=0 )
    y_valid_true = np.concatenate( y_valid_gold, axis=0 )
    y_test_true = np.concatenate( y_test_gold, axis=0 )
    y_train_pred = np.concatenate( y_train_pred, axis=0 )
    y_valid_pred = np.concatenate( y_valid_pred, axis=0 )
    y_test_pred = np.concatenate( y_test_pred, axis=0 )
    train_acc_all.append(accuracy_score(y_train_true, y_train_pred))
    valid_acc_all.append(accuracy_score(y_valid_true, y_valid_pred))
    test_acc_all.append(accuracy_score(y_test_true, y_test_pred))
    print()
    print("Accuracy for train:" , accuracy_score(y_train_true, y_train_pred))
    print("Accuracy for valid:" , accuracy_score(y_valid_true, y_valid_pred))
    print("Accuracy for test:" , accuracy_score(y_test_true, y_test_pred))
    print()
    print("Recall score for train:", recall_score(y_train_true, y_train_pred, average='macro'))
    print("Recall score for valid:", recall_score(y_valid_true, y_valid_pred, average='macro'))
    print("Recall score for test:", recall_score(y_test_true, y_test_pred, average='macro'))
    print()
    print("F1 score for train:", f1_score(y_train_true, y_train_pred, average='macro'))
    print("F1 score for valid:", f1_score(y_valid_true, y_valid_pred, average='macro'))
    print("F1 score for test:", f1_score(y_test_true, y_test_pred, average='macro'))
    print()
    #append recall
    train_recall_all.append(recall_score(y_train_true, y_train_pred, average='macro'))
    valid_recall_all.append(recall_score(y_valid_true, y_valid_pred, average='macro'))
    test_recall_all.append(recall_score(y_test_true, y_test_pred, average='macro'))
    #append f1
    train_f1_all.append(f1_score(y_train_true, y_train_pred, average='macro'))
    valid_f1_all.append(f1_score(y_valid_true, y_valid_pred, average='macro'))
    test_f1_all.append(f1_score(y_test_true, y_test_pred, average='macro'))
    if early_stopper.early_stop(valid_loss):
        print('Early Stopping was activated.')
        print(f'Epoch {epoch}/{EPOCHS}, Loss at training set: {train_loss}\n\t Loss at validation set: {valid_loss}\n\t Loss at test set: {test_loss}')
        print('Training has been completed.\n')
        break
EPOCHS=epoch
print("Mean and max training accuracy is: ",np.mean(np.array(train_acc_all))," ",np.max(np.array(train_acc_all)))
print("Mean and max validation accuracy is: ",np.mean(np.array(valid_acc_all))," ",np.max(np.array(valid_acc_all)))
print("Mean and max testing accuracy is: ",np.mean(np.array(test_acc_all))," ",np.max(np.array(test_acc_all)))
print()
#for recall
print("Mean and max training recall is: ",np.mean(np.array(train_recall_all))," ",np.max(np.array(train_recall_all)))
print("Mean and max validation recall is: ",np.mean(np.array(valid_recall_all))," ",np.max(np.array(valid_recall_all)))
print("Mean and max testing recall is: ",np.mean(np.array(test_recall_all))," ",np.max(np.array(test_recall_all)))
print()
#for f1-score
print("Mean and max training f1-score is: ",np.mean(np.array(train_f1_all))," ",np.max(np.array(train_f1_all)))
print("Mean and max validation f1-score is: ",np.mean(np.array(valid_f1_all))," ",np.max(np.array(valid_f1_all)))
print("Mean and max testing f1-score is: ",np.mean(np.array(test_f1_all))," ",np.max(np.array(test_f1_all)))
print()
plt.plot(range(1,EPOCHS+1), train_loss_all,label='Train loss')
plt.plot(range(1,EPOCHS+1), valid_loss_all,label='Valid loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train-Validation Loss")
plt.legend()
plt.show()
plt.plot(range(1,EPOCHS+1), train_loss_all,label='Train loss')
plt.plot(range(1,EPOCHS+1), test_loss_all,label='Test loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train-Test Loss")
plt.legend()
plt.show()