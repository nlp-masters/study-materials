import logging
logging.basicConfig(filename=r'NPLM_GPU.log',level=logging.INFO)
import time
logging.info('\n')
logging.info(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

import nltk
import csv
from nltk.corpus import brown
from nltk.corpus import wordnet
import numpy as np
import pandas as pd

# BROWN_PATH = './corpora/brown'

num_para = len(brown.paras())
num_sent_per_para = [len(x) for x in brown.paras()]
num_sent_per_para = [len(x) for x in brown.paras()]
print("Number of Paragraphs: ", num_para)

num_sent_per_para_freq_tab = pd.Series(num_sent_per_para).value_counts().sort_index()
print("Number of Sentences per Paragraph (Top 10)")
print(num_sent_per_para_freq_tab[:10])

def create_datasets(num_corpus,context_size):
    
    num_train = round(num_corpus * 4/5)

    UNK_symbol = "<UNK>"
    vocab = set([UNK_symbol])

    paragraphs = brown.paras()
    corpus_data = list()
    for idx in np.arange(num_corpus):
        words = []
        for sentence in paragraphs[idx]:
            for word in sentence:
                words.append(word.lower())
        corpus_data.append(words)

    # create term frequency of the words
    words_term_frequency_train = {}
    for doc in corpus_data:
        for word in doc:
            words_term_frequency_train[word] = words_term_frequency_train.get(word,0) + 1

    # create vocabulary
    for doc in corpus_data:
        for word in doc:
            if words_term_frequency_train.get(word,0) >= 5:
                vocab.add(word)

    # create required lists
    x_train = []
    y_train = []
    x_dev = []
    y_dev = []

    # create word to id mappings
    word_to_id_mappings = {}
    for idx,word in enumerate(vocab):
        word_to_id_mappings[word] = idx

    # function to get id for a given word
    # return <UNK> id if not found
    def get_id_of_word(word):
        unknown_word_id = word_to_id_mappings['<UNK>']
        return word_to_id_mappings.get(word,unknown_word_id)

    # creating training and dev set
    for idx,paragraph in enumerate(brown.paras()[:num_corpus]):
        for sentence in paragraph:
            for i,word in enumerate(sentence):
                if i+context_size >= len(sentence):
                    # sentence boundary reached
                    # ignoring sentence less than (context_size+1) words
                    break
                # convert word to id
                if context_size == 2:
                    x_extract = [get_id_of_word(word.lower()),get_id_of_word(sentence[i+1].lower())]
                elif context_size == 3:
                    x_extract = [get_id_of_word(word.lower()),get_id_of_word(sentence[i+1].lower()),get_id_of_word(sentence[i+2].lower())]
                elif context_size == 4:
                    x_extract = [get_id_of_word(word.lower()),get_id_of_word(sentence[i+1].lower()),get_id_of_word(sentence[i+2].lower()),get_id_of_word(sentence[i+3].lower())]
                y_extract = [get_id_of_word(sentence[i+context_size].lower())]
                if idx < num_train:
                    x_train.append(x_extract)
                    y_train.append(y_extract)
                else:
                    x_dev.append(x_extract)
                    y_dev.append(y_extract)

    # making numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_dev = np.array(x_dev)
    y_dev = np.array(y_dev)
    
    print("Size of Vocabulary: " + str(len(vocab)))
    logging.info("Size of Vocabulary: " + str(len(vocab)))
    print("Size of x_train, y_train, x_dev, and y_dev sets: " + str(x_train.shape) + ", " + str(y_train.shape) + ", " + str(x_dev.shape) + ", " + str(y_dev.shape))
    logging.info("Size of x_train, y_train, x_dev, and y_dev sets: " + str(x_train.shape) + ", " + str(y_train.shape) + ", " + str(x_dev.shape) + ", " + str(y_dev.shape))
    return x_train, y_train, x_dev, y_dev, vocab

import torch
import multiprocessing
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time

# Trigram Neural Network Model
class TrigramNNmodel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size, h):
        super(TrigramNNmodel, self).__init__()
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, h)
        self.linear2 = nn.Linear(h, vocab_size, bias = False)

    def forward(self, inputs):
        # compute x': concatenation of x1 and x2 embeddings
        embeds = self.embeddings(inputs).view((-1,self.context_size * self.embedding_dim))
        # compute h: tanh(W_1.x' + b)
        out = torch.tanh(self.linear1(embeds))
        # compute W_2.h
        out = self.linear2(out)
        # compute y: log_softmax(W_2.h)
        log_probs = F.log_softmax(out, dim=1)
        # return log probabilities
        # BATCH_SIZE x len(vocab)
        return log_probs
    
# helper function to get accuracy from log probabilities
def get_accuracy_from_log_probs(log_probs, labels):
    probs = torch.exp(log_probs)
    predicted_label = torch.argmax(probs, dim=1)
    acc = (predicted_label == labels).float().mean()
    return acc

# helper function to evaluate model on dev data
def evaluate(model, criterion, dataloader, epoch, gpu = -1):
    model.eval()
    mean_acc, mean_loss = 0, 0
    count = 0
    with torch.no_grad():
        dev_st = time.time()
        for it, data_tensor in enumerate(dataloader):
            context_tensor = data_tensor[:,0:model.context_size]
            target_tensor = data_tensor[:,model.context_size]
            if gpu >= 0:
                context_tensor, target_tensor = context_tensor.cuda(gpu), target_tensor.cuda(gpu)
            else:
                context_tensor, target_tensor = context_tensor.long(), target_tensor.long()
            log_probs = model(context_tensor)
            mean_loss += criterion(log_probs, target_tensor).item()
            mean_acc += get_accuracy_from_log_probs(log_probs, target_tensor)
            count += 1
            # if it % 1000 == 0: 
                # print("Valid., Epoch {}, Iter. {}, Loss {:.2f}, Acc. {:.2f}, Time: {:.0f}, ".format(epoch+1, it, mean_loss/count, mean_acc/count, time.time() - dev_st))
                # logging.info("Valid., Epoch {}, Iter. {}, Loss {:.2f}, Acc. {:.2f}, Time: {:.0f}, ".format(epoch+1, it, mean_loss/count, mean_acc/count, time.time() - dev_st))
                # dev_st = time.time()
    return it, mean_acc / count, mean_loss / count
    
# check if gpu is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
available_workers = multiprocessing.cpu_count()

def dataloader(x_train,y_train,x_dev,y_dev,BATCH_SIZE):
    print("--- Creating training and dev dataloaders with {} batch size ---".format(BATCH_SIZE))
    train_set = np.concatenate((x_train, y_train), axis=1)
    dev_set = np.concatenate((x_dev, y_dev), axis=1)
    train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, num_workers = available_workers)
    dev_loader = DataLoader(dev_set, batch_size = BATCH_SIZE, num_workers = available_workers)
    print("Device: " + str(device) + ", Aviailable workers: " + str(available_workers))
    logging.info("Device: " + str(device) + ", Aviailable workers: " + str(available_workers))
    return train_loader, dev_loader

gpu = 0 

def run_training(SIZE_CORPUS, EMBEDDING_DIM , CONTEXT_SIZE, BATCH_SIZE, H, NUM_EPOCHS):
    
    t0 = time.time()
    
    x_train, y_train, x_dev, y_dev, vocab = create_datasets(SIZE_CORPUS,CONTEXT_SIZE)
    train_loader, dev_loader = dataloader(x_train=x_train,y_train=y_train,x_dev=x_dev,y_dev=y_dev,BATCH_SIZE=BATCH_SIZE)
    loss_function = nn.NLLLoss()
    model = TrigramNNmodel(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE, H)
    if device=='cuda':
        model.cuda(gpu)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    t1 = time.time()

    # ------------------------- TRAIN & SAVE MODEL ------------------------
    best_acc = 0
    best_model_path = None
    for epoch in range(NUM_EPOCHS):
        st = time.time()
        for it, data_tensor in enumerate(train_loader):  
            if device == 'cuda':
                context_tensor = data_tensor[:,0:model.context_size].cuda(gpu)
                target_tensor = data_tensor[:,model.context_size].cuda(gpu)
            else:
                context_tensor = data_tensor[:,0:model.context_size].long()
                target_tensor = data_tensor[:,model.context_size].long()
            model.zero_grad()
            log_probs = model(context_tensor)
            acc = get_accuracy_from_log_probs(log_probs, target_tensor)
            loss = loss_function(log_probs, target_tensor)
            loss.backward()
            optimizer.step()

            # if it % 1000 == 0: 
            #    print("Train, Epoch {}, Iter. {}, Loss {:.2f}, Acc. {:.2f}".format(epoch+1, it, loss.item(), acc))
        print("Train, Epoch {}, Iter. {}, Loss {:.2f}, Acc. {:.2f}".format(epoch+1, it, loss.item(), acc))
        logging.info("Train, Epoch {}, Iter. {}, Loss {:.2f}, Acc. {:.2f}".format(epoch+1, it, loss.item(), acc))

        if device=='cuda':
            iter, dev_acc, dev_loss = evaluate(model, loss_function, dev_loader, epoch=epoch, gpu=0)
        else:
            iter, dev_acc, dev_loss = evaluate(model, loss_function, dev_loader, epoch=epoch)
        print("Valid, Epoch {}, Iter. {}, Loss {:.2f}, Acc. {:.2f}".format(epoch+1, iter, dev_loss, dev_acc))
        logging.info("Valid, Epoch {}, Iter. {}, Loss {:.2f}, Acc. {:.2f}".format(epoch+1, iter, dev_loss, dev_acc))
        if dev_acc > best_acc:
            best_acc = dev_acc
            best_model_path = 'best_model_{}.dat'.format(epoch+1)
            torch.save(model.state_dict(), best_model_path)
            final_epoch = epoch

    t2 = time.time()
    duration0 = round(t1-t0)
    duration1 = round(t2-t1)

    print("Corpus, Embedding, Context, Batch, Hidden, Epochs: {}, {}, {}, {}, {}, {}: ".format(SIZE_CORPUS, EMBEDDING_DIM, CONTEXT_SIZE, BATCH_SIZE, H, NUM_EPOCHS))
    logging.info("Corpus, Embedding, Context, Batch, Hidden, Epochs: {}, {}, {}, {}, {}, {}: ".format(SIZE_CORPUS, EMBEDDING_DIM, CONTEXT_SIZE, BATCH_SIZE, H, NUM_EPOCHS))
    print("Data Time, Training Time: {:.0f}s, {:.0f}s".format(duration0, duration1))
    logging.info("Data Time, Training Time: {:.0f}s, {:.0f}s".format(duration0, duration1))

    result_num = [SIZE_CORPUS, EMBEDDING_DIM, CONTEXT_SIZE, BATCH_SIZE, H, NUM_EPOCHS, final_epoch, duration0, duration1, round(dev_acc.item(),3), round(dev_loss,3)]
    result_str = str(result_num) + '\n'
    with open('nlpm_result.csv','a') as f:
        f.write(result_str)
        
        
# torch.manual_seed(13013)
#run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 200, CONTEXT_SIZE = 2, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 5)
#run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 200, CONTEXT_SIZE = 2, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 10)
#run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 200, CONTEXT_SIZE = 2, BATCH_SIZE = 256, H = 200, NUM_EPOCHS = 5)
#run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 200, CONTEXT_SIZE = 2, BATCH_SIZE = 256, H = 200, NUM_EPOCHS = 10)
# run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 200, CONTEXT_SIZE = 3, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 5)
# run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 200, CONTEXT_SIZE = 3, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 10)
# run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 200, CONTEXT_SIZE = 3, BATCH_SIZE = 256, H = 200, NUM_EPOCHS = 5)
# run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 200, CONTEXT_SIZE = 3, BATCH_SIZE = 256, H = 200, NUM_EPOCHS = 10)
# run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 400, CONTEXT_SIZE = 2, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 5)
# run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 400, CONTEXT_SIZE = 2, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 10)
# run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 400, CONTEXT_SIZE = 2, BATCH_SIZE = 256, H = 200, NUM_EPOCHS = 5)
# run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 400, CONTEXT_SIZE = 2, BATCH_SIZE = 256, H = 200, NUM_EPOCHS = 10)
# run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 400, CONTEXT_SIZE = 3, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 5)
# run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 400, CONTEXT_SIZE = 3, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 10)
# run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 400, CONTEXT_SIZE = 3, BATCH_SIZE = 256, H = 200, NUM_EPOCHS = 5)
# run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 400, CONTEXT_SIZE = 3, BATCH_SIZE = 256, H = 200, NUM_EPOCHS = 10)

# run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 200, CONTEXT_SIZE = 4, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 5)
# run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 200, CONTEXT_SIZE = 4, BATCH_SIZE = 256, H = 200, NUM_EPOCHS = 5)
# run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 200, CONTEXT_SIZE = 4, BATCH_SIZE = 256, H = 300, NUM_EPOCHS = 5)
# run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 400, CONTEXT_SIZE = 4, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 5)
# run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 400, CONTEXT_SIZE = 4, BATCH_SIZE = 256, H = 200, NUM_EPOCHS = 5)
# run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 400, CONTEXT_SIZE = 4, BATCH_SIZE = 256, H = 300, NUM_EPOCHS = 5)
# run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 800, CONTEXT_SIZE = 4, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 5)
# run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 800, CONTEXT_SIZE = 4, BATCH_SIZE = 256, H = 200, NUM_EPOCHS = 5)
# run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 800, CONTEXT_SIZE = 4, BATCH_SIZE = 256, H = 300, NUM_EPOCHS = 5)

for repeat in range(10):
    run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 50, CONTEXT_SIZE = 2, BATCH_SIZE = 128, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 50, CONTEXT_SIZE = 2, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 50, CONTEXT_SIZE = 2, BATCH_SIZE = 512, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 50, CONTEXT_SIZE = 4, BATCH_SIZE = 128, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 50, CONTEXT_SIZE = 4, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 50, CONTEXT_SIZE = 4, BATCH_SIZE = 512, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 100, CONTEXT_SIZE = 2, BATCH_SIZE = 128, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 100, CONTEXT_SIZE = 2, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 100, CONTEXT_SIZE = 2, BATCH_SIZE = 512, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 100, CONTEXT_SIZE = 4, BATCH_SIZE = 128, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 100, CONTEXT_SIZE = 4, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 100, CONTEXT_SIZE = 4, BATCH_SIZE = 512, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 200, CONTEXT_SIZE = 2, BATCH_SIZE = 128, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 200, CONTEXT_SIZE = 2, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 200, CONTEXT_SIZE = 2, BATCH_SIZE = 512, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 200, CONTEXT_SIZE = 4, BATCH_SIZE = 128, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 200, CONTEXT_SIZE = 4, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 15667, EMBEDDING_DIM = 200, CONTEXT_SIZE = 4, BATCH_SIZE = 512, H = 100, NUM_EPOCHS = 5)

    run_training(SIZE_CORPUS = 8000, EMBEDDING_DIM = 50, CONTEXT_SIZE = 2, BATCH_SIZE = 128, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 8000, EMBEDDING_DIM = 50, CONTEXT_SIZE = 2, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 8000, EMBEDDING_DIM = 50, CONTEXT_SIZE = 2, BATCH_SIZE = 512, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 8000, EMBEDDING_DIM = 50, CONTEXT_SIZE = 4, BATCH_SIZE = 128, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 8000, EMBEDDING_DIM = 50, CONTEXT_SIZE = 4, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 8000, EMBEDDING_DIM = 50, CONTEXT_SIZE = 4, BATCH_SIZE = 512, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 8000, EMBEDDING_DIM = 100, CONTEXT_SIZE = 2, BATCH_SIZE = 128, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 8000, EMBEDDING_DIM = 100, CONTEXT_SIZE = 2, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 8000, EMBEDDING_DIM = 100, CONTEXT_SIZE = 2, BATCH_SIZE = 512, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 8000, EMBEDDING_DIM = 100, CONTEXT_SIZE = 4, BATCH_SIZE = 128, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 8000, EMBEDDING_DIM = 100, CONTEXT_SIZE = 4, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 8000, EMBEDDING_DIM = 100, CONTEXT_SIZE = 4, BATCH_SIZE = 512, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 8000, EMBEDDING_DIM = 200, CONTEXT_SIZE = 2, BATCH_SIZE = 128, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 8000, EMBEDDING_DIM = 200, CONTEXT_SIZE = 2, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 8000, EMBEDDING_DIM = 200, CONTEXT_SIZE = 2, BATCH_SIZE = 512, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 8000, EMBEDDING_DIM = 200, CONTEXT_SIZE = 4, BATCH_SIZE = 128, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 8000, EMBEDDING_DIM = 200, CONTEXT_SIZE = 4, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 8000, EMBEDDING_DIM = 200, CONTEXT_SIZE = 4, BATCH_SIZE = 512, H = 100, NUM_EPOCHS = 5)

    run_training(SIZE_CORPUS = 4000, EMBEDDING_DIM = 50, CONTEXT_SIZE = 2, BATCH_SIZE = 128, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 4000, EMBEDDING_DIM = 50, CONTEXT_SIZE = 2, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 4000, EMBEDDING_DIM = 50, CONTEXT_SIZE = 2, BATCH_SIZE = 512, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 4000, EMBEDDING_DIM = 50, CONTEXT_SIZE = 4, BATCH_SIZE = 128, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 4000, EMBEDDING_DIM = 50, CONTEXT_SIZE = 4, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 4000, EMBEDDING_DIM = 50, CONTEXT_SIZE = 4, BATCH_SIZE = 512, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 4000, EMBEDDING_DIM = 100, CONTEXT_SIZE = 2, BATCH_SIZE = 128, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 4000, EMBEDDING_DIM = 100, CONTEXT_SIZE = 2, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 4000, EMBEDDING_DIM = 100, CONTEXT_SIZE = 2, BATCH_SIZE = 512, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 4000, EMBEDDING_DIM = 100, CONTEXT_SIZE = 4, BATCH_SIZE = 128, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 4000, EMBEDDING_DIM = 100, CONTEXT_SIZE = 4, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 4000, EMBEDDING_DIM = 100, CONTEXT_SIZE = 4, BATCH_SIZE = 512, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 4000, EMBEDDING_DIM = 200, CONTEXT_SIZE = 2, BATCH_SIZE = 128, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 4000, EMBEDDING_DIM = 200, CONTEXT_SIZE = 2, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 4000, EMBEDDING_DIM = 200, CONTEXT_SIZE = 2, BATCH_SIZE = 512, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 4000, EMBEDDING_DIM = 200, CONTEXT_SIZE = 4, BATCH_SIZE = 128, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 4000, EMBEDDING_DIM = 200, CONTEXT_SIZE = 4, BATCH_SIZE = 256, H = 100, NUM_EPOCHS = 5)
    run_training(SIZE_CORPUS = 4000, EMBEDDING_DIM = 200, CONTEXT_SIZE = 4, BATCH_SIZE = 512, H = 100, NUM_EPOCHS = 5)
