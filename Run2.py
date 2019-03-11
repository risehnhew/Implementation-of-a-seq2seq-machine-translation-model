# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 06:20:25 2019

@author: Administrator
"""

from io import open
import unicodedata
import string
import re
import random
import argparse
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print("It's running by:", device)
#device = torch.device("cpu")


parser = argparse.ArgumentParser(description='This is the description')

#learning
learn = parser.add_argument_group('learning options')
learn.add_argument('--lr', type= float, default= 0.01, help = 'learning rate')
learn.add_argument('--dropout', type = float, default = 0.1, help = 'Dropout rate')
learn.add_argument('--hidden_size', type = int, default = 256, help = 'size of hidden_layer')

#language processing
lang = parser.add_argument_group('language processing options')
lang.add_argument('--MAX_LENGTH', type = int, default = 10, help = ' Here the maximum length is 10 words to simplify training.')
lang.add_argument('--data_path', type = str, default = 'data/%s-%s.txt', help = 'Path of training data' )
lang.add_argument('--input_lang',type = str, default = 'eng', help = 'input language name')
lang.add_argument('--output-lang', type = str, default = 'spa', help = 'output language name')

#Training arguments
trainArgs = parser.add_argument_group('Training argument')
trainArgs.add_argument('--n_iters', type = int, default = 7500, help = 'number of iters times') #75000
trainArgs.add_argument('--print_every', type = int, default = 500, help = 'to print results after how many times run ') # 5000

args = parser.parse_args()







SOS_token = 0
EOS_token = 1


    
    
#---------------------------------------------------------------Dataset preprocess----------------------------------------------------
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1:"EOS"}
        self.n_words = 2 # count SOS and EOS
        
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
     
        
        
def unicodeToAscii(s):  #Turn a Unicode string to plain ASCII
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s): # Normalize String: Lowercase, trim, and remove non-letter characters
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False): # read Dataset
    print("Reading lines...")
    
    
    #Read the file and split into lines by open().read().strip().split()
    #string.split(sepatator, max) 'sepatator'
    lines = open(args.data_path %(lang1, lang2), encoding='utf-8').read().strip().split('\n')  # eng-fra has 164327 sentence pairs  eng-cmn has 20403 sentence pairs
        
    #Split every line into pairs and normalize    
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    if reverse:

        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs



MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):   # reverse may improve the result
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData(args.input_lang, args.output_lang, True)  # input langauage pairs. Datasets from https://www.manythings.org/anki/
print(random.choice(pairs))


#------------------------------------------------------------------------------
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(input_size, hidden_size)#A simple lookup table that stores embeddings of a fixed dictionary and size.
        self.gru = nn.GRU(hidden_size, hidden_size)
        
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden
    def initHidden(self):  #initial Hidden-layer
        return torch.zeros(1, 1, self.hidden_size, device = device)
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
    
class AttnDecoderRNN(nn.Module):  #  'Neural Machine Translation by Jointly Learning to Align and Translate'
    def __init__(self, hidden_size, output_size, dropout_p=args.dropout, max_length=args.MAX_LENGTH): # 
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size) # A simple lookup table that stores embeddings of a fixed dictionary and size.
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)  # 
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self): # Initial hidden-layer
        return torch.zeros(1, 1, self.hidden_size, device=device)            
#------------------------------------------------------------------------------
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)
#------------------------------------------------------------------------------
teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=args.MAX_LENGTH):
    encoder_hidden = encoder.initHidden() # init hidden layer

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length): # word by word
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

#    if use_teacher_forcing:
#        # Teacher forcing: Feed the target as the next input
#        for di in range(target_length):
#            decoder_output, decoder_hidden, decoder_attention = decoder(
#                decoder_input, decoder_hidden, encoder_outputs)
#            loss += criterion(decoder_output, target_tensor[di])
#            decoder_input = target_tensor[di]  # Teacher forcing
#
#    else:
        # Without teacher forcing: use its own predictions as the next input
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input

        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == EOS_token:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length
#--------------------print time consume------------------------------------------------------------- 
import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

#--------------------------------------------------------------------------------------------

def trainIters(encoder, decoder, n_iters, print_every=100, plot_every=10, learning_rate=args.lr):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:  # print results
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0: # plot results
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)
    #torch.save(encoder.state_dict(), './model_parameters/encoder.weights') #Returns a dictionary containing a whole state of the module.
    #torch.save(decoder.state_dict(), './model_parameters/decoder.weights')
#------------------------------------------------------------------------------
# plot the result   
    
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
#------------------------------------------------------------------------------
# evaluate the result    
def evaluate(encoder, decoder, sentence, max_length=args.MAX_LENGTH):  
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]
    
def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('input:', pair[0])
        print('target:', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('predict:', output_sentence)
        print('')

#------------------------------------------------------------------------------
        
#hidden_size = 256 

encoder1 = EncoderRNN(input_lang.n_words, args.hidden_size).to(device) # Enconder
attn_decoder1 = AttnDecoderRNN(args.hidden_size, output_lang.n_words, dropout_p=args.dropout).to(device)#decoder


#------------Load trained weights-------------------------------
encoder1.load_state_dict(torch.load('./model_parameters/encoder.weights'))
attn_decoder1.load_state_dict(torch.load('./model_parameters/decoder.weights'))

encoder1 = encoder1.eval()
attn_decoder1 = attn_decoder1.eval()
#------------------------------------------------------

#if torch.cuda.device_count() >1: # Multi-GPUs
#    print("Let's use", torch.cuda.device_count(),"GPUs!")
#    encoder1 = nn.DataParallel(encoder1)
#    attn_decoder1 = nn.DataParallel(attn_decoder1)
#encoder1.to(device)
#attn_decoder1.to(device)


trainIters(encoder1, attn_decoder1, args.n_iters, print_every=args.print_every)  # hyperparameters
evaluateRandomly(encoder1, attn_decoder1)


