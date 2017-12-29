
# coding: utf-8

# In[7]:

import PyPDF2
import os
import re
import string
import random

PARTIJPATH = 'partijprogrammas'
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")


# In[2]:

def basic_tokenizer(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]

def get_index_end_sentence(tokens):
    for index,token in enumerate(tokens):
        if token in ["?","!","."]:
            return index
    return len(tokens)
def get_sentences_from_tokens(tokens):
    sentences = []
    while len(tokens) > 0:
        nextindex = get_index_end_sentence(tokens)
        if nextindex == len(tokens):
            sentences.append(tokens)
            return sentences
        else:
            sentences.append(tokens[:nextindex+1])
            tokens = tokens[nextindex+1:]
    return sentences
def get_parties_and_sentences():
    partijprogrammas = os.listdir(PARTIJPATH)
    part_sentences = dict()
    
    
    for partijprogramma_name in partijprogrammas:
        print(partijprogramma_name)
        pdf_obj = open(os.path.join(PARTIJPATH,partijprogramma_name),'rb')
        pdfreader = PyPDF2.PdfFileReader(pdf_obj)

        sentences = []
        
        for page in pdfreader.pages:
            text = page.extractText()
            # remove all words with a number
            text = text.replace("\n"," ")

            text = re.sub(r'\w*\d\w*', '', text).strip()
            
            # make lower case
            text = text.lower()
            if(len(text)>0):
                tokens = basic_tokenizer(text)
                sentences_this_page = get_sentences_from_tokens(tokens)
                sentences.extend(sentences_this_page)
        part_sentences[partijprogramma_name] = sentences
    return part_sentences

parties_and_sentences = get_parties_and_sentences()
print("done!")


# In[8]:

print([a for a in parties_and_sentences.keys()])
print(parties_and_sentences['SP.pdf'][100])
vocab = dict()
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]


for party in parties_and_sentences.keys():
    for line in parties_and_sentences[party]:
        for word in line:
            if not word in vocab:
                vocab[word]=0
            vocab[word] +=1
        
sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
sorted_vocab = [word for word in sorted_vocab if vocab[word] >= 5]
vocab_list = _START_VOCAB + sorted_vocab
print(len(vocab_list))
for uncommon in vocab_list[-20:]:
    print(uncommon + " : " + str(vocab[uncommon]))
    
class IdOfWordGetter:
    def get_word_of_id(self,index):
        return self.vocab[index]
    def get_id_of_word(self,word):
        if word in self.word_dict:
            return self.word_dict[word]
        else:
            return self.word_dict[_UNK]
    def __init__(self,vocab):
        self.vocab = vocab
        self.word_dict = dict()
        for index,word in enumerate(vocab):
            self.word_dict[word] = index
id_of_word_getter = IdOfWordGetter(vocab_list)
party_count = len(parties_and_sentences.keys())
#if len(vocab_list) > max_vocabulary_size:
#     vocab_list = vocab_list[:max_vocabulary_size]
# with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
# for w in vocab_list:
#   vocab_file.write(w + b"\n")




# In[11]:

import random
def load_data(parties_and_sentences,sorted_vocab,getter):
    labels = parties_and_sentences.keys()
    train_x = list()
    train_y = list()
    for party_index,name in enumerate(parties_and_sentences.keys()):
        print(name + " has "  + str(len(parties_and_sentences[name])) + " sentences ")
        for sentence in parties_and_sentences[name]:
            numeric_sentence = list()
            for word in sentence:
                numeric_sentence.append(getter.get_id_of_word(word))
            train_x.append(numeric_sentence)
            train_y.append(party_index)
    train_set = list(zip(train_x,train_y))
    random.shuffle(train_set)
    percentage_split = 0.8
    train = train_set[:int(percentage_split*len(train_set))]
    test = train_set[int(percentage_split*len(train_set)):]
    return zip(*train),zip(*test)


(X_train, y_train), (X_test, y_test) = load_data(parties_and_sentences,sorted_vocab,id_of_word_getter)
    


# In[16]:

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils





import numpy as np
# fix random seed for reproducibility
np.random.seed(7)
top_words = len(vocab_list)
#top_words = 5000
#(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)


# In[17]:

print(y_train[3])


# In[23]:



# truncate and pad input sequences
max_review_length = 20
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
y_train = np_utils.to_categorical(y_train, party_count)
prevytest = y_test
y_test = np_utils.to_categorical(y_test, party_count)


# In[24]:

print(X_train[3])


# In[41]:

# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(party_count, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=3, batch_size=64,verbose=1)


# In[ ]:

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:

classes_predicted = model.predict_classes(X_test, batch_size=32, verbose=1)
for i in range(100):
    for id in X_test[i]:
        print(id_of_word_getter.get_word_of_id(id),end=' ')
    print()
    print("Predicted: " + list(parties_and_sentences.keys())[classes_predicted[i]])
    print("Actual: " + list(parties_and_sentences.keys())[prevytest[i]])


# In[ ]:



