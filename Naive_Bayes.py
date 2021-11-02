#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from csv import reader
import string
import re
import collections
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from math import sqrt
from math import pi
from math import exp, log
import math
import numpy as np


# In[3]:


def load_file(filename):
    file = open(filename, "r")
    lines = reader(file)
    data = list(lines)
    all_text = []
    for row in data:
        all_text.append(', '.join(row))
    text = all_text[7:-1]
    words = []
    words_in_array = []
    for x in range (len(text)):
        words += (text[x]).split()
    words_in_array.append(words)
    words_in_text = words_in_array[0]
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    stripped_words = [re_punc.sub('', w) for w in words_in_text]
    stripped_words = [word.lower() for word in stripped_words]
    return stripped_words


# In[4]:


def load_all_files(path):
    files = os.listdir(path)
    class_data = []
    for file in files:
        class_data.append(load_file(path + '\\' + file))
    return class_data


# In[5]:


hardware = load_all_files(r'dataset2/train/comp.sys.ibm.pc.hardware')
electronics = load_all_files(r'dataset2/train/sci.electronics')


# In[6]:


full_dataset = hardware + electronics


# In[7]:


vocab = collections.Counter()

for items in full_dataset:
    vocab.update(items)
    
min_occurrence = 5
tokens = [k for k, c in vocab.items() if c >= min_occurrence]


# In[8]:


vectorizer =  TfidfVectorizer()


# In[9]:


vectorizer.fit(tokens)


# In[10]:


def tokenizer(path):
    files = os.listdir(path)
    class_data = []
    for file in files:
        text = load_file(path + '\\' + file)
        text_array = ''
        for words in text:
            text_array += words + ' '
        vector = vectorizer.transform([text_array])
        array = vector.toarray()
        list = array.tolist()
        class_data.append(list[0])
    return class_data


# In[11]:


hardware_tokens = tokenizer(r'dataset2/train/comp.sys.ibm.pc.hardware')
electronic_tokens = tokenizer(r'dataset2/train/sci.electronics')


# In[12]:


def get_counts(dataset):
    array = np.array(dataset)
    counts = []
    for i in range(len(dataset[0])):
        counts.append(sum(array[:,i]))
    return counts


# In[13]:


def get_all_probabilities(hardware_tokens, electronic_tokens):
    hardware_counts = np.array(get_counts(hardware_tokens))
    electronic_counts = np.array(get_counts(electronic_tokens))
    total_counts = hardware_counts + electronic_counts
    hardware_probabilities = hardware_counts / total_counts
    electronic_probabilities = electronic_counts / total_counts
    return [hardware_probabilities.tolist(), electronic_probabilities.tolist()]


# In[14]:


hardware_probabilities = get_all_probabilities(hardware_tokens, electronic_tokens)[0]
electronic_probabilities = get_all_probabilities(hardware_tokens, electronic_tokens)[1]


# In[15]:


def classify(instance, hardware_probabilities, electronic_probabilities):
    a = math.exp(math.log(0.5)+math.log(sum(np.array(hardware_probabilities)*np.array(instance))+0.00000000001))
    b = math.exp(math.log(0.5)+math.log(sum(np.array(electronic_probabilities)*np.array(instance))+0.00000000001))
    if a > b:
        return 0
    elif a < b:
        return 1


# In[16]:


def tokenizer_with_labels(path, label):
    files = os.listdir(path)
    class_data = []
    for file in files:
        text = load_file(path + '\\' + file)
        text_array = ''
        for words in text:
            text_array += words + ' '
        vector = vectorizer.transform([text_array])
        array = vector.toarray()
        list = array.tolist()
        class_data.append([list[0], label])
    return class_data


# In[17]:


test_hardware_tokens = tokenizer(r'dataset2/test/comp.sys.ibm.pc.hardware')
test_electronic_tokens = tokenizer(r'dataset2/test/sci.electronics')
testing_dataset = test_hardware_tokens + test_electronic_tokens


# In[18]:


labeled_test_hardware_tokens = tokenizer_with_labels(r'dataset2/test/comp.sys.ibm.pc.hardware', 0)
labeled_test_electronic_tokens = tokenizer_with_labels(r'dataset2/test/sci.electronics', 1)
labeled_testing_dataset = labeled_test_hardware_tokens + labeled_test_electronic_tokens


# In[19]:


def get_all_predictions(labeled_testing_dataset, hardware_probabilites, electronic_probabilities):
    prediction_array = []
    for instance in labeled_testing_dataset:
        prediction = classify(instance[0], hardware_probabilities, electronic_probabilities)
        prediction_array.append([prediction, instance[1]])
    return prediction_array


# In[20]:


def get_accuracy(array):
    correct = 0
    for i in range(len(array)):
        if array[i][0] == array[i][1]:
            correct += 1
    return correct / float(len(array)) * 100.0


# In[21]:


prediction_array = get_all_predictions(labeled_testing_dataset, hardware_probabilities, electronic_probabilities)


# In[22]:


get_accuracy(prediction_array)


# In[ ]:





# In[ ]:




