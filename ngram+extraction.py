
# coding: utf-8

# In[8]:

import nltk
import pandas as pd
import numpy as np
import re
from nltk.tokenize import RegexpTokenizer, word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams


# In[37]:

df = pd.read_csv('/Users/jrnash/Downloads/juliacsv.csv')

text = df.loc[:, "Comments"].values
txt = str(text)

#trying to get rid of 'r'
# gg = df['Comments'].replace(regex=True,inplace=False,to_replace='\\r',value='')
# gk = pd.DataFrame(gg)

regex_token = RegexpTokenizer(r'[a-zA-Z]+')                    
rx = regex_token.tokenize(txt)
rx = [x.lower() for x in rx]
line = [i for i in rx if len(i) > 1]
stop_words = set(stopwords.words('english'))
sw = [j for j in line if not j in stop_words]

no_words = ['ft','psi', 'mph', 'inch','gpm','rpm','pbq','ppt']

clean = [word for word in sw if word not in no_words]

# as defined per measurements, I added a few. Maybe limit to only 3 character words?

print ('no useless measurements: ', clean)
    
O = ngrams(clean,1)
B = ngrams(clean,2)
T = ngrams(clean,3)

onegramlist = []
bigramlist = []
trigramlist = []

def excel(onegrams, bigrams, trigrams):
    for o in onegrams:
        f = nltk.pos_tag(o)
        onegramlist.append(f)
        
    for b in bigrams:
        w = nltk.pos_tag(b)
        bigramlist.append(w)  
        
    for t in trigrams:
        s = nltk.pos_tag(t)
        trigramlist.append(s)
        
        
    return onegramlist, bigramlist,trigramlist


onegrams_with_pos, bigrams_with_pos,trigrams_with_pos = excel(O, B, T)

def is_onegram_valid(pos1):
    if pos1 == "NN":
        return True
    else:
        return False

def is_bigram_valid(pos1,pos2):
    if pos1 == "VBG" and pos2 == "NN":
        return True
    elif pos1 == "NN" and pos2 == "NN":
        return True
    else:
        return False
    
def is_trigram_valid(pos1,pos2,pos3):
    if pos1 == "NN" and pos2 == "NN" and pos3 == "NN":
        return True
    else:
        return False
    

onegrams_without_pos =  [" ".join([x[0][0]]) for x in onegrams_with_pos if is_onegram_valid(x[0][1])]
bigrams_without_pos = [" ".join([x[0],y[0]]) for x,y in bigrams_with_pos if is_bigram_valid(x[1],y[1])]
trigrams_without_pos = [" ".join([x[0],y[0],z[0]]) for x,y,z in trigrams_with_pos if is_trigram_valid(x[1],y[1],z[1])]


max_length = max(len(onegrams_without_pos),len(bigrams_without_pos),len(trigrams_without_pos))

onegram_result = onegrams_without_pos
trigram_result = trigrams_without_pos
bigram_result = bigrams_without_pos

if len(bigrams_without_pos) <= max_length:
    bigram_result = bigrams_without_pos + [""]*(max_length-len(bigrams_without_pos))
if len(onegrams_without_pos) <= max_length:
    onegram_result = onegrams_without_pos + [""]*(max_length-len(onegrams_without_pos))
if len(trigrams_without_pos) <= max_length:
    trigram_result = trigrams_without_pos + [""]*(max_length-len(trigrams_without_pos))
    
# print(len(onegram_result))                                                 
# print(len(bigram_result))
# print(len(trigram_result))  

result_df = pd.DataFrame({"onegrams":onegram_result,"bigrams":bigram_result,"trigrams":trigram_result})

result_df.to_csv("umoranz.csv")


# In[ ]:



