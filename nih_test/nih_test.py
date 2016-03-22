"""
This should run with python2.x
(gensim's dtm wrapper makes an error for python3.x
 Hopefully will fix this issue later.)

Jihyun Park
jihyunp@uci.edu
March 18, 2016
"""

from gensim import corpora, models, similarities
import gensim.models.wrappers as wrappers 
import numpy as np
import cPickle as cp
import os




data_prefix = '/Users/jihyun/Documents/jihyun/research/topicmodel/codes/dtm/gensim_python/nih_test/data_test/'
if not os.path.exists(data_prefix):
    os.makedirs(data_prefix)

## Sample Doc (load doc)
gensim_X_file = '/Users/jihyun/Documents/jihyun/research/topicmodel/data/gensim_ver_X.pkl'
gensim_dict_file = '/Users/jihyun/Documents/jihyun/research/topicmodel/data/gensim_ver_dictionary.pkl'

corpus = cp.load(open(gensim_X_file, 'rb'))
dictionary = cp.load(open(gensim_dict_file, 'rb'))

## Sample Timeslices
year_file = '/Users/jihyun/Documents/jihyun/research/topicmodel/data/years_gensim.txt'
years = []
import csv
csvreader = csv.reader(open(year_file))
for line in csvreader:
    for item in line:
        years.append(int(item))


import matplotlib.pyplot as plt
maxyear = int(np.max(years))
minyear = int(np.min(years))


hist = np.histogram(years, bins=maxyear-minyear, range=(minyear, maxyear) )
timeslices = hist[0]


#print(minyear)
#print(maxyear)
#plt.hist(years, bins=maxyear-minyear, range=(minyear, maxyear))
#plt.xlabel(np.unique(years))
#plt.savefig('fig.pdf', format='pdf')





#
# print(dictionary.token2id)
#
# new_doc = "Human computer interaction"
# new_vec = dictionary.doc2bow(new_doc.lower().split())
# print(new_vec) # the word "interaction" does not appear in the dictionary and is ignored
#
#
#
# #dtm_path = '/Users/jihyun/Documents/jihyun/research/topicmodel/dtm/dtm-master/bin/dtm-darwin64'
dtm_path = '/Users/jihyun/Documents/jihyun/research/topicmodel/dtm/dtm-master/bin/dtm-darwin64'
#
#
# model = wrappers.DtmModel(dtm_path, corpus, timeslices, num_topics=50, id2word=dictionary, prefix=data_prefix, initialize_lda=False)




## Run with partial data


dictionary.save(os.path.join(data_prefix, 'dictionary.dict'))

vocFile = open(os.path.join(data_prefix, 'vocabulary.dat'), 'w')
for word in dictionary.values ():
    try:
        vocFile.write(word+'\n')
    except UnicodeEncodeError:
        vocFile.write('Unreadable\n')
        print(word)

vocFile.close()
print("Dictionary and Vocabulary files saved.")



import datetime
print(datetime.datetime.now())
timeslices = timeslices[:5]
numdocs = sum(timeslices)
corpus = corpus[:numdocs]
model = wrappers.DtmModel(dtm_path, corpus, timeslices, num_topics=50, id2word=dictionary, prefix=data_prefix,
                          initialize_lda=True)
print(datetime.datetime.now())

# model = wrappers.DtmModel(dtm_path, corpus, timeslices, num_topics=50, id2word=dictionary, prefix=data_prefix,
#                           initialize_lda=False)


# model.show_topics()
