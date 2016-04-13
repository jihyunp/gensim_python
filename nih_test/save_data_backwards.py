"""
Reverse the data
(time 0 starts from the late years)
"""

import numpy as np
import cPickle as cp
import os
import datetime
import sys


conf_file = sys.argv[1]
execfile(conf_file)

## Load doc
corpus = cp.load(open(gensim_X_file, 'rb'))
dictionary = cp.load(open(gensim_dict_file, 'rb'))

# Time slices
years = cp.load(open(year_pkl_file,'rb'))
for idx in np.argwhere(years==0).T[0]:
    years[idx] = years[idx-1]


docs_tobe_removed = np.argwhere(years>2011)
years_2011 = np.delete(years, docs_tobe_removed)
corpus_2011 = np.delete(corpus, docs_tobe_removed)



maxyear = int(np.max(years)) 
minyear = int(np.min(years))
hist = np.histogram(years, bins=maxyear-minyear+1, range=(minyear, maxyear+1) )  


# Only take the docs upto 2011
corpus = corpus[::-1]


# save the reversed version
new_gensim_X_file = '/home/jihyunp/research/topicmodel/data/150K/bow_sparse_gensim_reverse_upto2011.pkl'
new_gensim_years_file = '/home/jihyunp/research/topicmodel/data/150K/did_years_reverse_upto2011.pkl'
cp.dump(corpus_2011, open(new_gensim_X_file,'wb'))
cp.dump(years_2011, open(new_gensim_years_file,'wb'))
