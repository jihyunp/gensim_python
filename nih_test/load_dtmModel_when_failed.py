# When the training finished without having alpha/gamma files


from gensim import corpora, models, similarities
#import gensim.models.wrappers as wrappers 
import dtmmodel
import numpy as np
import cPickle as cp
import os
import datetime


#dtm_path = '/Users/jihyun/Documents/jihyun/research/topicmodel/dtm/dtm-master/bin/dtm-darwin64'
dtm_path = '/home/jihyunp/research/topicmodel/dtm/dtm_release/dtm/main'

#data_prefix = '/Users/jihyun/Documents/jihyun/research/topicmodel/codes/dtm/gensim_python/nih_test/data_150K_4time/'
data_prefix = '/home/jihyunp/research/topicmodel/codes/dtm/gensim_python/nih_test/data/data_150K_init/'
data_train = '/home/jihyunp/research/topicmodel/codes/dtm/gensim_python/nih_test/data/data_150K_init/train_out/'
if not os.path.exists(data_train):
    os.makedirs(data_train)


lda_nzw_file = '/home/jihyunp/research/topicmodel/data/150K/subset1/nwz_gls_form.dat'

from subprocess import call
call(["cp", lda_nzw_file, data_train + 'initial-lda-ss.dat'])
call(["touch", data_train + 'em_log.dat'])


## Sample Doc (load doc)

#gensim_X_file = '/Users/jihyun/Documents/jihyun/research/topicmodel/data/150K/bow_sparse_gensim.pkl'
#gensim_dict_file = '/Users/jihyun/Documents/jihyun/research/topicmodel/data/150K/wid_word_4timeslices.pkl'

gensim_X_file = '/home/jihyunp/research/topicmodel/data/150K/bow_sparse_gensim.pkl'
gensim_dict_file = '/home/jihyunp/research/topicmodel/data/150K/wid_word.pkl'



corpus = cp.load(open(gensim_X_file, 'rb'))
dictionary = cp.load(open(gensim_dict_file, 'rb'))

## Sample Timeslices
year_pkl_file = '/home/jihyunp/research/topicmodel/data/150K/did_year.pkl'

years = cp.load(open(year_pkl_file,'rb'))
for idx in np.argwhere(years==0).T[0]:
    years[idx] = years[idx-1]
import matplotlib.pyplot as plt
maxyear = int(np.max(years))
minyear = int(np.min(years))


hist = np.histogram(years, bins=maxyear-minyear+1, range=(minyear, maxyear+1) )
timeslices = hist[0]
yid_year = hist[1]
print(timeslices)


#print(minyear)
#print(maxyear)
#plt.hist(years, bins=maxyear-minyear, range=(minyear, maxyear))
#plt.xlabel(np.unique(years))
#plt.savefig('fig.pdf', format='pdf')

print("Data loading finished")















print(datetime.datetime.now())

model = dtmmodel.DtmModel(dtm_path, corpus=None, time_slices=timeslices, num_topics=98, id2word=dictionary, prefix=data_prefix,initialize_lda=False)


model.lencorpus = len(corpus)
model.convert_input(corpus, timeslices)


#model.em_steps = np.loadtxt(model.fem_steps())
#model.init_alpha = np.loadtxt(model.finit_alpha())
#model.init_beta = np.loadtxt(model.finit_beta())
model.init_ss = np.loadtxt(model.flda_ss())


model.gamma_ = np.loadtxt(model.fout_gamma())
# cast to correct shape, gamme[5,10] is the proprtion of the 10th topic
# in doc 5
model.gamma_.shape = (model.lencorpus, model.num_topics)
# normalize proportions
model.gamma_ /= model.gamma_.sum(axis=1)[:, np.newaxis]


print(datetime.datetime.now())
print('from lambda')

model.lambda_ = np.zeros((model.num_topics, model.num_terms * len(model.time_slices)))
model.obs_ = np.zeros((model.num_topics, model.num_terms * len(model.time_slices)))

print('last step!')
for t in range(model.num_topics):
        topic = "%03d" % t
        model.lambda_[t, :] = np.loadtxt(model.fout_prob().format(i=topic))
        model.obs_[t, :] = np.loadtxt(model.fout_observations().format(i=topic))
# cast to correct shape, lambda[5,10,0] is the proportion of the 10th
# topic in doc 5 at time 0
model.lambda_.shape = (model.num_topics, model.num_terms, len(model.time_slices))
model.obs_.shape = (model.num_topics, model.num_terms, len(model.time_slices))



execfile('print_result.py')
