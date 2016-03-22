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
import datetime


# dtm_path = '/Users/jihyun/Documents/jihyun/research/topicmodel/dtm/dtm-master/bin/dtm-darwin64'
dtm_path = '/Users/jihyun/Documents/jihyun/research/topicmodel/dtm/dtm_release/dtm/main'

data_prefix = '/Users/jihyun/Documents/jihyun/research/topicmodel/codes/dtm/gensim_python/nih_test/data_150K_4time/'
if not os.path.exists(data_prefix):
    os.makedirs(data_prefix)

## Sample Doc (load doc)

gensim_X_file = '/Users/jihyun/Documents/jihyun/research/topicmodel/data/150K/bow_sparse_gensim.pkl'
# gensim_dict_file = '/Users/jihyun/Documents/jihyun/research/topicmodel/data/150K/wid_word.pkl'
gensim_dict_file = '/Users/jihyun/Documents/jihyun/research/topicmodel/data/150K/wid_word_4timeslices.pkl'

corpus = cp.load(open(gensim_X_file, 'rb'))
dictionary = cp.load(open(gensim_dict_file, 'rb'))

## Sample Timeslices
year_file = '/Users/jihyun/Documents/jihyun/research/topicmodel/data/years_gensim.txt'
year_pkl_file = '/Users/jihyun/Documents/jihyun/research/topicmodel/data/150K/did_year.pkl'

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


timeslices = timeslices[:4]
numdocs = sum(timeslices)
corpus = corpus[:numdocs]


print(datetime.datetime.now())
print("Done with loading the data. Now start training")
model = wrappers.DtmModel(dtm_path, corpus, timeslices, num_topics=50, id2word=dictionary, prefix=data_prefix, initialize_lda=True)
print(datetime.datetime.now())

print("Saving model..")
cp.dump(model, open(data_prefix+'model.pkl', 'wb'))


# topics_with_prob = model.show_topics(topics=20, times=4, topn=12, formatted=False)

outfile = open(data_prefix + 'topic_output.txt', 'w')
for tid in range(20):
    outfile.write("\n-----Topic "+str(tid) + "-----\n")
    for timeid in range(4):
        outfile.write('--Year: ' + str(yid_year[timeid]) + '\n')
        topics_with_prob = model.show_topic(topicid=tid, time=timeid, topn=12)
        for tup in topics_with_prob:
            prob = tup[0]
            word = tup[1]
            outfile.write("{} ({})  ".format(word, "%.4f" % line[0][0] ))
        outfile.write('\n')


# Plot
import matplotlib.pyplot as plt

for tid in range(20):
# tid = 0
    word2problist = {}
    for timeid in range(4):
        topics_with_prob = model.show_topic(topicid=tid, time=timeid, topn=15)
        for tup in topics_with_prob:
            prob = tup[0]
            word = tup[1]
            if word not in word2problist:
                word2problist[word] = np.zeros(4)
            word2problist[word][timeid] = prob
    wlist = []
    linelist = []
    cnt = 0
    for word in word2problist.keys():
        if cnt == 5:
            break
        problist = word2problist[word]
        line, = plt.plot(problist,  'o-', label=word)
        linelist.append(line)
        wlist.append(word)
    plt.legend(handles=linelist)
    plt.savefig(data_prefix + 'topic' + str(tid) +'.pdf')
    plt.clf()


#
# print(dictionary.token2id)
#
# new_doc = "Human computer interaction"
# new_vec = dictionary.doc2bow(new_doc.lower().split())
# print(new_vec) # the word "interaction" does not appear in the dictionary and is ignored
#
#
#

# print(datetime.datetime.now())
# dtm_path = '/Users/jihyun/Documents/jihyun/research/topicmodel/dtm/dtm-master/bin/dtm-darwin64'
# model = wrappers.DtmModel(dtm_path, corpus, timeslices, num_topics=70, id2word=dictionary, prefix=data_prefix,
#                           initialize_lda=True)
#
# print("-------FINISHED-------")
# print(datetime.datetime.now())
