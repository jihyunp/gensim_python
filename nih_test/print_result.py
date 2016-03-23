"""
This script is called right after running 'nih_test_150K.py' script.
model should have the type <gensim.models.wrappers.dtmmodel.DtmModel object at 0x121c3eb50>

March 22, 2016
Jihyun Park
jihyunp@uci.edu
"""

import numpy as np
import matplotlib.pyplot as plt


total_num_topics = model.num_topics
total_vocab_size = model.num_terms
num_timeslices = len(model.time_slices)


dictionary_word2id = dict((value, key) for key, value in dictionary.iteritems())

# Print out the top words with probabilities
outfile1 = open(data_prefix + 'topic_output_with_prob.txt', 'w')
outfile2 = open(data_prefix + 'topic_output.txt', 'w')
for tid in range(total_num_topics):
    outfile1.write("\n-----Topic "+str(tid) + "-----\n")
    outfile2.write("\n-----Topic "+str(tid) + "-----\n")
    for timeid in range(num_timeslices):
        outfile1.write('--Year: ' + str(yid_year[timeid]) + '\n')
        outfile2.write('--Year: ' + str(yid_year[timeid]) + '\n')
        topics_with_prob = model.show_topic(topicid=tid, time=timeid, topn=12)
        for tup in topics_with_prob:
            prob = tup[0]
            word = tup[1]
            outfile1.write("{} ({})  ".format(word, "%.4f" % prob ))
            outfile2.write("{}  ".format(word))
        outfile1.write('\n')
        outfile2.write('\n')


""" Below code can be removed """
#
#
#
#
# # Plot
# year_label = map(str, np.array(yid_year[::5], dtype=int))
#
# for tid in range(total_num_topics):
# # tid = 0
#     word2problist = {}
#     top_wlist = []
#     cnt = 0
#     for timeid in range(len(yid_year)-1):
#         cnt += 1
#         topics_with_prob = model.show_topic(topicid=tid, time=timeid, topn=10)
#         for tup in topics_with_prob:
#             prob = tup[0]
#             word = tup[1]
#             if word not in word2problist:
#                 word2problist[word] = np.zeros(len(yid_year)-1)
#             word2problist[word][timeid] = prob
#             # Only consider the first 5 words. This will be called later
#             if cnt < 6:
#                 top_wlist.append(word)
#
#     linelist = []
#     cnt = 0
#
#     fig, axes = plt.subplots()
#     for word in top_wlist:
#         if cnt == 5:
#             break
#         cnt += 1
#         problist = word2problist[word]
#         line, = axes.plot(problist,  'o-', label=word)
#         linelist.append(line)
#     axes.legend(handles=linelist)
#     axes.set_xticklabels(year_label)
#     # fig.savefig(data_prefix + 'topic' + str(tid) +'.pdf')
#     fig.clf()
#
#
# plt.close("all")


## Get marginal probabilities for those words
# For each year, how the p(w) = sum_t p(w|t) changes over year

topic_wordpatterns = []
marginal_wordpatterns = {}
top_wlist = []
wset= set()

for yid in range(num_timeslices):
    print('--- time: ' + str(yid))

    # lambda_ has shape [topic, word, time]
    # Normalized prob matrix
    probmat = np.exp(model.lambda_[:, :, yid])
    normvec = np.sum(probmat, axis=1)  # Normalize const
    probmat = probmat / np.tile(normvec,(total_vocab_size,1)).T

    # Get the top 5 word lists for each topic
    for tid in range(total_num_topics):
        if yid == 0:
            top_wlist.append([])  # top n word list for each topic
        # Save just top 5
        topwords_w_prob = model.show_topic(topicid=tid, time=yid, topn=5)
        for tup in topwords_w_prob:
            word = tup[1]
            wid = dictionary_word2id[word]
            if yid == 0:
                top_wlist[tid].append(word)
                wset.add(word)

    # Get topic-word distribution
    for tid in range(total_num_topics):
        wlist = top_wlist[tid]
        if yid == 0:
            topic_wordpatterns.append({})
        for word in wlist:
            if yid == 0:
                topic_wordpatterns[tid][word] = []
            wid = dictionary_word2id[word]
            topic_wordpatterns[tid][word].append(probmat[tid, wid])

    # Get marginal probabilities
    marg_wordprob = np.sum(probmat, axis=0)
    marg_wordprob = marg_wordprob / marg_wordprob.sum()
    for word in wset:
        if yid == 0:
            marginal_wordpatterns[word] = []
        wid = dictionary_word2id[word]
        marginal_wordpatterns[word].append(marg_wordprob[wid])



# Plot p(w|t) for each topic for top n words
for tid in range(total_num_topics):
    wlist = top_wlist[tid]
    linelist = []
    fig, axes = plt.subplots()
    for word in wlist:
        # problist = topic_wordpatterns[tid][word]
        problist = np.array(topic_wordpatterns[tid][word])
        line, = axes.plot(problist,  'o-', label=word)
        linelist.append(line)
    axes.legend(handles=linelist)
    axes.set_xticklabels(year_label)
    axes.set_title('Word Probabilities for Topic '+ str(tid))
    fig.savefig(data_prefix + 'topic' + str(tid) +'.pdf')
    fig.clf()
plt.close("all")


# Plot marginal p(w) for each topic for top n words
for tid in range(total_num_topics):
    wlist = top_wlist[tid]
    linelist = []
    fig, axes = plt.subplots()
    for word in wlist:
        # problist = topic_wordpatterns[tid][word]
        problist = np.array(marginal_wordpatterns[word])
        line, = axes.plot(problist,  'o-', label=word)
        linelist.append(line)
    axes.legend(handles=linelist)
    axes.set_xticklabels(year_label)
    axes.set_title('Marginal probabilities')
    fig.savefig(data_prefix + 'topic' + str(tid) +'_marginal.pdf')
    fig.clf()
plt.close("all")







## Divide Probability of word given topic with Marginal Probability of the word
# p(w|t) = p(t|w) * p(w) / p(t)
# p(w) = p(t) * p(w|t) / p(t|w)
# p(w|t) / p(w) = p(t|w) / p(t) = p(
for tid in range(total_num_topics):
    wlist = top_wlist[tid]
    linelist = []
    fig, axes = plt.subplots()
    for word in wlist:
        # problist = topic_wordpatterns[tid][word]
        problist = np.array(topic_wordpatterns[tid][word]) / np.array(marginal_wordpatterns[word]) / total_num_topics
        line, = axes.plot(problist,  'o-', label=word)
        linelist.append(line)
    axes.legend(handles=linelist)
    axes.set_xticklabels(year_label)
    fig.savefig(data_prefix + 'topic' + str(tid) +'_divbymarginal.pdf')
    fig.clf()
plt.close("all")






