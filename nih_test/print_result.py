import numpy as np



outfile = open(data_prefix + 'topic_output.txt', 'w')
for tid in range(20):
    outfile.write("\n-----Topic "+str(tid) + "-----\n")
    for timeid in range(4):
        outfile.write('--Year: ' + str(yid_year[timeid]) + '\n')
        topics_with_prob = model.show_topic(topicid=tid, time=timeid, topn=12)
        for tup in topics_with_prob:
            prob = tup[0]
            word = tup[1]
            outfile.write("{} ({})  ".format(word, "%.4f" % prob ))
        outfile.write('\n')


# Plot
import matplotlib.pyplot as plt


year_label = map(str, np.array(yid_year[::5], dtype=int))
for tid in range(60):
# tid = 0
    word2problist = {}
    top_wlist = []
    cnt = 0
    for timeid in range(len(yid_year)-1):
        cnt += 1
        topics_with_prob = model.show_topic(topicid=tid, time=timeid, topn=70)
        for tup in topics_with_prob:
            prob = tup[0]
            word = tup[1]
            if word not in word2problist:
                word2problist[word] = np.zeros(len(yid_year)-1)
            word2problist[word][timeid] = prob
            if cnt < 6:
                top_wlist.append(word)

    linelist = []
    cnt = 0

    fig, axes = plt.subplots()
    for word in top_wlist:
        if cnt == 5:
            break
        cnt += 1
        problist = word2problist[word]
        line, = axes.plot(problist,  'o-', label=word)
        linelist.append(line)
    axes.legend(handles=linelist)
    axes.set_xticklabels(year_label)
    fig.savefig(data_prefix + 'topic' + str(tid) +'.pdf')
    fig.clf()

plt.close("all")