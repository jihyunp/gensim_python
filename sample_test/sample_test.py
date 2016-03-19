from gensim import corpora, models, similarities
import gensim.models.wrappers as wrappers 
import numpy as np

data_prefix = '/Users/jihyun/Documents/jihyun/research/topicmodel/codes/dtm/gensim_python/data'

## Sample Doc
documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

## Sample Timeslices
timeslices = [1,2,1,1,1,1,1,1]


## Tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1]
         for text in texts]

from pprint import pprint   # pretty-printer
pprint(texts)

dictionary = corpora.Dictionary(texts)
dictionary.save(data_prefix + '/deerwester.dict') # store the dictionary, for future reference
#print(dictionary)

print(dictionary.token2id)

new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec) # the word "interaction" does not appear in the dictionary and is ignored


corpus = [dictionary.doc2bow(text) for text in texts]
#corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)
pprint(corpus)
pprint(len(corpus))


#dtm_path = '/Users/jihyun/Documents/jihyun/research/topicmodel/dtm/dtm-master/bin/dtm-darwin64'
dtm_path = '/Users/jihyun/Documents/jihyun/research/topicmodel/dtm/dtm-master/bin/dtm-darwin64'


model = wrappers.DtmModel(dtm_path, corpus, timeslices, num_topics=5, id2word=dictionary, prefix=data_prefix,
                          initialize_lda=True)



