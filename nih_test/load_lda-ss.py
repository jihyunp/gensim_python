import numpy as np
vt_vector = np.loadtxt('initial-lda-ss.dat')
ntopics = 80
nvocab = 29713
tv_mat = np.zeros((ntopics, nvocab))

vid = 0
tid = 0
for i in vt_vector:
    tv_mat[tid,vid] = i
    tid += 1
    if tid == ntopics:
        tid = 0
        vid += 1



