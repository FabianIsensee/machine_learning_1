from libc.stdlib cimport calloc, free
from libc.string cimport memcpy
import numpy as np
import cython
cimport numpy as np
import math

ctypedef np.float_t FTYPE_t
ctypedef np.int_t DTYPE_t

@cython.boundscheck(False) # turn of bounds-checking for entire function
def split_gini(np.ndarray[FTYPE_t, ndim=1] feat_vec,
                 np.ndarray[DTYPE_t, ndim=1] labels,
                 np.ndarray[DTYPE_t, ndim=1] classes,
                 np.ndarray[DTYPE_t, ndim=1] class_distrib):

    cdef int n_instances = labels.shape[0]
    cdef float best_split_score = 9999999999
    cdef float best_split_threshold = 9999999999

    cdef int n_classes = len(classes)

    cdef np.ndarray[DTYPE_t, ndim=1] idx_sorted_by_feature = np.argsort(feat_vec).astype("int")
    cdef int left_idx = 0
    cdef int right_idx
    cdef int *class_distr_left = <int*>calloc(n_classes, sizeof(int))
    cdef int *class_distr_right = <int*>calloc(n_classes, sizeof(int))

    cdef int i, j, k

    cdef int n_idx_left, n_idx_right
    cdef float p_left, p_right, p_k_left, p_k_right, gini_coeff

    cdef np.ndarray[FTYPE_t, ndim=1] res = np.zeros(2).astype("float")

    try:
        memcpy(class_distr_right, &class_distrib[0], n_classes * sizeof(int))

        idx_sorted_by_feature = np.argsort(feat_vec)

        while left_idx < (n_instances - 1):
            class_distr_left[labels[idx_sorted_by_feature[left_idx]]] += 1
            class_distr_right[labels[idx_sorted_by_feature[left_idx]]] -= 1
            right_idx = left_idx + 1
            while left_idx < (n_instances - 1) and \
                (feat_vec[idx_sorted_by_feature[left_idx]] == feat_vec[idx_sorted_by_feature[right_idx]]):
                left_idx += 1
                right_idx += 1
                class_distr_left[labels[idx_sorted_by_feature[left_idx]]] += 1
                class_distr_right[labels[idx_sorted_by_feature[left_idx]]] -= 1

            if right_idx > n_instances - 1:
                break

            n_idx_left = left_idx + 1
            n_idx_right = n_instances - n_idx_left
            p_left = <float> n_idx_left / <float>n_instances
            p_right = 1. - p_left
            gini_coeff = 0

            for k in range(n_classes):
                p_k_left = <float> class_distr_left[k] / <float> n_idx_left
                p_k_right = <float> class_distr_right[k] / <float> n_idx_right
                gini_coeff += p_left * p_k_left * (1. - p_k_left) + p_right * p_k_right * (1. - p_k_right)

            if best_split_score > gini_coeff:
                best_split_score = gini_coeff
                best_split_threshold = (<float> feat_vec[idx_sorted_by_feature[left_idx]] + <float> feat_vec[idx_sorted_by_feature[left_idx]]) / 2.

            left_idx += 1

    finally:
        free(class_distr_left)
        free(class_distr_right)

    res[0] = best_split_score
    res[1] = best_split_threshold
    return res

