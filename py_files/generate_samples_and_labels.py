# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random

import cv2
import numpy as np
from libtiff import TIFF
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from gensim import corpora, models

import py_files.feature as feature


def extract_features(img_array, cell_size, step):
    band_count = img_array.shape[2]
    spec_feature = np.zeros((1, 2*band_count))
    glcm_feature = np.zeros((1, 1*4*6*band_count))
    reg_rows, reg_cols, cha = img_array.shape
    for i in range(0, reg_rows, step):
        for j in range(0, reg_cols, step):
            if (i == reg_rows-1) or (j == reg_cols-1):
                continue
            rect = generate_rect(i, j, reg_rows, reg_cols, cell_size)
            data_arr = img_array[rect[0]:rect[2], rect[1]:rect[3], :].copy()

            feature_obj = feature.Feature(data_arr)
            cur_spec_feature = feature_obj.get_spectral()
            cur_glcm_feature = feature_obj.get_glcm([2], prop_name=('contrast', 'dissimilarity','homogeneity',
                                                                           'energy', 'correlation', 'ASM'))

            spec_feature = np.concatenate((spec_feature, cur_spec_feature[np.newaxis, :]), axis=0)
            glcm_feature = np.concatenate((glcm_feature, cur_glcm_feature[np.newaxis, :]), axis=0)
    spec_feature = np.delete(spec_feature, 0, axis=0)
    glcm_feature = np.delete(glcm_feature, 0, axis=0)
    return spec_feature, glcm_feature


def generate_rect(org_row, org_col, rows, cols, width):
    end_row = org_row+width
    end_col = org_col+width
    if end_row > rows:
        end_row = rows
    if end_col > cols:
        end_col = cols
    rect = (org_row, org_col, end_row, end_col)
    return rect


def train_cluster(feature_arrays, cluster_num=10):
    feature_list = []
    for arr in feature_arrays:
        feature_list += arr.tolist()

    if len(feature_list) > 60000:
        feature_list = random.sample(feature_list, 60000)

    cluster = KMeans(cluster_num)
    cluster.fit(feature_list)
    print(cluster.cluster_centers_)
    print(cluster.inertia_)
    return cluster


def train_lda(documents, num_topics=4, alpha='auto'):
    dictionary = corpora.Dictionary(documents)
    print('\t', dict(dictionary))
    corpus = [dictionary.doc2bow(doc) for doc in documents]
    print('\t', corpus)
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, alpha=alpha)
    return dictionary, lda


def generate_libsvm_format(doc, dictionary, lda_model):
    feature_str = ''
    corpus = dictionary.doc2bow(doc)
    dis = lda_model.get_document_topics(corpus)
    dis_dict = dict(dis)
    topic_num = lda_model.num_topics
    for i in range(0, topic_num):
        if i in dis_dict.keys():
            value = dis_dict[i]
            feature_str += '\t' + str(i+1) + ':' + ('%.10f' % value)
    return feature_str


def resize(data_array, width, height, interpolation='nn'):

    """ 重采样 """

    if interpolation == 'nn':
        inter_type = cv2.INTER_NEAREST
    elif interpolation == 'linear':
        inter_type = cv2.INTER_LINEAR
    elif interpolation == 'cubic':
        inter_type = cv2.INTER_CUBIC
    else:
        raise ValueError('Wrong interpolation value: %s' % interpolation)

    c, w, h = data_array.shape
    res = np.zeros((c, width, height), dtype='uint8')
    for i in range(0, c):
        res[i] = cv2.resize(data_array[i], (width, height), interpolation=inter_type)
    return res


# ===================Parameters Setting===========================
SAMPLE_DIR = r'D:\PCFiles\Desktop\Temp\homeworks\rs_homework\test_samples'
MODEL_SAVE_DIR = r'..\models'
FILE_SAVE_PATH = r'..\test2.txt'
POSITIVE_NAME = 'huapo'

CELL_SIZE = 100
STEP = 40
CLUSTER_NUM = [100, 100]
TOPIC_NUM = 10

CLUSTER_RETRAIN = False
LDA_RETRAIN = False

# ===================Program Action===============================

# -------------------Features Extraction--------------------------
labels = []
spec_fs = []
text_fs = []
visual_documents = []
g = os.walk(SAMPLE_DIR)
for root, dirs, files in g:
    if root.split('\\')[-1] == POSITIVE_NAME:
        label = 1
    else:
        label = 0
    for file in files:
        if file[-4:] != '.tif':
            continue

        print('Processing %s' % file)

        # Read tiff_type data
        print('Reading images ...')
        img_path = os.path.join(root, file)
        tiff = TIFF.open(img_path, mode='r')
        img_arr = tiff.read_image()
        print(img_arr.shape)
        img_arr = img_arr.transpose((2, 0, 1))
        img_arr = resize(img_arr, 200, 200)

        # Extract three types of features of the current image
        print('Extracting features ...')
        spec_f, text_f = extract_features(img_arr, CELL_SIZE, STEP)
        labels.append(label)
        spec_fs.append(spec_f)
        text_fs.append(text_f)

# -------------------Features Abstraction and Samples Generation--
# Train cluster models for features above
if CLUSTER_RETRAIN:
    print('Clustering ...')
    spec_clt = train_cluster(spec_fs, CLUSTER_NUM[0])
    text_clt = train_cluster(text_fs, CLUSTER_NUM[1])
    joblib.dump(spec_clt, os.path.join(MODEL_SAVE_DIR, 'spec_clt.pkl'))
    joblib.dump(text_clt, os.path.join(MODEL_SAVE_DIR, 'text_clt.pkl'))
else:
    print('Loading cluster models ...')
    spec_clt = joblib.load(os.path.join(MODEL_SAVE_DIR, 'spec_clt.pkl'))
    text_clt = joblib.load(os.path.join(MODEL_SAVE_DIR, 'text_clt.pkl'))


# Transform cluster results to documents for topic model training
print('Transforming ...')
training_num = len(labels)
for i in range(0, training_num):
    spec_cluster_res = spec_clt.predict(spec_fs[i])
    text_cluster_res = text_clt.predict(text_fs[i])
    document = spec_cluster_res.astype(str).tolist()
    text_cluster_res += CLUSTER_NUM[0]
    document += text_cluster_res.astype(str).tolist()
    visual_documents.append(document)

# Train LDA models based documents above
if LDA_RETRAIN:
    print('Training LDA model ...')
    visual_dict, visual_lda = train_lda(visual_documents, num_topics=TOPIC_NUM)
    visual_lda.save(os.path.join(MODEL_SAVE_DIR, 'visual_lda.model'))
    visual_dict.save(os.path.join(MODEL_SAVE_DIR, 'visual_dict'))
else:
    print('Loading LDA model ...')
    visual_lda = models.LdaModel.load(os.path.join(MODEL_SAVE_DIR, 'visual_lda.model'))
    visual_dict = corpora.Dictionary.load(os.path.join(MODEL_SAVE_DIR, 'visual_dict'))

# Generate sample file in libsvm format
print('Generating libsvm format file ...')
with open(FILE_SAVE_PATH, 'w') as t_f:
    for i in range(0, training_num):
        line_str = str(labels[i])
        visual_str = generate_libsvm_format(visual_documents[i], visual_dict, visual_lda)
        line_str += visual_str
        line_str += '\n'
        t_f.write(line_str)

print('Mission Completed!')
