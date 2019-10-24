import csv
import json
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

import pymorphy2 as pm2
pmm = pm2.MorphAnalyzer()
import re

import numpy as np
model_file = r'C:\Users\ASUS\Desktop\комплинг\project\fasttext\model.model'
model_fasttext = KeyedVectors.load(model_file)

import nltk
from numpy import dot
from numpy.linalg import norm

nltk.download("stopwords")
from nltk.corpus import stopwords
russian_stopwords = stopwords.words("russian")


def get_docs():
    with open(r'C:\Users\ASUS\Desktop\комплинг\project\docs.txt', 'r', encoding = 'utf-8') as r:
        docs = r.readlines()
    return docs  
    

def get_corp_and_quers():
    with open(r'C:\Users\ASUS\Desktop\комплинг\project\corp.txt', 'r', encoding = 'utf-8') as r:
        corp = r.readlines()
        
    with open(r'C:\Users\ASUS\Desktop\комплинг\project\quer.txt', 'r', encoding = 'utf-8') as r:
        quer = r.readlines()
    return corp, quer


def get_matrix():
    corpus, quer = get_corp_and_quers()
    
    with open(r'C:\Users\ASUS\Desktop\комплинг\project\fasttext_matrix.csv', 'r', encoding = 'utf-8') as r:
        reader = csv.reader(r)
        str_file = list(reader)
        
        for row in str_file:
            del row[-1]
        
        matrix = [[float(s) for s in sublist] for sublist in str_file]
    return matrix

q = 'не хочу учиться, хочу отдыхать'

def fasttext_search(q):
    docs = get_docs()
    corpus, quer = get_corp_and_quers()
    matrix = get_matrix()

    q = re.sub(r'[^\w\s]', '', q)
    q = re.sub(r'\d', '', q)
    q = re.sub(r'[A-Za-z]', '', q)
    q_arr = [pmm.normal_forms(x)[0] for x in q.split() if x not in russian_stopwords]

    lemmas_vectors = np.zeros((len(q_arr), model_fasttext.vector_size))
    vec = np.zeros((model_fasttext.vector_size,))
    
    for idx, lemma in enumerate(q_arr):
        if lemma in model_fasttext.vocab:
            lemmas_vectors[idx] = model_fasttext.wv[lemma]
        
    if lemmas_vectors.shape[0] is not 0:
        vec = np.mean(lemmas_vectors, axis=0)
        
##    mult = vec.dot(rev_matrix)
##    for_range = list(zip(mult, docs))
##    sort_range = sorted(for_range, reverse=True)

    for_range = []
    for indx, doc in enumerate(docs):
        if indx < len(matrix):
            rev_m = np.transpose(matrix)
            score = dot(vec, rev_m[indx]) / (norm(vec) * norm(rev_m[indx]))
            for_range.append((score, docs[indx]))
            
    sort_range = sorted(for_range, reverse=True)
    res = sort_range[:10]
    return res

#fasttext_search(q)

