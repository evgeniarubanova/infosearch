import os
import re
import nltk
from nltk import ngrams
from nltk.text import Text
import pymorphy2 as pm2
pmm = pm2.MorphAnalyzer()
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
import numpy as np
from collections import OrderedDict
import json
import math
import csv

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


def get_words(corp):
    X = vectorizer.fit_transform(corp)
    #уникальные слова в корпусе
    words = vectorizer.get_feature_names()
    
    matrix = X.toarray()
    #обратная матрица
    rev_matrix = np.transpose(matrix)

    return words, rev_matrix


def count_numbs_and_TFs():

    corp, quer = get_corp_and_quers()
    words, rev_matrix = get_matrix(corp)
    number_of_docs = json.load(open(r'C:\Users\ASUS\Desktop\комплинг\project\number_of_docs.json'))
    
    with open(r'C:\Users\ASUS\Desktop\комплинг\project\TFs.csv', 'r', encoding = 'utf-8') as r:
        reader = csv.reader(r)
        str_file = list(reader)
        
        for row in str_file:
            del row[-1]
        
        TFs = [[float(s) for s in sublist] for sublist in str_file] 
        
        return number_of_docs, TFs


def count_vars(corp, quer):
    #кол-во документов в коллекции
    N = len(corp)

    #длины документов
    lens_d = []
    for doc in corp:
        doc = doc.split()
        lens_d.append(len(doc))

    return N, lens_d

def count_IDFs():
    with open(r'C:\Users\ASUS\Desktop\комплинг\project\IDFs.csv', 'r', encoding = 'utf-8') as r:
        reader = csv.reader(r)
        str_file = list(reader)
        
        IDFs = [float(s[0]) for s in str_file] 
        
        return IDFs

def get_matrix():
    with open(r'C:\Users\ASUS\Desktop\комплинг\project\tf_idf.csv', 'r', encoding = 'utf-8') as r:
        reader = csv.reader(r)
        str_file = list(reader)

        for row in str_file:
            del row[-1]
        
        matrix = [[float(s) for s in sublist] for sublist in str_file]
        
        return matrix

q = 'не хочу учиться, хочу отдыхать'

def tf_idf_search(q):
    corp, quer = get_corp_and_quers()
    words, rev_matrix = get_words(corp)
    matrix = get_matrix()
    docs = get_docs()
    
    tf_idfs = []

    q = re.sub(r'[^\w\s]', '', q)
    q = re.sub(r'\d', '', q)
    q = re.sub(r'[A-Za-z]', '', q)
    q_arr = [pmm.normal_forms(x)[0] for x in q.split() if x not in russian_stopwords]

    for i in q_arr:
        for indx, word in enumerate(words):
            if i == word:
                tf_idf = matrix[indx]
                tf_idfs.append(tf_idf)

    if len(tf_idfs) > 0:
        q_tidf = sum(np.array(tf_idfs))
        rev_m = np.transpose(matrix)
        scores = q_tidf.dot(rev_m)
        for_range = list(zip(scores, docs))
        sort_range = sorted(for_range, reverse=True)
        
        res = sort_range[:10]

    return res
    
#tf_idf_search(q)

##def main():
##    corpus, eps = prep()
##    words, rev_matrix = index(corpus)
##    arr = rel(words)
##    search(words, rev_matrix, eps, arr)
##    stats(words, rev_matrix)
##    freq_words(words, rev_matrix)
##    pop_seasons(words)
##
##
##if __name__ == "__main__":
##    main()
