import math
from math import log
import re
import json

import pymorphy2 as pm2
pmm = pm2.MorphAnalyzer()
import csv

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
russian_stopwords = stopwords.words("russian")

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
import numpy as np


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


def count_vars(corp, quer):
    #кол-во документов в коллекции
    N = len(corp)

    #длины документов
    lens_d = []
    for doc in corp:
        doc = doc.split()
        lens_d.append(len(doc))

    #средняя длина документа
    avgdl = sum(lens_d)/len(lens_d)

    #константы
    k = 2
    b = 0.75

    return N, lens_d, avgdl, k, b


def get_matrix(corp):
    X = vectorizer.fit_transform(corp)
    #уникальные слова в корпусе
    words = vectorizer.get_feature_names()
    
    matrix = X.toarray()
    #обратная матрица
    rev_matrix = np.transpose(matrix)

    return words, rev_matrix


def count_numbs_and_TFs(words, corp, rev_matrix):
    number_of_docs = json.load(open(r'C:\Users\ASUS\Desktop\комплинг\project\number_of_docs.json'))
    
    with open(r'C:\Users\ASUS\Desktop\комплинг\project\TFs.csv', 'r', encoding = 'utf-8') as r:
        reader = csv.reader(r)
        str_file = list(reader)
        
        for row in str_file:
            del row[-1]
        
        TFs = [[float(s) for s in sublist] for sublist in str_file] 
        
        return number_of_docs, TFs


def bm_matrix():
    with open(r'C:\Users\ASUS\Desktop\комплинг\project\bm_matrix.csv', 'r', encoding = 'utf-8') as r:
        reader = csv.reader(r)
        str_file = list(reader)
        
        for row in str_file:
            del row[-1]
        
        file = [[float(s) for s in sublist] for sublist in str_file] 
        return file

q = 'не хочу учиться, хочу отдыхать'

def bm25_search(q):
    
    corp, quer = get_corp_and_quers()
    words, rev_matrix = get_matrix(corp)
    N, lens_d, avgdl, k, b = count_vars(corp, quer)
    number_of_docs, TFs = count_numbs_and_TFs(words, corp, rev_matrix)
    bm_mtrx = bm_matrix()
    docs = get_docs()
    
    bms = []

    q = re.sub(r'[^\w\s]', '', q)
    q = re.sub(r'\d', '', q)
    q = re.sub(r'[A-Za-z]', '', q)
    q_arr = [pmm.normal_forms(x)[0] for x in q.split() if x not in russian_stopwords]

    for i in q_arr:
        for indx, word in enumerate(words):
            if i == word:
                bm = bm_mtrx[indx]
                bms.append(bm)

    if len(bms) > 0:
        q_bm = sum(np.array(bms))
        rev_m = np.transpose(bm_mtrx)
        scores = q_bm.dot(rev_m)
        for_range = list(zip(scores, docs))
        sort_range = sorted(for_range, reverse=True)
        
        res = sort_range[:10]

    return res

#bm25_search(q)
#def main():
#    corp, quer = get_corp_and_quers()
#    words, rev_matrix = get_matrix(corp)
#    N, lens_d, avgdl, k, b = count_vars(corp, quer)
#    number_of_docs, TFs = count_numbs_and_TFs(words, corp, rev_matrix)
#    bm_mtrx = bm_matrix(words, corp, lens_d, number_of_docs, TFs, N, k, b, avgdl)
#    docs = get_docs()
#    search(quer, words, bm_mtrx, docs)
#
#
#if __name__ == "__main__":
#    main()

bm25_search(q)
    
