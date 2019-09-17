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

russian_stopwords = ['и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как',
                         'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у',
                         'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот',
                         'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда',
                         'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть',
                         'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь',
                         'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где',
                         'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам',
                         'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет',
                         'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой',
                         'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем',
                         'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда',
                         'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над',
                         'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая',
                         'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою',
                         'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой',
                         'им', 'более', 'всегда', 'конечно', 'всю', 'между']


def prep():
    episodes = []
    eps = []
    for root, dirs, files in os.walk('friends'):
        for name in files:
            episodes.append(os.path.join(root, name))
            eps.append(os.path.join(name))
    #словарь, в котором ключи - названия серий, а значения - список лемм-ых слов серии
    d = {}
    for i in episodes:
        with open(i, 'r', encoding = 'utf-8') as r:
            text = r.read()
            text = re.sub(r'[^\w\s]','',text)
            text = re.sub(r'\d', '', text)
            text = re.sub(r'[A-Za-z]', '', text)
            text = [pmm.normal_forms(x)[0] for x in text.split() if x not in russian_stopwords]
            d[i] = text
    #массив, в котором каждый элемент - все слова серии
    corpus = []
    for key, value in d.items():
        ep = ' '.join(value)
        corpus.append(ep)
    return corpus, eps


def index(corpus):
    X = vectorizer.fit_transform(corpus)
    #матрица с количеством вхождений слов
    matrix = X.toarray()
    #обратная матрица
    rev_matrix = np.transpose(matrix)
    words = vectorizer.get_feature_names()
    return words, rev_matrix


def rel(words):
    query = input('Введите запрос: ')
    query = re.sub(r'[^\w\s]','',query)
    query = re.sub(r'\d', '', query)
    q_norm = [pmm.normal_forms(x)[0] for x in query.split() if x not in russian_stopwords]
    arr = []
    #массив по запросу из нолей и единиц
    for word in words:
        if word in q_norm:
            arr.append(1)
        else:
            arr.append(0)
    return arr


def search(words, rev_matrix, eps, arr):
    rez = []
    for i in range(165):
        count = 0
        m = rev_matrix[:,i]
        for j in range(15240):
            if arr[j] == 1 and m[j] != 0:
                count += 1
        rez.append(count)
    final = dict(zip(eps, rez))
    sorted_final = OrderedDict(sorted(final.items(), key=lambda kv: kv[1], reverse=True))
    print('\n')
    print('Список серий по релевантности: ')
    for key, value in sorted_final.items():
        print(key)


def stats(words, rev_matrix):
    #подсчет самого частотного и самого редкого слова
    sums = []
    for row in rev_matrix:
        sums.append(sum(row))
    max_value = max(sums)
    max_index = sums.index(max_value)
    min_value = min(sums)
    min_index = sums.index(min_value)
    print('\n')
    print('Самое частотное слово - ' + words[max_index])
    print('\n')
    print('Самое редкое слово - ' + words[min_index])
    #самый популярный герой
    names = dict(zip(words, rev_matrix))
    n_sums = {}
    n_sums['Росс'] = sum(names['росс'])
    rachel = sum(names['рейч']) + sum(names['рэйчел'])
    n_sums['Рэйчел'] = rachel
    n_sums['Моника'] = sum(names['моника'])
    n_sums['Фиби'] = sum(names['фиби'])
    n_sums['Чендлер'] = sum(names['чендлер'])
    joe = sum(names['джо']) + sum(names['джоу'])
    n_sums['Джо'] = joe
    sorted_n_sums = OrderedDict(sorted(n_sums.items(), key=lambda kv: kv[1], reverse=True))
    print('\n')
    print('Статистически самый популярный герой "Друзей" - ' + list(sorted_n_sums.keys())[0])


def freq_words(words, rev_matrix):
    every_indx = []
    for indx, row in enumerate(rev_matrix):
        count = 0
        for i in row:
            if i != 0:
                count += 1
        if count == 165:
            every_indx.append(indx)
    print('\n')
    print('Слова, встречающиеся во всех сериях: ')
    for i in every_indx:
        print(words[i])


def pop_seasons(words):
    seasons = []
    for root, dirs, files in os.walk("friends"):
        for d in dirs:
            path = os.path.abspath('friends') + '\\' + d
            arr = []
            for i in os.listdir(path):
                path = os.path.abspath('friends') + '\\' + d + '\\' + i
                arr.append(path)
            seasons.append(arr)

    lemmas = []
    for season in seasons:
        s = []
        for ep in season:
            with open(ep, 'r', encoding = 'utf-8') as r:
                text = r.read()
                text = re.sub(r'[^\w\s]','',text)
                text = re.sub(r'\d', '', text)
                text = re.sub(r'[A-Za-z]', '', text)
                text = [pmm.normal_forms(x)[0] for x in text.split() if x not in russian_stopwords]
                e = ' '.join(text)
            s.append(e)
        l = ' '.join(s)
        lemmas.append(l)
        
    Y = vectorizer.fit_transform(lemmas)
    s_matrix = Y.toarray()
    s_rev_matrix = np.transpose(s_matrix)

    ch = words.index('чендлер')
    m = words.index('моника')

    print('\n')
    for indx, i in enumerate(s_rev_matrix[m]):
        if i == max(s_rev_matrix[m]):
            print('Самый популярный сезон у Моники - №', indx+1)

    print('\n')
    for indx, i in enumerate(s_rev_matrix[ch]):
        if i == max(s_rev_matrix[ch]):
            print('Самый популярный сезон у Чендлера - №', indx+1)
    

def main():
    corpus, eps = prep()
    words, rev_matrix = index(corpus)
    arr = rel(words)
    search(words, rev_matrix, eps, arr)
    stats(words, rev_matrix)
    freq_words(words, rev_matrix)
    pop_seasons(words)


if __name__ == "__main__":
    main()
