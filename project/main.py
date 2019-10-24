from bm25 import bm25_search
from fasttext import fasttext_search
from tf_idf import tf_idf_search

import logging
from flask import Flask, request, render_template
app = Flask(__name__, template_folder="templates")

logging.basicConfig(filename="search_eng.log", level=logging.DEBUG)
logging.info('The program started')

@app.route('/')
def init():
    logging.info("The main page's opened")
    return render_template('main.html')


@app.route('/results')
def res():
    query = request.args['query']
    method = request.args['method']
    result= ''

    if method == 'TF-IDF':
        logging.info("Processing TF-IDF request")
        result = tf_idf_search(query)
        logging.info("Got the results")
    
    elif method == 'BM25':
        logging.info("Processing BM25 request")
        result = bm25_search(query)
        logging.info("Got the results")
        
    elif method == 'Fasttext':
        logging.info("Processing Fasttext request")
        result = fasttext_search(query)
        logging.info("Got the results")
        
    return render_template('res.html', query = query, method = method, result = result)
    


if __name__ == "__main__":
    app.run()
