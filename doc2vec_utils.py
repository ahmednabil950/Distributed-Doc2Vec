from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import PathLineSentences
from random import shuffle, sample
from sklearn.metrics.pairwise import cosine_similarity
from termcolor import colored
from pathlib import Path
import pickle as pkl
import numpy as np
import os, sys
import argparse

def extract_documents():
    doc_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/tagged_docs.pickle'))
    with doc_path.open('rb') as docf:
        documents = pkl.load(docf)
    ## shuffle modify in place
    # return shuffle(documents)
    ## shuffle using sample
    return sample(documents, len(documents))

def compute_similarity(model, doc1, doc2):
    doc1_vec = model.infer_vector(doc1.split())
    doc1_vec = doc1_vec.reshape((1, doc1_vec.shape[0]))
    doc2_vec = model.infer_vector(doc2.split())
    doc2_vec = doc2_vec.reshape((1, doc2_vec.shape[0]))
    return cosine_similarity(doc1_vec, doc2_vec).flatten()

def load_model(model_path):
    model = Doc2Vec.load(os.path.join(
        os.path.dirname(os.path.abspath('__file__')), model_path)
    )
    return model

def print_paragraph(documents, doc_id):
    document = " ".join(documents[doc_id].words)
    print("Document ({}): <<{}>>".format(doc_id, document)+"\n")

def get_paragraph(documents, doc_id):
    document = " ".join(documents[doc_id].words)
    return document

def get_infered_vector(model, doc):
    return model.infer_vector(doc.split())

def most_similar_from_random(documents, model):
    file = 'result.txt'
    writer = open(file, 'w')
    doc_id = np.random.randint(model.docvecs.count)
    sims = model.docvecs.most_similar(doc_id, topn=model.docvecs.count)
    writer.write('TARGET {}   <<{}>>\n'.format(doc_id, get_paragraph(documents, doc_id)))
    writer.write('SIMILAR | DISSIMILAR')
    for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
        writer.write('{} {}:  <<{}>>\n'.format(label, sims[index], get_paragraph(documents, sims[index][0])))

def search_by_keywords(model, keys, documents, top_n):
    MAX_COL = 20
    keys = [k.lower() for k in keys]
    keys = [k for k in keys if k in model.wv.vocab]

    if not len(keys):
        print("NOT Found in the vocabulary !!\n try different keywords")
        return "NOT Found in the vocabulary !!"

    most_similar = model[keys]
    doc_ids = [i[0] for i in model.docvecs.most_similar(most_similar, topn=top_n)]
    doc_scores = [i[1] for i in model.docvecs.most_similar(most_similar, topn=top_n)]

    assert len(doc_scores) == len(doc_ids), "must be equal in size."

    writer = open('result.txt', 'w')
    for i, s in zip(doc_ids, doc_scores):
        writer.write('<< document id {} | score {} >>\n'.format(i, s))
        writer.write('--'*30 + '\n')
        print(" ".join(documents[i].words)); print('=='*50)
        for j, token in enumerate(documents[i].words):
            writer.write(token); writer.write(' ')
            if j % MAX_COL == 0 and j != 0: 
                writer.write('\n')
        # writer.write('\n\n ### score: {} ###'.format(s))
        writer.write('\n')
        writer.write('=='*50); writer.write('\n')

    return most_similar



if __name__ == '__main__':
    ## command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cmd', help='command (search | compare)', required=True)
    parser.add_argument('-m', '--method', help='search method (keywords | document file', required=True)
    parser.add_argument('-k', '--keywords', help='keywords to be used as keys in search', nargs='+')
    parser.add_argument('-n', '--topn', help='the top most similar documents', default=5, type=int)
    parser.add_argument('-f', '--fpath', help='file to be searched with')
    args = parser.parse_args(sys.argv[1:])
    parsed_args = dict(args._get_kwargs())

    ## load the pre-trained model
    # model = load_model('models\doc2vec.bin')
    model = load_model('models/doc2vec.bin')


    if parsed_args['cmd'] == 'search':
        ## to test the training result (may take a while to load large documents)
        documents = extract_documents()
        if parsed_args['method'] == 'kw':
            search_by_keywords(model, parsed_args['keywords'], documents, top_n=parsed_args['topn'])
        elif parsed_args['method'] == 'file':
            try:
                with open(parsed_args['fpath'], 'r') as f:
                    keywords = f.read().split()
                    search_by_keywords(model, keywords, documents, top_n=parsed_args['topn'])
            except FileNotFoundError as ex:
                print(ex)
    elif parsed_args['cmd'] == 'compare':
        ## comparison logic is here
        ## load some test documents form files
        # doc1 = open('test\doc1.txt').read()
        # doc2 = open('test\doc2.txt').read()
        doc1 = open('test/doc1.txt').read()
        doc2 = open('test/doc2.txt').read()
        ## use cosine simililarity function
        sim = compute_similarity(model, doc1, doc2)
        print(colored("cosine similarity between doc1, doc2 is: ", 'green'), sim[0])
    elif parsed_args['cmd'] == 'random':
        ## write reported documents to the corpus
        most_similar_from_random(documents, model)
        










   