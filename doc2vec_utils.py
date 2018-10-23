from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import PathLineSentences
from random import shuffle, sample
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import os, sys
import argparse

def extract_documents(corpus_path):
    sentence_corpus = PathLineSentences(corpus_path)

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentence_corpus)]

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
    file = os.path.join(os.path.abspath(__file__), 'result.txt')
    writer = open(file, 'w')
    doc_id = np.random.randint(model.docvecs.count)
    sims = model.docvecs.most_similar(doc_id, topn=model.docvecs.count)
    writer.write('TARGET {}   <<{}>>\n'.format(doc_id, get_paragraph(documents, doc_id)))
    writer.write('SIMILAR | DISSIMILAR')
    for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
        writer.write('{} {}:  <<{}>>\n'.format(label, sims[index], get_paragraph(documents, sims[index][0])))


if __name__ == '__main__':

    ## load the pre-trained model
    model = load_model('models\doc2vec.bin')

    ## to test the training result (may take a while to load large documents)
    documents = extract_documents(sys.argv[1])

    ## write reported documents to the corpus
    most_similar_from_random(documents, model)

    ## load some test documents form files
    doc1 = open('test\doc1.txt').read()
    doc2 = open('test\doc2.txt').read()

    ## use cosine simililarity function
    sim = compute_similarity(model, doc1, doc2)

    print(sim)