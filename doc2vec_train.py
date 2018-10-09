from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import PathLineSentences
from sklearn.linear_model import LogisticRegression
from random import shuffle, sample
import os
import time
import argparse
import sys
import logging


def extract_documents(corpus_path):
    sentence_corpus = PathLineSentences(corpus_path)

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentence_corpus)]

    ## shuffle modify in place
    # return shuffle(documents)
    ## shuffle using sample
    return sample(documents, len(documents))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='path of the corpus', required=True)
    parser.add_argument('-e', '--epoch', help='number of epochs to train on', default=15, type=int)
    parser.add_argument('-c', '--count', help='exclude the tokens less than minimum counts', default=5, type=int)
    parser.add_argument('-w', '--wind', 
        help='The maximum distance between the current and predicted word within a sentence', default=5, type=int
    )
    parser.add_argument('-v', '--vect', help='embeddings vector size (300 recommended for large corpus)', 
        default=100, type=int
    )

    args = parser.parse_args(sys.argv[1:])
    parsed_args = dict(args._get_kwargs())

    corpus_path = parsed_args['path']
    vector_size = parsed_args['vect']
    min_count = parsed_args['count']
    window_size = parsed_args['wind']
    epochs = parsed_args['epoch']

    print('loading the training data ..')

    documents = extract_documents(corpus_path)

    print("corpus sentences size: ", len(documents))

    # PV-DBOW
    model = Doc2Vec(
        vector_size=vector_size,
        negative=5, dm=0,
        window=window_size,
        min_count=min_count,
        workers=4,
        alpha=0.025, min_alpha=0.025,
    )

    model.build_vocab(documents)

    t0 = time.time()
    # for ep in range(epochs):
    #     print('training {} of {} .. '.format(ep, epochs))
    #     model.train(documents=documents, total_examples=len(documents), epochs=model.iter)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model.train(epochs=model.epochs, documents=documents, total_examples=len(documents))
    t1 = time.time()

    print("CPU training Time {}".format(t1-t0))

    print('saving the training result ..')

    model.save_word2vec_format(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/doc2vec_imdb.txt'))

    model.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/doc2vec_imdb.bin'))

    # model = Doc2Vec.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/doc2vec_imdb.bin'))