import os

corpus_dir_pos = os.path.join(os.environ['DATASET'], 'IMDB/aclImdb/train_preprocessed/pos')
corpus_dir_neg = os.path.join(os.environ['DATASET'], 'IMDB/aclImdb/train_preprocessed/neg')
imdb_dir = os.path.join(os.environ['DATASET'], 'IMDB/aclImdb')
original = ['train/pos', 'train/neg', 'test/pos', 'test/neg']
preprocessed = ['train_preprocessed/pos', 'train_preprocessed/neg', 'test_preprocessed/pos', 'test_preprocessed/neg']
preprocessed_map = {k: v for k, v in enumerate(preprocessed)}
models_dir = 'models'