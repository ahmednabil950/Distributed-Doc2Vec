# Distributed-Doc2Vec
Doc2Vec Project

# Preprocessing

## Preprocessing toolbox
clone the repository using
```
git clone https://github.com/ahmednabil950/Distributed-Doc2Vec
```
```
pip install -r requirements.txt
```
## Guides | Instructions
```
1- construct each paragraph to be one line per each:
2- convert all words to lower case
3- remove punctuation
4- remove stop words
5- convert nummeric to words example: (1 should be one, 19 should be nine-teen).
```
* [stop words and noise removal useful ressources](https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/)
* [different-resources](https://github.com/igorbrigadir/stopwords)



## Resources
* [Data Preprocessing | NLP](https://towardsdatascience.com/pre-processing-in-natural-language-machine-learning-898a84b8bd47)
* [Data Cleaning](https://towardsdatascience.com/basic-data-cleaning-engineering-session-twitter-sentiment-data-95e5bd2869ec)

# Training Doc2Vec Models
## Resources
* [Introduction to doc2vec](https://medium.com/scaleabout/a-gentle-introduction-to-doc2vec-db3e8c0cce5e)
* [doc2vec | gensim](https://medium.com/@gofortargets/doc2vec-word2vec-in-gensim-c9321c780079)
* [gensim tutorial official](https://radimrehurek.com/gensim/models/doc2vec.html)
* [sentence2vec](https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-6-doc2vec-603f11832504)
* [rare technology blog](https://rare-technologies.com/doc2vec-tutorial/)

# Common Issues:
Training on cpu for genesim faster than GPU for genesim:
* [word2vec cpu faster](https://rare-technologies.com/gensim-word2vec-on-cpu-faster-than-word2veckeras-on-gpu-incubator-student-blog/)

# Solution
* [BlazingText Algorithm | AWS](https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-blazingtext-parallelizing-word2vec-on-multiple-cpus-or-gpus/)

# Concurrency on gensim
* [deep learning with word2vec](https://rare-technologies.com/deep-learning-with-word2vec-and-gensim/)
* [optimization word2vec](https://rare-technologies.com/word2vec-in-python-part-two-optimizing/)
* [word2vec parallelization](https://rare-technologies.com/parallelizing-word2vec-in-python/)
