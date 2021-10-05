import csv 
import pandas as pd
import nltk
import os
import spacy
import logging
import warnings

from operator import itemgetter 

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

def process_tweets(clean_data_dir, language_code, language, spacy_model, csv_dir):
    fulldir = os.path.join(clean_data_dir, language_code + '.csv')   
    df = pd.read_csv(fulldir)
    df_clean_tweets = df[['Tweet', 'clean_tweets']]

    # lowercase all tweets
    df_clean_tweets['lowercase'] = df_clean_tweets['clean_tweets'].apply(lambda x: x.lower())

    stop_words = stopwords.words(language)
    # not adding custom stopwords since in this pipeline we are not doing any visualization
    df_clean_tweets['stopwords_removed'] = df_clean_tweets['lowercase'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop_words]))
    
    nlp = spacy.load(spacy_model, disable=['parser', 'ner'])
    df_clean_tweets['lemmatized'] = df_clean_tweets['stopwords_removed'].apply(lambda x: " ".join([token.lemma_ for token in nlp(x)]))


    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")
    print()

    # Vectorizing for sklearn LDA
    n_features = 1000
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=5,  max_features=n_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(df_clean_tweets['lemmatized'])
    tf_feature_names = tf_vectorizer.get_feature_names()

    # LDA model
    print("Executing Grid Search for LDA...")
    print()

    grid_params = {'n_components' : list(range(4,10))}

    # Grid Search to tune number of topics
    sk_lda = LatentDirichletAllocation()
    sk_lda_model = GridSearchCV(sk_lda,param_grid=grid_params)
    sk_lda_model.fit(tf)

    # Estimators for LDA model
    sk_lda_model1 = sk_lda_model.best_estimator_
    print("Best LDA model's params" , sk_lda_model.best_params_)
    print("Best log likelihood Score for the LDA model",sk_lda_model.best_score_)
    print("LDA model Perplexity on train data", sk_lda_model1.perplexity(tf))
    print()

    topic_count = sk_lda_model.best_estimator_.n_components

    # Process data for Gensim LDA
    data_words = list(sent_to_words(df_clean_tweets['lemmatized']))
    # Create Dictionary
    id2word = corpora.Dictionary(data_words)
    # Create Corpus
    texts = data_words
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    logging.basicConfig(filename='lda_model.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Train Gensim LDA
    print("Training Gensim LDA...")
    print()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lda_train = gensim.models.ldamulticore.LdaMulticore(
                            corpus=corpus,
                            num_topics=topic_count,
                            id2word=id2word,
                            chunksize=100,
                            workers=7, # Num. Processing Cores - 1
                            passes=50,
                            eval_every = 1,
                            per_word_topics=True)
        lda_train.save('lda_train.model')

    # Get topics for each tweet
    df_clean_tweets['topics'] = df_clean_tweets['lemmatized'].apply(lambda x : get_topic(lda_train, id2word, x))

    print("Extract Tweets by Topic...")
    print()
    df_export = df_clean_tweets[['Tweet', 'topics']]

    csv_name = 'topic_'
    lang_dir = os.path.join(csv_dir, language_code)   

    if not os.path.exists(lang_dir):
        os.mkdir(lang_dir)

    # Save each dataframe by topic
    for i in range(topic_count):
        save_dir = os.path.join(lang_dir, csv_name + str(i) + '.csv')   
        df_export[df_export['topics'] == i].to_csv(save_dir)


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def get_topic(lda_train, id2word, tweet):
    prob = lda_train[id2word.doc2bow(tweet.split())][0]
    return max(prob,key=itemgetter(1))[0]


