import csv
import pandas as pd
import warnings
import string
import re
import os
import fasttext

warnings.filterwarnings('ignore')
PRETRAINED_MODEL_PATH = './lid.176.ftz'

def language_detection(dataset, clean_data_dir, language_list):
    df = pd.read_csv(dataset, sep='\t', header=0)
    df = df[['Tweet']]

    model = fasttext.load_model(PRETRAINED_MODEL_PATH)

    df['clean_tweets'] = df['Tweet'].apply(lambda x: remove_non_words(x))
    df_clean_tweets = df[['Tweet', 'clean_tweets']]

    df_clean_tweets['prediction'] = df_clean_tweets['clean_tweets'].apply(lambda x : get_language(model, x))
    df_clean_tweets['language'] = df_clean_tweets['prediction'].apply(lambda x : x[0])
    df_clean_tweets['confidence'] = df_clean_tweets['prediction'].apply(lambda x : x[1])

    df_confident = df_clean_tweets[df_clean_tweets['confidence'] > 0.5]

    if not os.path.exists(clean_data_dir):
        os.mkdir(clean_data_dir)

    for language in language_list:
        df_lang = df_confident[df_confident['language'] == language]
        fulldir = os.path.join(clean_data_dir, language + '.csv')   
        df_lang.to_csv(fulldir)

def remove_non_words(tweet):
    # finally, this is to remove all the url links
    tweet = re.sub(r'^http?:\/\/.*[\r\n]*', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'^https?:\/\/.*[\r\n]*', '', tweet, flags=re.MULTILINE)

    
    # this is to remove all the punctuations in the string
    tweet  = "".join([char for char in tweet if char not in string.punctuation])
    tweet = re.sub('[0-9]+', '', tweet)
    
    # this is to remove all the emojis in the string
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    tweet = re.sub(emoji_pattern, '', tweet)
    
    return tweet

def get_language(model, tweet):
    y = model.predict(tweet)
    return (y[0][0].replace('__label__', ''), y[1][0])
