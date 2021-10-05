# Multilingual Topic Modelling 
Feel free to raise any suggestions for improvements. 

Note that this is just a simple assignment done in one night. There are a lot of mistakes or could haves that are not being implemented due to the time constraint of the author. 

## Usage
1. You can unzip the file or you can git clone it from [here](https://github.com/bingyuyap/dathena-assignment)

## Documentation

### Process of searching for data set / source

1. Look for data sources to scrape / get data from
   1.  Some data sources are suggested like Twitter.
       1.  Look at APIs provided by Twitter to determine whether this is feasible
       2. API documentation refered to is [Develop Platform from Twitter](https://developer.twitter.com/en/docs/twitter-api/v1/tweets/search/api-reference/get-search-tweets)
       3. Also looked at [Tweepy](https://docs.tweepy.org/en/latest/), an alternative API for Twitter
       4. However, twitter API requires Access Token and that requires waiting the Twitter to approve my developer account. So this was not an option.
   2.  Another source was Wikipedia
       1.  Looking into Wikipedia, I observed that mostly they have a subject but not a broad topic. Compared to Twitter, which usually has a hot topic, this pales in comparison to be the dataset.
   3.  Last source I looked at was kaggle. One particular dataset that I looked at was [this](https://www.kaggle.com/rtatman/analyzing-multilingual-data/data). As it has multilingual data and also is scraped from twitter.

2. Next, I was looking for best library to process text and label the languages. Refered to [stackOverflow](https://stackoverflow.com/questions/39142778/python-how-to-determine-the-language) links like this.
   - However, I would need to do an EDA of the dataset from Kaggle to determine how a normal Twitter tweet looks like before I can decide on which ones to use.
   - After some EDA and research, I dediced to use [fastText](fasttext.cc) by Facebook as it provides a more efficient language detection with the compact model as I am prioritizing speed over accuracy for this assignment and as it also gives us a confidence level of its prediction.


### Data preprocessing
This can be seen more clearly under `notebooks > langauge_detection.ipynb`

1. EDA
    - Mostly done using `df.sample()` to get a sense of how a tweet looks like

    Result: Tweets in the dataset has quite a few link, punctuations, and emojis. Hence some cleaning is required to be done before language detection.

2. Data cleaning
    - In order for the [fastText](fasttext.cc) library to be able to detect languages more accurately, I firstly removed the links, punctuation and emojis.

3. Language detection
    - Language for each Tweet is predicted by fastText with a confidence level. This is separated into two columns called `df['language']` and `df['confidence']`
    - Only predictions of `>50%` is taken.
        - Considerations: from the Kaggle dataset 95 languages are tagged so presumably `50%` is a safe number to use.
        - We have a large dataset (`>10,000` Tweets) so we could assume a larger confidence level and drop those below this confidence level.
    
4. Choosing the languages
    - A simple bar chart and pie chart is used to visualize the languages with most Tweets.
    Result: English, Portuguese and Spanish

5. Exporting Tweets by languages
    - The Tweets are then separated by languages predicted and exported to their own separated csv files.

### Tweet processing    
1. Extracting the data 
    - Extracted the original Tweet and cleaned Tweet from the respective files that was previously separated.
        - Original Tweet is needed for use to export and persist in clustered csv files.
        - Cleaned Tweet is the data used in further cleaning and processing

2. Further cleaning
    - Firstly, lowercase all the tweets to reduce the dimension as words with different casing introduces more tokens. EG: `Tweet` and `tweet` is considered as two different tokens.
    - Wordcloud is used to visualize the big picture of the tokens. It can be seen that there are a lot of Internet lingos that is used and unmeaningful words.
        - Note that this is not useful in processing large amount of languages as we need to visualize the word clouds one by one. This is tradeoff between accuracy and scalability of the model. In this assignment, we focus in scalability.
    - Remove stopwords using [`nltk`](nltk.org) library 
        - This library has quite an extension stopword library in the sense that it supports a range of languages.
        - But still as we scale to support more languages we might face the problem where `nltk` library might not support it. But `nltk` is one of the better ones available. 
        - In order to scale to other languages, we should consider language specific data cleaning pipeline instead of using the general pipeline.
    - Lemmatizer is used instead of Stemmer. Rationale is that  Stemmer could lead to mispelled words and this will cause duplicated tokens for a supposedly same word.
        - I don't this is scalable as we increase the scope of the languages as there might not be Lemmatizer / good Lemmatizers for specific languages. That introduces the need for language specific lemmatization.
        - More considerations:
            1. Speed - while Lemmatizer is usually slower than Stemmer, it does a better job in getting the actual word / meaning of the tokens. So here is a tradeoff introduced.
            2. Preprocess all the data source fairly - I want to create a pipeline to preprocess all languages the same way. For this assignment purpose, the SpaCy Lemmatizer used supports all three languages just fine. However, when it comes to scaling to other languages, we need to consider preprocessing each language differently.
3. Processing 
    - `CountVectorizer` is used to get `term frequency` features

4. Model training
    - Models considered:
        - Latent Semantic Analysis (LSA)
            - LSA is not used as it has a less efficient representation and less interpretable embeddings
            - Since we are clustering Tweets, and usually this requires the analyst to understand what generally people are tweeting - to find more popular topics and etc - LSA is not a very good algorithm.
        - Probabilistic LSA
            - This is not used as number of parameters for pLSA grows linearly with the number of documents we have, so it is prone to overfitting.
            - Analysing tweets can go up to millions or even more per day based on the usage of Twitter. If we want to have a good clustering model for analysis Tweets, PLSA should not be used.
        - Latent Dirichlet Allocation (LDA)
            - We can extract human-interpretable topics from a document corpus, where each topic is characterized by the words they are most strongly associated with.
            - Chosen as this fits more of what analysing tweet clusters require.
    - Hyperparameter tuning 
        - a simple `GridSearchCV` is done to find the optimal number of topics for the tweets.
    - `sklearn LDA` was used at first. However, `sklearn LDA` does not provide the functionality to predict the documents hence I had to look for another LDA model that provides this functionality.
    - `Gensim LDA` was then used. The number of topics used will be the same as the one we got from `sklearn GridSearchCV` as the wrapper provided to tune Gensim LDA is depreciated.
    - Visualization is done using `pyLDAvis` to check the clusters obtained by the 2 `LDA` models obtained. [Notebook for reference](https://github.com/bingyuyap/dathena-assignment/blob/main/notebooks/tweets-processing.ipynb)
        - The `GridSearchCV` seems to have done a great job in find the optimal number of topics and LDA has produced 5 well separated topics from the looks of Intertopic Distance Map, since none of them are overlapping.

            Most of the topics are human-interpretable too. For example:

            - Topic 1: This topic seems like it's talking about dating, meeting people in general.
            - Topic 2: It's mostly about a job at company, hiring of the company and career related words.
            - Topic 3 is uninterpretable, but it also shows that the data cleaning is not done as well as it could be.
            - Topic 4: Seems like it is about a great day some one had.
        - `Gensim's` distribution looks pretty similar to that of sklearn as the topics are almost 1-to-1 the same. We can safely assume that sklearn GridSearchCV can be used to tune the number of topics for Gensim.

            Now we can use `Gensim LDA` to predict which Tweet belongs to which 
            topic.
5. Clustering documents by topic
    - `Gensim LDA` is used to get the probability of the topic for each Tweet. The maximum probability one is chosen.
    - Each tweet is then labelled with the topic number on column `df['topics']`

6. Exporting documents by topic
    - Since we are persisting Tweets, there are no point persisting each tweet as a file under a directory as there are relatively small (`<280 chars`). Assuming we are using a relational database, we can persist the dataframe as a table with Tweet and Topic as the two columns.

## Conclusion

After looking through the files, it seems like the topics are not very consistent, in the sense that there are a lot of tweets that are not related. This could be due to a few reasons:

1. There are still some mismatched languages where tweets in other languages are classified as English tweets. In future implementation, I want to look into other language detection libraries that can yield higher accuracy.
2. There are tweets that are multilingual, in the sense that a tweet could be in both English, and other languages, but still classified as English. The non-English words will introduce quite a fair bit of noise to the dataset. In future implementation, I want to look into libraries that could detect multiple languages in a given text, and possible remove the non-English text from the tweet, or just drop the row completely since the topic features could be in the non-English words.
3. There are usage of Internet lingo like lol, lololol and etc, the meaning of lol and lololol can be interpreted similarly but is not done so in the Lemmatizer. In future implementation, I want to look into identifying all these Internet lingo to improve this model.
4. Lack of features, I did not use n-gram words in this assignment. Using n-gram words could lead to a higher interpretability as some words do not make sense on their own.
5. Nature of the Tweets, some tweets could be retweets and in the nature where the tweets do not mean anything. In the future when obtaining these tweets, I need to filter away the tweets that are retweets, replies and etc. This can be done in the cleaning stage too.

However, due to the time constraint, we can only continue with what we have right here and construct the pipeline.