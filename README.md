# dathena-assignment
5-day assignment for Dathena Interview

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
   1. However, I would need to do an EDA of the dataset from Kaggle to determine how a normal Twitter tweet looks like before I can decide on which ones to use.
