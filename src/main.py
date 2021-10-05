from language_detection import language_detection
from tweet_processing import process_tweets
import warnings

warnings.filterwarnings('ignore')
nltk.download('stopwords')
print()

language_list = ['en', 'es', 'pt']

# define what each iso-639 code means
language_dict = {
    'en': 'english',
    'es': 'spanish',
    'pt': 'portuguese'
}

spacy_models = {
    'en': 'en_core_web_sm',
    'es': 'es_core_news_sm',
    'pt': 'pt_core_news_sm'
}

# define directories
dataset = './files/all_annotated.tsv'
clean_data_dir = './files/clean_data'
csv_dir = './files/processed_csv'

language_detection(dataset = dataset, clean_data_dir = clean_data_dir, language_list = language_list)

for language_code in language_list:
    print("Processing tweets for " + language_dict[language_code])
    process_tweets(clean_data_dir = clean_data_dir, language_code = language_code, language = language_dict[language_code],spacy_model = spacy_models[language_code], csv_dir = csv_dir)

print('Finish processing all tweets.')