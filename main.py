from language_detection import language_detection

language_list = ['en', 'es', 'pt']

# define what each iso-639 code means
lang_dict = {
    'en': 'english',
    'es': 'spanish',
    'pt': 'portugese'
}

# define directories
dataset = './all_annotated.tsv'
clean_data_dir = './clean_data'
joblib_dir = './joblib'
csv_dir = './processed_csv'
lang = 'en'

language_detection(dataset = dataset, clean_data_dir = clean_data_dir, language_list = language_list)