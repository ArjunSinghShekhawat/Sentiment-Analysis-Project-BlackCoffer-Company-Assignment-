import pandas as pd
import numpy as np
import requests 
from bs4 import BeautifulSoup as bs
import logging
from concurrent.futures import ThreadPoolExecutor
import os
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import cmudict
from concurrent.futures import ThreadPoolExecutor 

## if run the code then delete scrapper.log and extracion data folder file because this file is automatically generate when code is run.


##Approach
#1. imports: The code imports necessary libraries such as Pandas for data manipulation, NumPy for numerical operations, Requests for making HTTP requests, BeautifulSoup for web scraping, and others.
#2. Logging Configuration: Sets up logging to save information about the program's execution into a log file named 'scrapper.log'.
#3. Reading Data: Reads data from an Excel file located at a specific path.
#4. Fetching Web Content: Defines a function fetch_content(url) to fetch HTML content from URLs, extract title and article content, and save them to text files.
#5. Multithreaded Web Scraping: Uses ThreadPoolExecutor to concurrently fetch content from multiple URLs.
#6. Sentiment Analysis Setup: Sets up paths and initializes variables and sets for stopwords, positive words, and negative words.
#7. Read Stopwords: Reads stopwords from files located in a folder.
#8. Read Positive and Negative Words: Reads positive and negative words from separate files.
#9. Initialize CMU Dictionary: Initializes a dictionary for the Carnegie Mellon University Pronouncing Dictionary.
#10. Calculate Readability Metrics: Defines a function to calculate readability metrics such as average sentence length, percentage of complex words, Fog Index, etc.
#11. Process Text Files: Defines a function to process each text file, calculating sentiment scores, readability metrics, and other statistics.
#12. Multithreaded Text Processing: Processes multiple text files concurrently using ThreadPoolExecutor.
#13. Save Results: Saves the processed data into a CSV file named 'output.csv', including sentiment scores, readability metrics, and the original URLs.
#14. Error Handling: The code includes exception handling to catch and log any errors that occur during execution.

# Configure logging
logging.basicConfig(filename='scrapper.log', filemode='w', level=logging.INFO,
                    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")

try:
    # Read the data from Excel file
    data = pd.read_excel(os.getcwd() + r'\20211030 Test Assignment\Input.xlsx')
    logging.info('Data loaded successfully from Excel')

    # Create folder for storing extracted data
    os.makedirs('Extraction Data Folder', exist_ok=True)

    # Define function to fetch content from URL and save to file
    def fetch_content(url):
        try:
            # Fetch HTML content from URL
            logging.info(f'Fetching data from URL: {url}')
            blackcoffer_page = requests.get(url, headers=headers)
            blackcoffer_html = bs(blackcoffer_page.text, 'html.parser')

            # Extract title and article content
            try:
                title = blackcoffer_html.find_all('h1', {'class': "entry-title"})[0].text.strip()
                article = blackcoffer_html.find_all('div', {'class': 'td-post-content tagdiv-type'})[0].text 
            except IndexError as e:
                title = blackcoffer_html.find_all('h1', {'class':"tdb-title-text"})[0].text.strip()
                article = blackcoffer_html.find_all('div', {'class':'tdb-block-inner td-fix-index'})[14].text

            # Create folder for URL ID if not exists
            url_id = data.loc[data['URL'] == url, 'URL_ID'].values[0]
            folder_path = 'Extraction Data Folder'
            os.makedirs(folder_path, exist_ok=True)

            # Save title and article content to text file
            file_path = os.path.join(folder_path, f'{url_id}_data.txt')
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(f'Title: {title}\n\n')
                file.write(f'Article:\n{article}')

            logging.info(f'Data saved to {file_path}')
            return url, file_path  # Return URL and file path
        except Exception as e:
            logging.error(f'Error fetching or saving data from URL {url}: {e}')
            return None, None

    # Define headers for HTTP request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'
    }

    # Fetch content from all URLs using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        url_file_paths = list(executor.map(fetch_content, data['URL']))

    # Filter out None values (in case of errors)
    url_file_paths = [pair for pair in url_file_paths if pair[0] is not None and pair[1] is not None]

    ## Sentiment analysis

    # Set paths
    text_files_directory = os.getcwd() + r"\Extraction Data Folder"
    stopwords_folder = os.getcwd() + r"\20211030 Test Assignment\StopWords"
    master_dictionary_folder = os.getcwd() + r"\20211030 Test Assignment\MasterDictionary"

    # Initialize logging
    logging.info('Initialize the stopwords and positive and negative variables')

    # Initialize sets and dictionaries
    stopwords = set()
    positive_words = set()
    negative_words = set()

    # Read stop words from files
    def read_stopwords_from_file(file_path):
        with open(file_path, "r") as file:
            return set(file.read().split())

    stopwords_files = [os.path.join(stopwords_folder, filename) for filename in os.listdir(stopwords_folder) if os.path.isfile(os.path.join(stopwords_folder, filename))]
    stopwords = set().union(*[read_stopwords_from_file(file) for file in stopwords_files])

    # Read positive and negative words from files
    def read_words_from_file(file_path):
        with open(file_path, "r") as file:
            return set(file.read().split())

    logging.info('Find positive and negative words that are required')

    positive_words_file = os.path.join(master_dictionary_folder, "positive-words.txt")
    negative_words_file = os.path.join(master_dictionary_folder, "negative-words.txt")
    positive_words = read_words_from_file(positive_words_file)
    negative_words = read_words_from_file(negative_words_file)

    # Initialize CMU dictionary
    cmudict_dict = cmudict.dict()

    # Function to count syllables in a word
    def syllable_count(word):
        if word.lower() in cmudict_dict:
            return max([len(list(y for y in x if y[-1].isdigit())) for x in cmudict_dict[word.lower()]])
        else:
            return max(1, len(re.findall(r'[aeiouy]+', word)))

    NUM_THREADS = 50

    logging.info('Calculate readability metrics')

    # Function to calculate readability metrics
    def calculate_readability_metrics(text):
        sentences = sent_tokenize(text)
        num_sentences = len(sentences)

        words = word_tokenize(text)
        cleaned_words = [word.lower() for word in words if word.lower() not in stopwords and word.isalpha()]
        num_words = len(cleaned_words)

        # Calculate syllable count per word
        syllable_counts = [syllable_count(word) for word in cleaned_words]

        avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0

        num_complex_words = sum(1 for count in syllable_counts if count > 2)
        percentage_complex_words = (num_complex_words / num_words) if num_words > 0 else 0

        fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

        personal_pronouns = re.findall(r'\b(I|we|my|ours|us)\b', text, flags=re.IGNORECASE)
        num_personal_pronouns = len(personal_pronouns)

        total_word_length = sum(len(word) for word in cleaned_words)
        avg_word_length = total_word_length / num_words if num_words > 0 else 0

        # Calculate average number of words per sentence
        avg_words_per_sentence = num_words / num_sentences if num_sentences > 0 else 0

        return avg_sentence_length, percentage_complex_words, fog_index, num_personal_pronouns, avg_word_length, num_words, syllable_counts, avg_words_per_sentence

    logging.info('Execute process file function')

    # Function to process a single file
    def process_file(filename):
        file_path = os.path.join(text_files_directory, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read().lower()  # Read the text file and define the text variable
            readability_metrics = calculate_readability_metrics(text)
            positive_score = sum(1 for word in word_tokenize(text) if word in positive_words)
            negative_score = sum(1 for word in word_tokenize(text) if word in negative_words)
            polarity_score = (positive_score - negative_score) / (positive_score + negative_score + 0.000001)
            num_words = readability_metrics[5]  # Accessing num_words from readability_metrics
            subjectivity_score = (positive_score + negative_score) / (num_words + 0.000001)

            # Calculate complex word count in multiple threads
            with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                complex_word_count = sum(executor.map(lambda word: syllable_count(word) > 2, word_tokenize(text)))

            return {
                "File": filename,
                "Positive Score": positive_score,
                "Negative Score": negative_score,
                "Polarity Score": polarity_score,
                "Subjectivity Score": subjectivity_score,
                "Average Sentence Length": readability_metrics[0],
                "Percentage of Complex Words": readability_metrics[1],
                "Fog Index": readability_metrics[2],
                "Number of Personal Pronouns": readability_metrics[3],
                "Average Word Length": readability_metrics[4],
                "Complex Word Count": complex_word_count,
                "Num Words": num_words,
                "Syllable Count Per Word": readability_metrics[6],  # Add syllable count per word to the return
                "Average Words Per Sentence": readability_metrics[7]  # Add average words per sentence to the return
            }

    # Process all text files in the directory
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(executor.map(process_file, [filename for filename in os.listdir(text_files_directory) if filename.endswith('.txt')]))

    save_data = pd.DataFrame(results)
    save_data = pd.concat([save_data, data['URL']], axis=1)

    # Save data to output CSV
    output_csv_path = 'output.csv'
    save_data.to_csv(output_csv_path, index=False)
    logging.info(f'Data saved to {output_csv_path}')
    
except Exception as e:
    logging.error(f'Error: {e}')
