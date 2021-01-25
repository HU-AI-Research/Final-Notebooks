import textstat
import nltk
import re
import json
import pandas as pd

from textstat.textstat import textstat #for some reason it otherwise gives an error (https://stackoverflow.com/questions/45239998/python-cannot-find-modules-in-installed-textstat)
from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk.corpus import wordnet
from collections import Counter

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob, Word, Blobber

"""
################## Sasha Bunink ##################
"""
# Altered to not return a selected DataFrame head as this messes with other features, did this with all 4 features.
# Source: https://github.com/HU-AI-Research/Text-features-Sasha--count-of-stopwords-mentions-hashtags-words-characters-/blob/main/Extracting%20text%20features%20(count%20of%20stopwords%2C%20mentions%2C%20hashtags%2C%20words%2C%20characters).ipynb

def add_vader_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()

    df['compound'] = [analyzer.polarity_scores(x)['compound'] for x in df['clean_text']]
    df['neg'] = [analyzer.polarity_scores(x)['neg'] for x in df['clean_text']]
    df['neu'] = [analyzer.polarity_scores(x)['neu'] for x in df['clean_text']]
    df['pos'] = [analyzer.polarity_scores(x)['pos'] for x in df['clean_text']]
    return df

def textblob(df):
    df['sentiment_textblob'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment[0]) #Returns column sentiment_textblob with the sentiment score for TextBlob
    return df

def show_char_count(df):
    df['char_count'] = df['text'].str.len() ## this also includes spaces
    return df

def stopwords_count(df):
    stop = stopwords.words('english')
    df['stopwords'] = df['text'].apply(lambda x: len([x for x in x.split() if x in stop]))
    return df

def count_hashtags(df):
    df['hashtags'] = df['text'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
    return df

def count_mentions(df):
    df['mentions'] = df['text'].apply(lambda x: len([x for x in x.split() if x.startswith('@')]))
    return df


"""
################## Nayereh Parpanchi ##################
"""
# Altered to make it a function, as it would otherwise just be a package and copy the apply code
# Source: https://github.com/HU-AI-Research/Shared_code/blob/main/features.ipynb

def reading_ease(df):
    df['flesch_grade'] = df['clean_text'].apply(textstat.flesch_reading_ease)
    return df


"""
################## Shania Spierings ##################
"""
# Altered pos_numbers to take a DataFrame instead of calling a df by specific name from my Jupyter file.
# Source: https://github.com/HU-AI-Research/POS-tagging/blob/main/POS_8:%20Count%20the%20POS-tags.ipynb

def library_tags():
    brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
    tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)
    return tag_fd

def token_tag(x):
    tokens = nltk.word_tokenize(x)
    tags = nltk.pos_tag(tokens, tagset="universal") # tags contains of a list per row 
    return tags

def pos_to_possible_tags(counter, keys_to_add):
    for k in keys_to_add:
        counter[k] = counter.get(k, 0)
    return counter

def pos_numbers(df):
    tokens = df.loc[:, "clean_text"].apply(token_tag)
    pos_counts = tokens.apply(lambda x: Counter(tag for word,  tag in x))
    pos_counts = pos_counts.apply(lambda x: pos_to_possible_tags(x, library_tags()))
    df["pos_numbers"] = pos_counts.apply(lambda x: [count for tag, count in sorted(x.most_common())])
    
    pos_df = pd.DataFrame(df['pos_numbers'].tolist())
    pos_df.columns = sorted(library_tags())
    return pos_df

def get_POS_tags(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV,}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_text(text):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w, get_POS_tags(w)) for w in nltk.word_tokenize(text)]

def add_ref_count(input_df): 
    with open("reference_verbs.json") as json_file:
        reference_list = json.load(json_file)

    text = input_df["clean_text"]
    lem_text = text.apply(lemmatize_text)
    
    
    ret_list = [] 
    
    for line in lem_text:
        this_count = 0
   
        for ref_word in line: 

            if ref_word in reference_list :
                this_count += 1  

        ret_list.append(this_count)    
  
    return ret_list

def wash_pandas_str(text):
    ret_text = text.replace(r'…', '')
    ret_text = ret_text.replace(u'\u2019', '')

    ret_text = ret_text.replace(r'https\S*?\s', '')  
    ret_text = ret_text.replace(r'https\S*?$', '')
    ret_text = ret_text.replace(r'RT\s', '')
    ret_text = ret_text.replace(r'\s$', '')

    ret_text = ret_text.replace(r'@\S*?\s', '')
    ret_text = ret_text.replace(r'@\S*?$', '')

    return ret_text

"""
################## Ali Afshar ##################
"""
# Source: https://github.com/HU-AI-Research/Ali_features/blob/main/detect-fake-tweet-with-logistic-regression%20Version_4.ipynb

def count_capital_letters(tweet):
    return sum(1 for c in tweet if c.isupper())