import pandas as pd
import numpy as np
import nltk
import re
import string
import matplotlib.pyplot as plt
import matplotlib

import pdb
import seaborn as sns
from nltk.corpus import stopwords
from nltk.corpus import brown


# for single tweet test
df_single_word = pd.read_csv('./df_word.csv')
df_bigrm = pd.read_csv('./df_bigrm.csv')
fake_p_prior = 0.32

word_set = set(df_single_word['word'])
bigrm_set = set(df_bigrm['two_words'])


#use NLTK get the stopwords
stopwordslist = set(stopwords.words('english'))

#strip punctuation from string for future use
punct_regex = re.compile('[%s]' % re.escape('!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'))

#give a num to each tag, make it easier to count for bigram-tagging 
sing_tag_dict = {'ADJ': 1, 'ADP':2, 'ADV': 3, 'AUX': 4, 'CCONJ': 5, 'DET': 6, 'INTJ': 7, 'NOUN': 8, 'NUM': 9, 'PART': 10, 'PRON': 11, 'PROPN': 12, 'PUNCT': 13, 'SCONJ': 14, 'SYM': 15, 'VERB': 16, 'X': 17, '.': 18}
bi_tags_dict = {}

#18 * 18 types of tagged bigrams
for i in sing_tag_dict.keys():
    for j in sing_tag_dict.keys():
        bi_tags_dict[tuple([i,j])] = (sing_tag_dict[i] - 1 ) * 18 + (sing_tag_dict[j])



#1 make a function to clean the tweets text
def wash_pandas_str( input_df ):
    ret_text = input_df['text'].str.replace(r'…', '')
    ret_text = ret_text.str.replace(u'\u2019', '')

    ret_text = ret_text.str.replace(r'https\S*?\s', ' ')  
    ret_text = ret_text.str.replace(r'https\S*?$', '')
    ret_text = ret_text.str.replace(r'RT\s', '')
    ret_text = ret_text.str.replace(r'\s$', '')

    ret_text = ret_text.str.replace(r'@\S*?\s', '')
    ret_text = ret_text.str.replace(r'@\S*?$', '')
    ret_text = ret_text.str.replace('“', '')
    ret_text = ret_text.str.replace('--', '')
    ret_text = ret_text.str.replace('-', ' ')

    input_df['text'] = ret_text
    return input_df
  
 
#2 Data analysis, train the data by using Naive Bayes   
#2.1 To save testing time, I've set a default limit 2000 words, it can also be changed to higher value to get higher accuracy. 
def naive_bayes_train(X_train, Y_train, limit = 2000):
    
    #count the true tweets and high tweets numbers.
    fake_cnt = len(Y_train[Y_train == 1].index)
    true_cnt = len(Y_train[Y_train == 0].index)

    #get the priori probability of fake tweet.
    fake_prob_prior = fake_cnt / (fake_cnt + true_cnt)

    #{word：(cnt_in_true, cnt_in_fake),}, cnt_in_true/fake means the number of word occurrences in true/fake tweet
    ret_dict = {}

    for ind in X_train.index:
        twit_txt = punct_regex.sub('', X_train['text'][ind])

        #use set() convert tweets words to tuple, no repetition
        for i in set(twit_txt.lower().split()):

            if i not in stopwordslist:
                if i not in ret_dict.keys():    # new word found
                    
                    if Y_train[ind] == 0:       # new word found in true tweet
                                                # because I need the cnt for future calculations, 0 & 1 will cause problem, so I add 1 & 2.
                        ret_dict[i] = [2,1]      
                    else:                       # new word found in fake tweet
                        ret_dict[i] = [1,2]      
                                                
                else:                           # old word found
                    if Y_train[ind] == 0:       # old word found in true tweet 
                        ret_dict[i][0] += 1      
                    else:                       # old word found in fake tweet
                        ret_dict[i][1] += 1      


    #[word, cnt_in_true, cnt_in_fake,freq_true, freq_fake,total_cnt]    
    train_df = pd.DataFrame.from_dict(ret_dict, orient = 'index')
    train_df = train_df.reset_index()
    train_df.columns = ['word', 'cnt_in_true', 'cnt_in_fake']

    train_df['freq_true'] = train_df['cnt_in_true'] / true_cnt
    train_df['freq_fake'] = train_df['cnt_in_fake'] / fake_cnt
    train_df['total_cnt'] = train_df['cnt_in_true'] + train_df['cnt_in_fake']

    #sort by the word occurrences number, get 500 words.
    train_df = train_df.sort_values(by = ['total_cnt'],ascending=False).iloc[0:limit,:]

    return train_df, fake_prob_prior

#2.2 generate the high frequency words map
#if use ipython --pylab in Visual Studio Code can get the high quality image for word frequency
def plot_word_map(train_df, word_count = 50, xlimit = 0.3):

    X = train_df['freq_true'].tolist()
    Y = train_df['freq_fake'].tolist()
    s = (train_df['total_cnt']/40).tolist()
    labels = train_df['word'].tolist()

    assist_x = [0, 0.3]
    assist_y = [0, 0.3]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(X, Y, s = s)
    plt.plot(assist_x, assist_y, color = 'r')

    for i in range(len(labels)):
        #pdb.set_trace()
        plt.text(X[i] + 0.01 * (np.random.rand()-0.4), Y[i] + 0.01 * (np.random.rand()-0.4), labels[i])

    ax.set_aspect('equal')
    plt.xlim(0,xlimit)
    plt.ylim(0,xlimit)
    plt.xlabel('Frequency in true tweets')
    plt.ylabel('Frequency in fake tweets')
    matplotlib.rcParams.update({'font.size': 16})

    fig.show()
 


#3 Use the train df from above function, generate the "words feature"  
def naive_bayes_generate_feature(train_df, fake_prob_prior,X_input,Y_input):
 
    #the most frequently occurring words
    words_set = set(train_df['word'])
    
    accurate_count = 0

    ret_list=[]

    j = 0

    for ind in X_input.index:

        twit_txt = punct_regex.sub('', X_input['text'][ind])
        fake_prob = fake_prob_prior        #priori probability of fake tweet
        true_prob = 1 - fake_prob_prior    #priori probability of true tweet

        for i in set(twit_txt.lower().split()):
            if i in words_set:             

                #train_df['word','cnt_in_true', 'cnt_in_fake','freq_true', 'freq_fake','total_cnt']
                #Probability of being a true tweet, and a fake tweet（according to naivebayes） 
                true_prob_temp = true_prob * train_df[train_df['word'] == i].iloc[0,3]
                fake_prob_temp = fake_prob * train_df[train_df['word'] == i].iloc[0,4]

                #Since the probability values become smaller when multiplied together, I changed the format
                true_prob = true_prob_temp / (fake_prob_temp + true_prob_temp)
                fake_prob = fake_prob_temp / (fake_prob_temp + true_prob_temp)

        ret_list.append(fake_prob)

        #if the probability of being a fake tweet larger than true tweet, predict it to be fake.
        pred = int(fake_prob > true_prob)

        #if the prediction is correct, count to accurate
        accurate_count += (Y_input[ind] == pred)  

        j += 1
        #as this function takes quite a few mins to completion, I add this print to show the process
        if j % 1000 == 0:
            print ('{0} processed, {1:3f}'.format( j, accurate_count/j) )

    return ret_list



def conv_array(ret_list):
    x = np.array(ret_list)
    x = x.reshape(-1,1)
    return x

  
#4 Bigram frequency feature:  
#4.1 To save testing time, I've set a default limit 2000 two_words, it can also be changed to higher value to get higher accuracy. 
def naive_bayes_bigrm_train(X_train, Y_train, limit = 2000):
    
    #count the true tweets and high tweets numbers.
    fake_cnt = len(Y_train[Y_train == 1].index)
    true_cnt = len(Y_train[Y_train == 0].index)

    #get the priori probability of fake tweet.
    fake_prob_prior = fake_cnt / (fake_cnt + true_cnt)

    #{two_word：(cnt_in_true, cnt_in_fake),}, cnt_in_true/fake means the number of two_word occurrences in true/fake tweet
    ret_dict = {}

    for ind in X_train.index:
        tweet = punct_regex.sub('', X_train['text'][ind])

        tweet = tweet.lower()  
        tokens = nltk.word_tokenize(tweet)
        bigrm = list(nltk.bigrams(tokens))

        #use set() convert tweets words to tuple in two_words, no repetition

        for i in bigrm:
            if i not in ret_dict.keys():    # new two_words found
                if Y_train[ind] == 0:       # new two_words found in true tweet
                                            
                    ret_dict[i] = [2,1]      
                else:                       # new two_words found in fake tweet
                    ret_dict[i] = [1,2]      
                                                
            else:                           # old two_words found
                if Y_train[ind] == 0:       # old two_words found in true tweet 
                    ret_dict[i][0] += 1      
                else:                       # old two_words found in fake tweet
                    ret_dict[i][1] += 1      


    #[two_words, cnt_in_true, cnt_in_fake,freq_true, freq_fake,total_cnt]    
    train_df_bigrm = pd.DataFrame.from_dict(ret_dict, orient = 'index')
    train_df_bigrm = train_df_bigrm.reset_index()
    train_df_bigrm.columns = ['two_words', 'cnt_in_true', 'cnt_in_fake']

    train_df_bigrm['freq_true'] = train_df_bigrm['cnt_in_true'] / true_cnt
    train_df_bigrm['freq_fake'] = train_df_bigrm['cnt_in_fake'] / fake_cnt
    train_df_bigrm['total_cnt'] = train_df_bigrm['cnt_in_true'] + train_df_bigrm['cnt_in_fake']

    #sort by the bigram occurrences number, get 2000 bigrams.
    train_df_bigrm = train_df_bigrm.sort_values(by = ['total_cnt'],ascending=False).iloc[0:limit,:]

    return train_df_bigrm, fake_prob_prior


#4.2 Use the train df from above function, generate the "bigram feature" of train data
def naive_bayes_generate_feature_bigrm(train_df_bigrm, fake_prob_prior,X_input,Y_input):
 
    #the most frequently occurring two_words
    words_set = set(train_df_bigrm['two_words'])
    
    accurate_count = 0

    #the "bigram feature"--the probability of fake tweet, will be save to this list
    ret_list=[]

    j = 0

    for ind in X_input.index:

        tweet = punct_regex.sub('', X_input['text'][ind])
        tweet = tweet.lower()  
        tokens = nltk.word_tokenize(tweet)
        bigrm = list(nltk.bigrams(tokens))

        fake_prob = fake_prob_prior        #priori probability of fake tweet
        true_prob = 1 - fake_prob_prior    #priori probability of true tweet
   

        for i in bigrm:
            if i in words_set:             

                #train_df['word','cnt_in_true', 'cnt_in_fake','freq_true', 'freq_fake','total_cnt']
                #Probability of being a true tweet, and a fake tweet
                true_prob_temp = true_prob * train_df_bigrm[train_df_bigrm['two_words'] == i].iloc[0,3]
                fake_prob_temp = fake_prob * train_df_bigrm[train_df_bigrm['two_words'] == i].iloc[0,4]

                #Since the probability values become smaller when multiplied together, I changed the format
                true_prob = true_prob_temp / (fake_prob_temp + true_prob_temp)
                fake_prob = fake_prob_temp / (fake_prob_temp + true_prob_temp)

        ret_list.append(fake_prob)

        #if the probability of being a fake tweet larger than true tweet, predict it to be fake.
        pred = int(fake_prob > true_prob)

        #if the prediction is correct, count to accurate
        accurate_count += (Y_input[ind] == pred)  

        j += 1
        #as this function takes quite a few mins to completion, I add this print to show the process
        if j % 1000 == 0:
            print ('{0} processed {1:3f}'.format( j, accurate_count/j) )

    return ret_list
  

#5 Bigram with Tagging feature:

def tags_bigram_generate_features(X_train):

    #from bigrams of tags for SVM.
    #got the num of total bigram_tagged number : 18 * 18 = 324.
    ret_array = np.zeros((len(X_train), 324))

    cnt = 0

    for ind in X_train.index:
        tweet = punct_regex.sub('', X_train['text'][ind])

        tweet = tweet.lower()  
        tokens = nltk.word_tokenize(tweet)
        bigrm = list(nltk.bigrams(tokens))

        for i in bigrm:
            #convert the bigram to tags
            j = nltk.pos_tag([i[0], i[1]], tagset='universal')
            tags_bigrm = tuple([j[0][1], j[1][1]])

            if tags_bigrm in bi_tags_dict.keys(): #insurance only
                #pdb.set_trace()
                ret_array[cnt, bi_tags_dict[tags_bigrm] - 1] += 1

        cnt += 1
        if cnt%1000 == 0:
            print(str(cnt) + " processed")

    return ret_array        
  







#6 functions for single tweet test:
#function to clean the single tweet text.
def clean_str( input_string ):
    text = input_string.replace(r'…', '')
    text = text.replace(u'\u2019', '')

    text = text.replace(r'https\S*?\s', ' ')  
    text = text.replace(r'https\S*?$', '')
    text = text.replace(r'RT\s', '')
    text = text.replace(r'\s$', '')

    text = text.replace(r'@\S*?\s', '')
    text = text.replace(r'@\S*?$', '')
    text = text.replace('“', '')
    text = text.replace('--', '')
    text = text.replace('-', ' ')
    text = punct_regex.sub('', text)
    text = text.lower()

    input_string = text
    return input_string

def generate_feature_1( input_string ):
 
    fake_prob = fake_p_prior        #priori probability of fake tweet
    true_prob = 1 - fake_p_prior    #priori probability of true tweet

    for i in set(input_string.lower().split()):
        if i in word_set:             
             
            true_prob_temp = true_prob * df_single_word[df_single_word['word'] == i].iloc[0,3]
            fake_prob_temp = fake_prob * df_single_word[df_single_word['word'] == i].iloc[0,4]

            #Since the probability values become smaller when multiplied together, I changed the format
            true_prob = true_prob_temp / (fake_prob_temp + true_prob_temp)
            fake_prob = fake_prob_temp / (fake_prob_temp + true_prob_temp)

    return fake_prob

def generate_feature_2( text ):

    tokens = nltk.word_tokenize(text)
    bigrm = list(nltk.bigrams(tokens))

    fake_prob = fake_p_prior        #priori probability of fake tweet
    true_prob = 1 - fake_p_prior    #priori probability of true tweet

    for i in bigrm:
        #pdb.set_trace()
        if str(i) in bigrm_set:             
             
            true_prob_temp = true_prob * df_bigrm[df_bigrm['two_words'] == str(i)].iloc[0,3]
            fake_prob_temp = fake_prob * df_bigrm[df_bigrm['two_words'] == str(i)].iloc[0,4]

            #Since the probability values become smaller when multiplied together, I changed the format
            true_prob = true_prob_temp / (fake_prob_temp + true_prob_temp)
            fake_prob = fake_prob_temp / (fake_prob_temp + true_prob_temp)

    return fake_prob

def generate_feature_3(text):

    ret_array = np.zeros((324))

    #simple feature from bigrams of tags for SVM.

    tokens = nltk.word_tokenize(text)
    bigrm = list(nltk.bigrams(tokens))

    for i in bigrm:
        j = nltk.pos_tag([i[0], i[1]], tagset='universal')
        tags_bigrm = tuple([j[0][1], j[1][1]])

        if tags_bigrm in bi_tags_dict.keys(): #insurance only
                #pdb.set_trace()
            ret_array[bi_tags_dict[tags_bigrm] - 1] += 1


    return ret_array


