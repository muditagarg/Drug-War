#!/usr/bin/env python
# coding: utf-8

# In[1]:


headerList = ['0','0.1','hashtag', 'id', 'created_at', 'lang' ,'author_id','text','public_metrics','retweet_count','public_metricsreply_count','public_metricslike_count	public_metricsquote_count','geoplace_id','withheldcopyright','withheldcountry_codes','name','place_type','hashtags extracted','CleanedTweet','Sentiment']


# In[2]:


df.to_csv("Sentiment Analysis.csv", header=headerList, index=False)


# In[1]:


columns = ["hashtag","created_at","author_id","hashtags extracted","CleanedTweet","Sentiment"]


# In[2]:


import pandas as pd
df = pd.read_csv('/Users/muditagarg/Sentiment Analysis.csv', usecols = columns)


# In[3]:


df.head(10)


# In[4]:


#df.drop(['0','0.1'],axis=1)


# In[5]:


duplicate_rows_df = df[df.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)


# In[6]:


df = df.drop_duplicates()


# In[7]:


#Grouping them with sentiment for keyword Extraction


# In[8]:


df.groupby(["Sentiment"]).count()


# In[9]:


positive = df[df['Sentiment'] == 1]
negative = df[df['Sentiment'] == 2]


# In[10]:


positive.head(10)


# In[11]:


#Positive Text 


# In[207]:


text = ''
positive['CleanedTweet'] = positive['CleanedTweet'].fillna('')
for i in positive['CleanedTweet']:
  text += str(i)


# In[13]:


#Negative Text


# In[14]:


text = ''
negative['CleanedTweet'] = negative['CleanedTweet'].fillna('')
for i in negative['CleanedTweet']:
  text += str(i)


# In[15]:


import nltk
nltk.download('punkt')


# In[16]:


#Tokenizing the sentences


# In[17]:


from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize


# In[18]:


#tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]


# In[208]:


#to put the text without stopwords
all_stopwords = stopwords.words('english')
sw_list = ['drug','drugs','drugonwar','drugwar','warondrugsthe','good','warondrugs','comes','best','use']
all_stopwords.extend(sw_list)

text_tokens = word_tokenize(text)
tokens_without_sw = [word for word in text_tokens if not word in all_stopwords]


# In[20]:


conda install -c conda-forge wordcloud=1.6.0 


# In[21]:


get_ipython().system('pip install wordcloud')
import numpy as np
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt


# In[209]:


filtered_sentence = (" ").join(tokens_without_sw)
print(filtered_sentence)


# In[198]:


wordcloud = WordCloud(max_font_size=70, max_words=100, background_color="white").generate(filtered_sentence)
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[24]:


#YAKE Keyword Extraction
# Applying Keyword Extraction 
get_ipython().system('pip install git+https://github.com/LIAAD/yake')


# In[210]:



import yake     
# Building Yake model
def fitYAKE (text):
    # Specify parameters
    language = "en"
    max_ngram_size = 3
    deduplication_thresold = 0.9
    deduplication_algo = 'seqm'
    windowSize = 1
    numOfKeywords = 50

    # Yake model with specified parameters
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size,  dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
    keywords = custom_kw_extractor.extract_keywords(text)
    return keywords


# In[211]:


p = fitYAKE(filtered_sentence)


# In[212]:


#p.append(('-', 20.0))


# In[28]:


# score = []
# for item in p:
# #     print(item[0],'----', item[1])
#     score.append(item[1])
# max(score)


# In[29]:


with open('P Sentiment Analysis.txt','w') as f:
 for t in p:
    f.write('----'''.join(str(s) for s in t) + '\n')


# In[213]:



#converting list to dictionary
dictionary={}
for item in p:
    dictionary[item[0]]=item[1]
dictionary

f = open("1dict.txt","w")

# write file
f.write( str(dictionary) )

# close file
f.close()


# In[31]:


# WORD CLOUD


# In[214]:


from os import path, getcwd
d = getcwd()
mask = np.array(Image.open(path.join(d, "/Users/muditagarg/4592.jpg")))
all_terms = []
for list in p:
    all_terms.append(list[0])
    #wordcloud = WordCloud( max_font_size=200,background_color="white",mask= mask,max_words=100,relative_scaling=0.5,normalize_plurals=False).generate_from_frequencies(dictionary)


# In[215]:


mask


# In[216]:


wordcloud = WordCloud( mask= mask, colormap="brg", collocations=False, max_font_size=500, background_color="white" ,relative_scaling=0.5,normalize_plurals=False).generate_from_frequencies(dictionary)


# In[218]:


from matplotlib.pyplot import figure

plt.figure()
plt.figure(figsize=(30,28),dpi=1000)
import random
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")
plt.show()
plt.tight_layout(pad=0)
#plt.imshow(wordcloud.recolor(color_func=grey_color_func, random_state=3),interpolation="bilinear")
wordcloud.to_file("1drugwordcloud.png")


# In[ ]:




