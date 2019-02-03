import pandas as pd
import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import tensorflow as tf
import keras
def import_tweets(filename, header = None):
	#import data from csv file via pandas library
	tweet_dataset = pd.read_csv(filename, header = header)
	#the column names are based on sentiment140 dataset provided on kaggle
	tweet_dataset.columns = ['sentiment','id','date','flag','user','text']
	#delete 3 columns: flags,id,user, as they are not required for analysis
	for i in ['flag','id','user','date']: del tweet_dataset[i] # or tweet_dataset = tweet_dataset.drop(["id","user","date","user"], axis = 1)
	#in sentiment140 dataset, positive = 4, negative = 0; So we change positive to 1
	tweet_dataset.sentiment = tweet_dataset.sentiment.replace(4,1)
	return tweet_dataset

def preprocess_tweet(tweet):
	tweet.lower()
	tweet=re.sub('((www\.[^\s]+)|(https?://[^\s]+))','url',tweet)
	tweet=re.sub('@[^\s]+','AT_USER',tweet)
	tweet=re.sub('[\s]+',' ',tweet)
	tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
	return tweet

def extract_feature(data):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    stopWords=set(stopwords.words('english'))
    dic=[x for x in cleaned_word if x not in stopWords]
    word_set=set(dic)
    word_set1=list(word_set)
    data1=np.zeros((len(data),len(word_set)))
    for i in range(len(data)):
    	words1=data[i].split()
    	for j in range(len(words1)):
         if words1[j] in word_set:
          data1[i][word_set1.index(words1[j])]=1;
    return data1      
              
     
def preprocesed_data():
        x=import_tweets("train.csv")
        x['text']=x['text'].apply(preprocess_tweet)
        col=x['sentiment']
        ''' col1=[col]
                             col1=np.transpose(col1)'''
        data=extract_feature(x['text'])
        '''print len(data)
                                print len(data[0])
                                print len(col)'''
        '''dataset=np.hstack((data,col1))'''
        return data,col

x,y=preprocesed_data()
print (np.shape(x))
print (np.shape(y))
x_train,x_test,y_train,y_test=train_test_split(x,
 y, test_size=0.20, random_state=42)
print(len(x_train))
print(len(x_train[0]))
print(len(x_test))
print(len(x_test[0]))
print(len(y_train))
print(len(y_test))
clf = tree.DecisionTreeClassifier()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
acc=accuracy_score(y_test,y_pred)
print(acc)

'''df = pd.DataFrame(x)
df.to_csv("train_feature.csv", sep='\t', encoding='utf-8')
df1 = pd.DataFrame(y)
df1.to_csv("train_label.csv", sep='\t', encoding='utf-8')'''
