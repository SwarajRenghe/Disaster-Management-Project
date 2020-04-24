import pickle
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

filename = './disaster_tweet_assets/finalized_model.sav'
model = pickle.load(open(filename, 'rb'))

BASE = "./"
train = pd.read_csv(BASE + "disaster_tweet_assets/train.csv")
test = pd.read_csv(BASE + "disaster_tweet_assets/test.csv")
sub = pd.read_csv(BASE + "disaster_tweet_assets/sample_submission.csv")

tweets = train[['text', 'target']]
tweets_list = list(tweets['text'])

def remove_punctuation(text):
    import string
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

sw = stopwords.words('english')
def stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    return " ".join(text)

def stemming(text):    
    '''a function which stems each word in the given text'''
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text) 


vectorizer = CountVectorizer(analyzer='word', binary=True)
vectorizer.fit(tweets_list)

def get_pred (tweet):
	print ("tweet is", tweet)
	sent_tweet = str(tweet).split(" ")
	print (sent_tweet)
	tweet = map (stemming, map (stopwords, map(remove_punctuation, [tweet])))
	tweet = vectorizer.transform(tweet).todense()
	words = open ("disaster_tweet_assets/words.txt").read().split(" ")
	val = [x in words for x in sent_tweet]
	if any(val):
		return [True]
	else:
		return model.predict (tweet)