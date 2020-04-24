import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle 

BASE = "./"
train = pd.read_csv(BASE + "train.csv")
test = pd.read_csv(BASE + "test.csv")
sub = pd.read_csv(BASE + "sample_submission.csv")

tweets = train[['text', 'target']]

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import classification_report

def remove_punctuation(text):
    import string
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

tweets_target = list (tweets['target'])
tweets_list = list(tweets['text'])

# tweets_list = tweets_list.apply(remove_punctuation)
tweets = [remove_punctuation(x) for x in tweets_list]

sw = stopwords.words('english')

def stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    return " ".join(text)

# tweets_list = tweets_list.apply(stopwords)
tweets = [stopwords(x) for x in tweets_list]

# create an object of stemming function
stemmer = SnowballStemmer("english")

def stemming(text):    
    '''a function which stems each word in the given text'''
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text) 

# tweets_list = tweets_list.apply(stemming)
tweets = [stemming(x) for x in tweets_list]
# tweets.head(10)

vectorizer = CountVectorizer(analyzer='word', binary=True)
vectorizer.fit(tweets_list)

X = vectorizer.transform(tweets_list).todense()
# y = tweets['target'].values
y = tweets_target

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020)

model = LogisticRegression()
model.fit(X_train, y_train)

print (X_test)

y_pred = model.predict(X_test)

f1score = f1_score(y_test, y_pred)
print(classification_report(y_test, y_pred))

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

print ("Model saved to disk")
