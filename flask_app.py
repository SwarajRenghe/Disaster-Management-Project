
from disaster_tweet_indentification import *

from flask import Flask
from flask import render_template
from flask import request

app = Flask(__name__, template_folder = './', static_url_path='/static')
app.debug = True
app.secret_key = 'development key'

@app.route ("/")
def home():
	return render_template ("home.html")

@app.route ("/disaster_tweets")
def disaster_tweets():
	return render_template ("disaster_tweets.html")

@app.route ("/get_pred", methods=["POST"])
def get_prediction():
	print (request.data)
	sent_tweet = str(request.data.decode('utf-8'))
	val = get_pred (sent_tweet)
	print (val)
	if val == [True]:
		return "yes"
	else:
		return "no"

@app.route ("/temp")
def temp():
	return render_template ("temp.html")

app.run ()