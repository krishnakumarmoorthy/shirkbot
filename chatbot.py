from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np
import random
import string
f=open('chatbot.txt','r')
raw=f.read()
raw=raw.lower()#convert to lowercase
nltk.download('punkt')#first time use only)
nltk.download('wordnet')#first time use only
sent_tokens=nltk.sent_tokenize(raw)#convert to list of sentences
word_tokens=nltk.word_tokenize(raw)
lemmer=nltk.stem.WordNetLemmatizer()
sent_tokens[:2]
word_tokens[:5]
lemmer=nltk.stem.WordNetLemmatizer()
#wordnet is a semantically oriented dictionary of english included in nltk
def LemTokens(tokens):
	return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict=dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
	return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


GREETING_INPUTS=("hello","hi","greetings","sup","what's up","hey",)
GREETING_RESPONSES=["hi","hey","*nods*","hi there","hello","i am glad! you are talking to me"]


def greeting(sentence):
	for word in sentence.split():
		if word.lower() in GREETING_INPUTS:
			return random.choice(GREETING_RESPONSES)
def response(user_response):
	robo_response=''
	sent_tokens.append(user_response)
	TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
	tfidf=TfidfVec.fit_transform(sent_tokens)
	vals=cosine_similarity(tfidf[-1],tfidf)
	idx=vals.argsort()[0][-2]
	flat=vals.flatten()
	flat.sort()
	req_tfidf=flat[-2]
	if(req_tfidf==0):
		robo_response=robo_response+"i am sorry!. i dont understand you"
		return robo_response
	else:
		robo_response=robo_response+sent_tokens[idx]
		return robo_response
flag=True
print("ShirkBot: My name is shirkbot. I will answer your queries about chatbots. If you want to exit, type Bye!")
while(flag == True):
	user_response=raw_input()
	user_response=user_response.lower()
	if(user_response!='bye'):
		if(user_response=='thanks' or user_response=='thank you'):
			flag=False
			print("ShirkBot:you are welcome")
		else:
			if(greeting(user_response)!=None):
				print("ShirkBot: "+greeting(user_response))
			else:
				print("ShirkBot:"+response(user_response))
				sent_tokens.remove(user_response)
	else:
		flag=False
		print("ShirkBot: Bye! Take care dude...")
