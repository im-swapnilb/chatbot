#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install nltk')


# In[2]:


import nltk
import numpy as np
import pandas as pd
import flask as Flask
from flask import request
from flask import render_template
import pickle


# In[3]:


nltk.download('punkt')


# In[4]:


from flask import Flask

app = Flask(__name__)


# In[5]:


nltk.download('wordnet')


# In[6]:


# nltk.download()


# In[7]:


from nltk.tokenize import sent_tokenize


# In[8]:


from nltk.tokenize import word_tokenize


# In[9]:


text = "My name is Swapnil and I am Studying AI"


# In[10]:


print(sent_tokenize(text))


# In[11]:


print(word_tokenize(text))


# In[12]:


from nltk.stem import WordNetLemmatizer


# In[13]:


lemitizer = WordNetLemmatizer()


# In[14]:


print("Swapnil:", lemitizer.lemmatize("Swapnil"))


# In[15]:


print("friends:", lemitizer.lemmatize("friends"))


# In[16]:


# a denotes objective in "pos"
print("better:", lemitizer.lemmatize("better", pos = "a"))


# In[17]:


# Identify root word
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


# In[18]:


ps = PorterStemmer()
# Select some words to stem
words = ["enjoys","enjoying","enjoyed","enjoyment","enjoyable"]

for w in words:
  print(w, ":", ps.stem(w))


# In[17]:


# pip install chatterbot


# In[20]:


# pip install chatterbot_corpus


# In[21]:


# pip install --upgrade chatterbot


# In[ ]:


# pip install --upgrade chatterbot_corpus


# In[18]:


from chatterbot import ChatBot


# In[23]:


from chatterbot.trainers import ListTrainer


# In[24]:


swapnil_bot  = ChatBot(name="PyBot", read_only=True, logic_adapters=['chatterbot.logic.MathematicalEvaluation','chatterbot.logic.BestMatch'])


# In[25]:


greetings = ['hi','hey','hello','how are you','I am fine', 'nice tallking to you']


# In[26]:


formula1 = ['pythagorean theorem', 'a sqaud plus b sqaud equals c sqaud']


# In[27]:


formula2 = ['law of cosine', 'c^2=a^2+b^2+2ab']


# In[28]:


train_bot = ListTrainer(swapnil_bot)


# In[29]:


for item in (greetings,formula1,formula2):
  train_bot.train(item)


# In[30]:


print(swapnil_bot.get_response('hey'))


# In[34]:


from chatterbot.trainers import ChatterBotCorpusTrainer


# In[37]:


swapnil_bot_new  = ChatBot(name="PyBot", read_only=True, logic_adapters=['chatterbot.logic.MathematicalEvaluation','chatterbot.logic.BestMatch'])


# In[38]:


trainer = ChatterBotCorpusTrainer(swapnil_bot_new)
trainer.train("chatterbot.corpus.english")


# In[39]:


print(swapnil_bot_new.get_response('Hello'))


# In[46]:


print(swapnil_bot_new.get_response('bye'))


# In[48]:


print(swapnil_bot_new.get_response('rude'))


# In[42]:


print(swapnil_bot_new.get_response('I hate you'))


# In[43]:


print(swapnil_bot_new.get_response('you okay ?'))


# In[44]:


print(swapnil_bot_new.get_response('mad'))


# In[45]:


print(swapnil_bot_new.get_response('I am hungry'))


# In[19]:


from flask import Flask

app = Flask(__name__)


# In[20]:


@app.route('/')
def home():
    return render_template('Chatbot.html')


# In[ ]:


@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
#         to_predict_list = str(to_predict_list)
        to_predict_list = list(map(str, to_predict_list))
#         to_predict_list.capitalize()
        result = swapnil_bot_new.get_response(to_predict_list)        
        return render_template("result.html", prediction = result)


# In[ ]:


# Main function
if __name__ == "__main__":
    app.run(debug=True)
    app.config['TEMPLATES_AUTO_RELOADED'] = True

