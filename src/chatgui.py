import json
import nltk
import numpy as np
import pickle
import random
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

from src.models import Message

lemmatizer = WordNetLemmatizer()

model = load_model('./util/chatbot_model.h5')
intents = json.loads(open('./util/intents.json').read())
words = pickle.load(open('./util/words.pkl', 'rb'))
classes = pickle.load(open('./util/classes.pkl', 'rb'))


# User input processing
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# bag of words creation
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def prediction_calculation(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    error_threshold = 0.25  # ERROR_THRESHOLD
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    result = list_of_intents[3]['responses']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            result = random.choice(intent['responses'])
            break
    response = {"message": result, "id": "RECEIVE_ID"}
    return Message(response)


def start(msg):
    ints = prediction_calculation(msg, model)
    return get_response(ints, intents)