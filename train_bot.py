import nltk

nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

import json
import pickle

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)
print(intents)

# intents: typical conversation groups
# patterns: possible user interactions
for intent in intents['intents']:
    for pattern in intent['patterns']:

        # tokenize every word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # add to the documents array
        documents.append((w, intent['tag']))

        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# preparation for network training
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # bag of words
    bag = []
    # tokens list
    pattern_words = doc[0]
    # lemmatization of tokens
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # if the word matches, insert 1, otherwise 0
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

training = np.array(training)
# creation of train and test sets: X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# model creation
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=300, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")