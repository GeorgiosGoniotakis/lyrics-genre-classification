# -*- coding: utf-8 -*-

"""This experiment has been individually packaged to enable
easier experimentation on online solutions like Kaggle and
Google Collaboratory. The current implements an LSTM architecture."""

import pandas as pd
import numpy as np

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

# Parameters and definitions
RANDOM_SEED = 0
VAL_SET_SIZE = 0.2

np.random.seed(RANDOM_SEED)

"""### File Paths"""

DATA = "../../data/380000_final.csv"
EMB_FILE_PATH = "../../emb/glove.840B.300d.txt"

"""### Helper Methods"""


def load_data():
    """Loads the training and testing sets into the memory.
    """
    return pd.read_csv(DATA, usecols=["genre", "lyrics"])


"""### Data Wrangling"""

df = load_data()
df.dropna(inplace=True)
df.drop(df[(df.genre == "Not Available") | (df.genre == "Other")].index, inplace=True)

"""### Load word embeddings"""


# Load GloVe Word Embeddings
def load_embeddings(file_path):
    """ Loads word embeddings and returns embeddings index
    """
    embeddings_index = {}
    f = open(file_path)
    for line in tqdm(f):
        values = line.split(" ")
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


emb_index = load_embeddings(EMB_FILE_PATH)
print('Found %s word vectors.' % len(emb_index))

"""### Data Preparation"""

# Convert targets to OHE
X = df["lyrics"]
y = pd.get_dummies(df['genre'], prefix='genre_')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

# Number of unique words in our dataset
NUM_UNIQUE_WORDS = 3000
# Maximum number of words in a question
MAX_WORDS = 125

# Convert questions into vectors of integers using Keras Tokenizer
tokenizer = Tokenizer(num_words=NUM_UNIQUE_WORDS)
tokenizer.fit_on_texts(list(X_train.values))

X_train = tokenizer.texts_to_sequences(X_train)
# X_val = tokenizer.texts_to_sequences(val_questions)
X_test = tokenizer.texts_to_sequences(X_test)

# Pad sequences so that they are all the same length. Questions shorter than maxlen are padded with zeros.
X_train = sequence.pad_sequences(X_train, maxlen=MAX_WORDS)
# X_val = sequence.pad_sequences(X_val, maxlen=MAX_WORDS)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_WORDS)

# Create word index
word_index = tokenizer.word_index

# Dimension of embedding matrix
EMB_DIM = 300

"""### LSTM"""

embedding_matrix = np.zeros((len(word_index) + 1, EMB_DIM))
for word, i in word_index.items():
    embedding_vector = emb_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMB_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_WORDS,
                            trainable=False)

lstm_out = 20  # dimensionality of output space

model = Sequential()
model.add(embedding_layer)
model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

# Fit model to training data
model.fit(X_train,
          y_train,
          validation_split=0.2,
          epochs=20,
          batch_size=1024,
          verbose=1,
          callbacks=[EarlyStopping(monitor='val_acc',
                                   mode="max",
                                   verbose=1,
                                   patience=5,
                                   min_delta=.5)]
          )

# Make predictions for training and test sets
y_pred_train = model.predict(X_train, verbose=1)
y_pred_test = model.predict(X_test, verbose=1)

# Convert probabilities into predictions for training and test set
y_pred_train = (y_pred_train == y_pred_train.max(axis=1, keepdims=True)).astype(int)
y_pred_test = (y_pred_test == y_pred_test.max(axis=1, keepdims=True)).astype(int)

"""### Evaluation"""


def evaluate(y, y_pred):
    print("Accuracy: {}, F1 Score: {}, Precision: {}, Recall: {}".format(accuracy_score(y, y_pred),
                                                                         f1_score(y, y_pred, average="macro"),
                                                                         precision_score(y, y_pred, average="macro"),
                                                                         recall_score(y, y_pred, average="macro")))


evaluate(y_train, y_pred_train)
evaluate(y_test, y_pred_test)
