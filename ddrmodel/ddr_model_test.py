#NOT YET COMPLETE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
import numpy as np
import random
import sys
import io
model = load_model("ddr_model.h5")

LSTM_HISTORY_LENGTH = 32

if len(sys.argv) > 1:
  path = sys.argv[1]
else:
  path = "../preprocessing/ddr_data/In The Groove/Anubis/c6_5820.mnd"

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

diversity = 0.5

with open(path, mode='r', encoding="utf-8") as generic:
    songpath = generic.readline()
    bpm = generic.readline()
    offset = generic.readline()
    last_time = 0
    history = [0, 0, 0, 0]*LSTM_HISTORY_LENGTH
    while(line = generic.readline()):
        line_data = line.split()
        (time_point, note, start_long, end_long) = (int(x) for x in line_data)
        for t in [48, 24, 16, 12, 8, 6, 4, 3, 2, 1]: #time_resolution of individual notes
            if time_point%t == 0:
                beat_fract = t
                break
        mnd_data = [time_point-last_time, beat_fract, note, start_long, end_long]
        last_time = time_point
        model.predict([mnd_data,history[-1:-LSTM_HISTORY_LENGTH]], verbose=0)[0]

for diversity in [0.5]:
    print('----- diversity:', diversity)
    sys.stdout.write(generated)

    for i in range(400):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        sentence = sentence[1:] + next_char

        print(next_char, end = "")
    print()