""" This notebook used for creating dataset. """

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import torch
import torchtext
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torchtext.vocab import vocab  
from tqdm import tqdm                                       # progressbar
tqdm.pandas() 
from def_preprocessing import preprocessing                 # for def of preprocessing
from collections import Counter                             # for definition of unique words (tokens) in dataframe

df = pd.read_csv(r'C:\Users\Admin\WORK\Project_CV\Model_NLP_sentiment\data\IMDB Dataset.csv')    # insert path to your data

def transform_label(label):
    return 1 if label == 'positive' else 0

df['label'] = df['sentiment'].apply(transform_label)



df_test = df.loc[0:10]                                            # Choose count of review  for test model (example [0:10])
                                                                  # or for work with dataset  put  [ : ]
df_test['clean'] = df_test['review'].apply(preprocessing) 




#get all processed reviews
reviews = df_test.clean.values
# merge into single variable, separated by whitespaces
words = ' '.join(reviews)
# obtain list of words
words = words.split()

# build vocabulary
counter = Counter(words)
vocab = sorted(counter, key=counter.get, reverse=True)
int2word = dict(enumerate(vocab, 1))
int2word[0] = '<PAD>'
word2int = {word: id for id, word in int2word.items()}

# encode words
reviews_enc = [[word2int[word] for word in review.split()] for review in tqdm(reviews)]

# padding sequences

def pad_features(reviews, pad_id, seq_length=128):
    # features = np.zeros((len(reviews), seq_length), dtype=int)
    features = np.full((len(reviews), seq_length), pad_id, dtype=int)      #Return a new array of given shape (len(reviews), seq_length) and type = int, filled with pad_id.

    for i, row in enumerate(reviews):
        # if seq_length < len(row) then review will be trimmed
        features[i, :len(row)] = np.array(row)[:seq_length]

    return features

seq_length = 256
features = pad_features(reviews_enc, pad_id=word2int['<PAD>'], seq_length=seq_length)       # creating array len 256 filled zero up and then it records real values
                                                                                            # if len < 256 there will be zero 

assert len(features) == len(reviews_enc)                                                    # check count of reviews (true or false)
assert len(features[0]) == seq_length                                                       # check count of  words in review [0] (true or false)

# get labels as numpy
labels = df_test.label.to_numpy()
labels

# train test splitting
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.0005, random_state=0)

data_train_x = torch.from_numpy(x_train)                            # array to tensor
data_train_y = torch.from_numpy(y_train)
data_test_x = torch.from_numpy(x_test)
data_test_y = torch.from_numpy(y_test)

# create tensor datasets
trainset = TensorDataset(data_train_x, data_train_y)
testset = TensorDataset(data_test_x, data_test_y)