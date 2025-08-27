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
from src.preprocessing import preprocessing                 # for def of preprocessing
from collections import Counter                             # for definition of unique words (tokens) in dataframe



df = pd.read_csv(r"C:\Users\Admin\WORK\Project_CV\Model_NLP_sentiment\data\clean_IMDB_Dataset.csv")    # insert path to your data
df = df.sample(1600)                                                                        # Choose count of review  for test model (example 100)

# def transform_label(label):
#     return 1 if label == 'positive' else 0

# df['label'] = df['sentiment'].progress_apply(transform_label)



# # df["label"] = df["sentiment"].apply(lambda x: x == "positive")
# df.loc[:,'clean'] = df.loc[:, 'review'].apply(preprocessing)


reviews = df.clean.values                                 # get all processed reviews
words = " ".join(reviews)                                 # merge into single variable, separated by whitespaces
words = words.split()                                     # obtain list of words


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
    features = np.full((len(reviews), seq_length), pad_id, dtype=int)      #Return a new array of given shape (len(reviews), seq_length) and type = int, filled with pad_id.
    for i, row in enumerate(reviews):
        features[i, :len(row)] = np.array(row)[:seq_length]               # if seq_length < len(row) then review will be trimmed
    return features

seq_length = 200
features = pad_features(reviews_enc, pad_id=word2int['<PAD>'], seq_length=seq_length)       # creating array len 256 filled zero up and then it records real values
                                                                                            # if len < 256 there will be zero 
assert len(features) == len(reviews_enc)                                                    # check count of reviews (true or false)
assert len(features[0]) == seq_length                                                       # check count of  words in review [0] (true or false)



# get labels as numpy
labels = df.label.to_numpy()
labels

# train test splitting
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0, stratify=labels)

data_train_x = torch.from_numpy(x_train)                            # array to tensor
data_train_y = torch.from_numpy(y_train)
data_test_x = torch.from_numpy(x_test)
data_test_y = torch.from_numpy(y_test)

# create tensor datasets
trainset = TensorDataset(data_train_x, data_train_y)
testset = TensorDataset(data_test_x, data_test_y)