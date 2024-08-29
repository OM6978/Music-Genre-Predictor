import pandas as pd
from ...models.knn.knn import knn
from ...performance.measures import measures
import numpy as np

file = "/home/om/Acads/smai-m24-assignments-OM6978/data/external/spotify-2/train.csv"
file2 = "/home/om/Acads/smai-m24-assignments-OM6978/data/external/spotify-2/validate.csv"

data = pd.read_csv(file)
test = pd.read_csv(file2,nrows=20)

to_drop = ['Unnamed: 0','track_id','artists','album_name','track_name','explicit','mode']

for column in to_drop:
    del data[column]
    del test[column]

#normalize data

data_norm = data.copy()
for column in data_norm.columns:
    if(column == 'track_genre'): continue
    data_norm[column] = abs(data[column])  / (data[column].abs().max() - data[column].abs().min())

test_norm = test.copy()
for column in test_norm.columns:
    if(column == 'track_genre'): continue
    test_norm[column] = abs(test[column])  / (data[column].abs().max() - data[column].abs().min())

data_norm = data_norm.to_numpy()
test_norm = test_norm.to_numpy()

mask = np.ones(data_norm.shape[1], dtype=bool)
mask[-1] = False

data_norm[:,mask] = data_norm[:,mask].astype(np.float32)
test_norm[:,mask] = test_norm[:,mask].astype(np.float32)

#train model

spotify_model = knn(data_norm)
predicted = spotify_model.test_data(test_norm,10,'manhattan')

knn_perf = measures(test_norm[:,-1],predicted)
print("accuracy:" , knn_perf.accuracy())