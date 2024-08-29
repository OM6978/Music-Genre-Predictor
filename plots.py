import pandas as pd
import matplotlib.pyplot as plt

#Visualisations

file = "/home/om/Acads/smai-m24-assignments-OM6978/data/external/spotify.csv"

data = pd.read_csv(file)

# print(data)
def plot_genre_vs_param(data, param):
  plt.figure(figsize=(10, 6))
  plt.scatter(data['track_genre'], data[param])
  plt.xlabel('Genre')
  plt.ylabel(param)
  plt.title(f'Genre vs. {param}')
  plt.xticks(rotation=45)
  plt.show()

data['explicit'] = data['explicit'].astype(bool)

# params_to_plot = ['loudness', 'danceability', 'popularity', 'energy', 'speechiness' , 'acousticness' , 'instrumentalness' , 'liveness' , 'valence' , 'tempo']
params_to_plot = ['loudness', 'danceability', 'popularity', 'energy', 'speechiness' , 'acousticness' , 'instrumentalness' , 'liveness' , 'valence' , 'tempo']

for param in params_to_plot:
  plot_genre_vs_param(data, param)