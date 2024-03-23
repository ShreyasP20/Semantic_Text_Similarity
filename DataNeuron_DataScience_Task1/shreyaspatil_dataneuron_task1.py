"""
Original file is located at
    https://colab.research.google.com/drive/1HUqgG0AHu5OqHD_twNONzPID6C7Cigsl
"""

import pandas as pd
import numpy as np

from google.colab import files

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

uploaded=files.upload()

df= pd.read_csv('DataNeuron_Text_Similarity.csv')

df.head()

df.tail()

df.shape

df.isna().sum()

df.isnull().sum()

df.nunique()

embeddings1 = model.encode(list(df['text1']), convert_to_tensor=True)
embeddings2 = model.encode(list(df['text2']), convert_to_tensor=True)

cosine_scores = util.cos_sim(embeddings1, embeddings2)

x= cosine_scores[0][0]

def map_to_scale(x):
    return (x + 1) / 2

for i in range(3000):
  cosine_scores[i][i]=map_to_scale(cosine_scores[i][i])

df['SimilarityScore']= False

df['SimilarityScore'] = [cosine_scores[i][i] for i in range(3000)]

df.head()

df.to_csv('Solution_Task1.csv')

