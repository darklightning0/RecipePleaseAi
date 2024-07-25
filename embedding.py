
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import euclidean_distances
import pickle

file_path = '/Users/efekapukulak/Desktop/Hobbies/Coding/ML/RecipeAi/recipes_dataset/full_dataset.csv'
df = pd.read_csv(file_path)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

ingredients_list = df['NER'].tolist()

embeddings = model.encode(ingredients_list)

with open('recipes_embeddings.pkl', 'wb') as file:
    pickle.dump({'embeddings': embeddings}, file)