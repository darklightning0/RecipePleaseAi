
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import pickle
import pandas as pd
import ast

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

client = MilvusClient("recipe_database_13k.db")

if client.has_collection(collection_name="recipe_collection13k"):
    client.drop_collection(collection_name="recipe_collection13k")
client.create_collection(
    collection_name="recipe_collection13k",
    dimension=384,  
)

with open('recipes_embeddings.pkl', 'rb') as file:
    data = pickle.load(file)
    embeddings = data['embeddings']

df = pd.read_csv('/Users/efekapukulak/Desktop/Hobbies/Coding/ML/RecipeAi/recipes_dataset/FoodIngredientsandRecipeDatasetwithImageNameMapping.csv')

ingredients_list = df['Cleaned_Ingredients']
names_list = df['Title'].tolist()
recipe_list = df['Instructions'].tolist()

data = [ 

    {"id": i, "vector": embeddings[i], "name": names_list[i], "recipe": recipe_list[i], "ingredients": ast.literal_eval(ingredients_list[i])}
    for i in range(len(embeddings))
]

res = client.insert(collection_name="recipe_collection13k", data=data)