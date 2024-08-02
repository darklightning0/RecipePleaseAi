
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from sentence_transformers import SentenceTransformer
import pickle
import pandas as pd
import ast

client = MilvusClient("recipe_database_13k_gpt.db")

if client.has_collection(collection_name="recipe_collection_13k_gpt"):
    client.drop_collection(collection_name="recipe_collection_13k_gpt")
client.create_collection(
    collection_name="recipe_collection_13k_gpt",
    dimension=1536,  
)

with open('gpt_12k_embeddings.pkl', 'rb') as file:
    data = pickle.load(file)
    embeddings = data
    
df = pd.read_csv('/Users/efekapukulak/Desktop/Hobbies/Coding/ML/RecipeAi/recipes_dataset/FoodIngredientsandRecipeDatasetwithImageNameMapping.csv')

ingredients_list = df['Cleaned_Ingredients'].tolist()
names_list = df['Title'].tolist()
recipe_list = df['Instructions'].tolist()

data = [ 

    {"id": i, "vector": embeddings[i], "name": names_list[i], "recipe": recipe_list[i], "ingredients": ingredients_list[i]}
    for i in range(len(embeddings))
]

res = client.insert(collection_name="recipe_collection_13k_gpt", data=data)
