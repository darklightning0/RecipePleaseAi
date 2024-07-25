
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import pickle
import pandas as pd
import ast

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = MilvusClient("recipe_database.db")

if client.has_collection(collection_name="recipe_collection"):

    client.drop_collection(collection_name="recipe_collection")

client.create_collection(

    collection_name = "recipe_collection",
    dimension = 384, 

)

with open('recipes_embeddings.pkl', 'rb') as file:
    data = pickle.load(file)
    embeddings = data['embeddings']


df = pd.read_csv('/Users/efekapukulak/Desktop/Hobbies/Coding/ML/RecipeAi/recipes_dataset/FoodIngredientsandRecipeDatasetwithImageNameMapping.csv')

ingredients_list = df['Cleaned_Ingredients']
names_list = df['Title'].tolist()
recipe_list = df['Instructions'].tolist()
image_names_list = df['Image_Name']

data = [ 

    {"id": i, "vector": embeddings[i], "name": names_list[i], "recipe": recipe_list[i], "ingredients": ast.literal_eval(ingredients_list[i]), "images": image_names_list[i]}
    for i in range(len(embeddings))
]

res = client.insert(collection_name="recipe_collection", data=data)