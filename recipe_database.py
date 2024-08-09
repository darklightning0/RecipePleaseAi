from pymilvus import MilvusClient
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

df = pd.read_csv('/Users/efekapukulak/Desktop/Hobbies/Coding/ML/RecipeAi/recipes_dataset/cluster_dataset_13k.csv')

ingredients_list = df['Cleaned_Ingredients'].apply(ast.literal_eval).tolist()
names_list = df['Title'].tolist()
recipe_list = df['Instructions'].tolist()
image_list = df['Image_Name'].tolist()
type_list = df['type'].tolist()
cuisine_list = df['cuisine'].tolist()
difficulty_list = df['difficulty'].tolist()
budget_list = df['budget'].tolist()
servings_list = df['servings'].tolist()
cooking_time_list = df['cooking_time'].tolist()
calories_list = df['calories'].tolist()
fat_list = df['fat'].tolist()
carbs_list = df['carbs'].tolist()
protein_list = df['protein'].tolist()
allergens_list = df['allergens'].apply(ast.literal_eval).tolist()  
overall_list = df['overall'].tolist()

data = [ 
    {
        "id": i, 
        "vector": embeddings[i], 
        "name": names_list[i], 
        "image": image_list[i], 
        "recipe": recipe_list[i], 
        "ingredients": ingredients_list[i], 
        "type": type_list[i], 
        "cuisine": cuisine_list[i], 
        "difficulty": difficulty_list[i], 
        "budget": budget_list[i], 
        "servings": servings_list[i], 
        "cooking_time": cooking_time_list[i], 
        "calories": calories_list[i], 
        "fat": fat_list[i], 
        "carbs": carbs_list[i], 
        "protein": protein_list[i], 
        "allergens": allergens_list[i], 
        "overall": overall_list[i]
    }
    for i in range(len(embeddings))
]

res = client.insert(collection_name="recipe_collection_13k_gpt", data=data)
