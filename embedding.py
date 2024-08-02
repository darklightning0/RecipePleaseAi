import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle


file_path = '/Users/efekapukulak/Desktop/Hobbies/Coding/ML/RecipeAi/recipes_dataset/FoodIngredientsandRecipeDatasetwithImageNameMapping.csv'
df = pd.read_csv(file_path)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

title = df['Title']
ingredients = df['Cleaned_Ingredients']
instrucitons = df['Instructions']
asd = ""

for i in ingredients:
    asd += i
print(len(asd))

#combined_text_list = df['combined_text'].tolist()
"""
embeddings = model.encode(combined_text_list)

with open('recipes_embeddings_13k.pkl', 'wb') as file:
    pickle.dump({'embeddings': embeddings}, file)
"""