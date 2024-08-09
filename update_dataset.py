import pandas as pd
from groq import Groq
import os
import json

# Initialize Groq
groq = Groq(api_key=os.getenv('GROQ_API_KEY'))
gmodel = "llama3-groq-70b-8192-tool-use-preview"

# Load the CSV file
file_path = '/Users/efekapukulak/Desktop/Hobbies/Coding/ML/RecipeAi/recipes_dataset/cluster_dataset_13k.csv'
df = pd.read_csv(file_path)

recipe_prompt = """
(Notes)
#Determine the values according to these comments.
#type (example: Dessert, Breakfast, Dinner, Snack, Salad, Soup, Sauce, Lunch, Tea, Barbecue, Cold drink, Coffee, Beverage...)
#cuisine (estimate the recipe's cultural origin, example: Italian, Mexican, Chinese, etc.)
#difficulty (rate recipe difficulty according to ingredients and instructions between 1 and 5)
#budget (rate recipe budget according to ingredients and instructions between 1 and 5)
#cooking_time (estimate total cooking time to prepare the recipe according to instructions. format is in: HH hours MM minutes)
#nutrition_facts (estimate all the 4 nutrition facts(calories, fat, carbs, protein) according to ingredients per serving.)
#allergens (estimate possible allergens. Example: Dairy (butter), Nuts (almonds), Gluten (flour))
#overall (give an overall rate for the recipe between 1 and 5 according to all these factors (exp. clearness of instructions, commonnes of ingredients, budget, cooking time and etc.))

(JSON format below)
{"recipes":[
  {
    "type": string, 
    "cuisine": string, 
    "difficulty": integer, 
    "budget": integer,
    "cooking_time": string, 
    "servings": int,
    "calories": int,
    "fat": int,
    "carbs": int,
    "protein": int, 
    "allergens": List(string), 
    "overall": int,
  },
  {
    ...
  },
]
}
"""

# Function to prepare request payload
def prepare_payload(chunk):
    recipes = []
    for _, row in chunk.iterrows():
        recipes.append({
            "name": row['Title'],
            "ingredients": row['Cleaned_Ingredients'],
            "instructions": row['Instructions']
        })
    return {"recipes": recipes}

# Function to process the Groq response
def process_response(response, chunk):
    data = json.loads(response.choices[0].message.content)["recipes"]
    for i, row in enumerate(chunk.index):
        for col in data[i]:
            if col in df.columns:
                df.at[row, col] = data[i].get(col, "")

# Process the DataFrame in chunks
chunk_size = 5
for start in range(0, len(df), chunk_size):
    chunk = df[start:start + chunk_size]

    # Filter out rows where 'calories' column is not empty
    chunk = chunk[chunk['calories'].isna()]

    # Skip the chunk if all rows have 'calories' filled
    if chunk.empty:
        continue

    payload = prepare_payload(chunk)

    # Send request to Groq
    for i in range(3):
        try:
            response = groq.chat.completions.create(model=gmodel, response_format={"type": "json_object"}, messages=[{"role": "system", "content": recipe_prompt},{"role": "user", "content": json.dumps(payload)}])
            process_response(response, chunk)
            break
        except Exception as e:
            print(e)
            continue
    
    df.to_csv('/Users/efekapukulak/Desktop/Hobbies/Coding/ML/RecipeAi/recipes_dataset/cluster_dataset_13k.csv', index=False)
    print(f"Processed chunk starting at index {start}")
