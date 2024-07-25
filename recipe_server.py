
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from flask import Flask, request, jsonify
from pymilvus import MilvusClient
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import pandas as pd


app = Flask(__name__)

sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = MilvusClient("recipe_database.db")
ai_client = OpenAI(api_key = os.environ.get('GPT_TOKEN'))

recipe_prompt = """Your task is to help users find the perfect recipe based on their ingredients. Present these 5 recipes in a friendly and approachable tone, using the format provided:

5 Recipes According to Your Ingredients
Ingredients: (write ingredients)
(space)
Recipe Name: (example: dessert, breakfast, dinner, snack, salad, soup, sauce, lunch, tea, barbecue, cold drink, coffee...)
(space)
Meal Type: (example: dessert, breakfast, dinner, snack, salad, soup, sauce, lunch, tea, barbecue, cold drink, coffee...)
Cuisine: (estimate the recipe's cultural origin, example: Italian, Mexican, Chinese, etc.)
(space)
Level: (categorize recipe difficulty according to ingredients and instructions, choices: Beginner, Intermediate, Advanced)
Estimated Cooking Time: (estimate total cooking time to prepare the recipe according to instructions. Give it in minutes, example: 30 mins)
(space)
Servings: (estimate the number of servings according to ingredients, example: 4 servings)
Estimated Nutrition Facts (per serving): (estimate nutrition facts according to ingredients per serving)
  Calories: (example: 1 kcal)
  Fat: (example: 1 grams)
  Carbs: (example: 1 grams)
  Protein: (example: 1 grams)
(space)
Possible Allergens: (estimate possible allergens; if none, write "None". Example: Dairy (butter), Nuts (almonds), Gluten (flour))
(space)
Ingredients: 
- (ingredient 1)
- (ingredient 2)
- ...
(space)
Instructions:
1. (instruction 1)
2. (instruction 2)
...
(space)
(space)
If you have any questions about these recipes or need further assistance, don't hesitate to ask. Happy cooking!
(Please ensure each recipe is clearly separated and follows the structure provided above. Aim to make the recipes easy to read and follow. Do not use these: * and #)
"""

messages = [{"role": "system", "content": recipe_prompt}]

#REQUEST

@app.route('/get_recipe', methods = ['POST'])
def get_recipe():
    
    data = request.get_json()
    ingredients = data.get("ingredients", [])

    if not ingredients:
        return jsonify({"error": "No ingredients provided"}), 400

#DATABASE SEARCH

    query_vectors = sentence_model.encode("\n".join(ingredients))

    res = client.search(

        collection_name="recipe_collection",  
        data=[query_vectors], 
        limit=5,  
        output_fields=["ingredients", "name", "recipe", "images"], 

    )

    results = f"Ingredients: {ingredients}"
    for hits in res:
        for result in hits:
            entity = result['entity']
            results += f"\n Recipe Name: {entity['name']}, Recipe Index: {result['id']}, Recipe Image: {entity['images']}, Ingredients: {entity['ingredients']}, Instructions: {entity['recipe']}"

#LLM RESPONSE

    messages.append({"role": "user", "content": results})

    response = ai_client.chat.completions.create(model = "gpt-4o", messages = messages)

    return jsonify(response.choices[0].message.content), 200

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5001, processes = 1, threaded = False)

