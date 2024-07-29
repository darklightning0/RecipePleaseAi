import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from flask import Flask, request, jsonify
from pymilvus import MilvusClient
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)

sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = MilvusClient("recipe_database_13k.db")
client.load_collection(collection_name="recipe_collection13k", replica_number=1)
openai = OpenAI(api_key = os.environ.get('GPT_TOKEN'))


recipe_prompt = """Your task is to help users find the perfect recipe based on their ingredients. Present these 5 recipes in a friendly and approachable tone, using the format provided:

5 Recipes According to Your Ingredients\n
Ingredients: (write ingredients)\n
\n
Recipe Name: (example: dessert, breakfast, dinner, snack, salad, soup, sauce, lunch, tea, barbecue, cold drink, coffee...)\n
\n
Meal Type: (example: dessert, breakfast, dinner, snack, salad, soup, sauce, lunch, tea, barbecue, cold drink, coffee...)\n
Cuisine: (estimate the recipe's cultural origin, example: Italian, Mexican, Chinese, etc.)\n
\n
Level: (categorize recipe difficulty according to ingredients and instructions, choices: Beginner, Intermediate, Advanced)\n
Estimated Cooking Time: (estimate total cooking time to prepare the recipe according to instructions. Give it in minutes, example: 30 mins)\n
\n
Servings: (estimate the number of servings according to ingredients, example: 1 servings)\n
Estimated Nutrition Facts (per serving): (estimate nutrition facts according to ingredients per serving)\n
  Calories: (example: 1 kcal)\n
  Fat: (example: 1 grams)\n
  Carbs: (example: 1 grams)\n
  Protein: (example: 1 grams)\n
\n
Possible Allergens: (estimate possible allergens; if none, write "None". Example: Dairy (butter), Nuts (almonds), Gluten (flour))\n
\n
Ingredients: \n
- (ingredient 1)\n
- (ingredient 2)\n
- ...
\n
Instructions:\n
1. (instruction 1)\n
2. (instruction 2)\n
...
\n
\n
If you have any questions about these recipes or need further assistance, don't hesitate to ask. Happy cooking!
(Please ensure each recipe is clearly separated and follows the structure provided above. Put space lines at (space). Aim to make the recipes easy to read and follow. Do not use these: * and #)
"""

messages = [{"role": "system", "content": recipe_prompt}]

#REQUEST

@app.route('/get_recipe', methods = ['POST'])
def get_recipe():
    
    data = request.get_json()
    user_ingredients = data.get("ingredients", [])
    userUnwantedIngredients = data.get("unwantedIngredients", [])

    if not data:
        return jsonify({"error": "No data"}), 400

#DATABASE SEARCH

    query_vectors = sentence_model.encode("\n".join(user_ingredients))

    results = client.search(

        collection_name="recipe_collection13k",  
        data=[query_vectors], 
        filter=f'ingredients not in {userUnwantedIngredients}',
        limit=10,  
        output_fields=["ingredients", "name", "recipe"], 

    )
    for hits in results:
         for result in hits:
              entity = result["entity"]
              for i in entity:
                   print(i)

    results_text = f"Ingredients: {', '.join(user_ingredients)}\n"
    results_text += f"Unwanted Ingredients: {', '.join(userUnwantedIngredients)}\n"
    for hits in results:
            for result in hits:
                entity = result["entity"]
                results_text += f"\nRecipe Name: {entity['name']}\nIngredients: {entity['ingredients']}\nInstructions: {entity['recipe']}\n"

    # LLM RESPONSE
    print(results_text)

    messages.append({"role": "user", "content": results_text})

    response = openai.chat.completions.create(model = "gpt-4o", messages = messages)

    return jsonify(response.choices[0].message.content), 200

#REQUEST

@app.route('/extract_recipe', methods = ['POST'])
def extract_recipe():
    
    data = request.get_json()
    string = data.get("string")

    if not data:
        return jsonify({"error": "No data"}), 400
    
    completion = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": string,
        }
    ],
    functions=[
        {
            "name": "extract_recipe",
            "description": "Extract required informations. If no information for specific field given return None.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ingredients": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "The ingredients user has.",
                        },
                    },
                    "cuisine": {
                        "type": "string",
                        "description": "The cuisine user wants to cook from."
                        },
                    "unwanted_ingredients": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "The ingredients user don't want to use..",
                        },
                    },
                    "difficulty": {
                        "type": "integer",
                        "description": "Scale the difficulty user wants between 1 and 5."
                    },
                    "meal_type": {
                        "type": "string",
                        "description": "Determine what type of meal user wants to do (example: dinner, beverage, snack...)."
                    },
                    "budget": {
                        "type": "integer",
                        "description": "Determine the budget of the user."
                    },
                    "nutrition_preferences": {
                        "type": "string",
                        "description": "Determine the nutrition preferences of the user (example: high protein, low calorie, no sugar)."
                    },
                    },
                },
                "required": ["ingredients"],
            },
        
    ],
    function_call="auto",
)

    response = completion.choices[0].message.function_call.arguments
    data = eval(response)

    print(data)

    return jsonify(data), 200

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5001, processes = 1, threaded = False)

