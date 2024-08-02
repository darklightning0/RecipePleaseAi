import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from flask import Flask, request, jsonify
from pymilvus import MilvusClient
from openai import OpenAI
import json
import umap
import hdbscan
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

client = MilvusClient("recipe_database_13k_gpt.db")
client.load_collection(collection_name="recipe_collection_13k_gpt", replica_number=1)

openai = OpenAI(api_key = os.environ.get('GPT_TOKEN'))

recipe_prompt = """Your task is to help users find the perfect recipe based on their ingredients. Present these 5 recipes in a friendly and approachable tone, using the format provided:
5 Recipes According To Your Ingredients
(respond in a JSON format like this:)
recipes: [
  {
    "name": "string",
    "type": "string", (example: Dessert, Breakfast, Dinner, Snack, Salad, Soup, Sauce, Lunch, Tea, Barbecue, Cold drink, Coffee, Beverage...)
    "cuisine": "string", (estimate the recipe's cultural origin, example: Italian, Mexican, Chinese, etc.)
    "difficulty": int,  (categorize recipe difficulty according to ingredients and instructions, choices: Beginner, Intermediate, Advanced)
    "cooking_time": "string", (estimate total cooking time to prepare the recipe according to instructions. format is in: HH hours MM minutes)
    "servings": int,
    "nutrition_facts": {"string": int, "string": int}, (estimate all the nutrition facts(calories, fat, carbs, protein) according to ingredients per serving)
    "allergens": ["string", "string"], (estimate possible allergens. Example: Dairy (butter), Nuts (almonds), Gluten (flour))
    "ingredients": ["string", "string"],
    "instructions": ["string", "string"],
  },
  {
    ...
  },

]
If you have any questions about these recipes or need further assistance, don't hesitate to ask. Happy cooking!
(Please ensure each recipe is clearly separated and follows the structure provided above. Put spaces in necessary areas. Aim to make the recipes easy to read and follow. Do not use these: * and #)
"""

messages = [{"role": "system", "content": recipe_prompt}]

#GET RECIPE

@app.route('/get_recipe', methods = ['POST'])
def get_recipe(ing = None, un_ing = [""]):

    data = request.get_json()
    user_ingredients = data.get("ingredients", [])
    userUnwantedIngredients = data.get("unwantedIngredients", [])

    if not data:
        print("NO DATA")
        user_ingredients = ing
        userUnwantedIngredients = un_ing
        if not user_ingredients:
            return jsonify({"error": "No data"}), 400
        
    if ing:
        user_ingredients = ing
        userUnwantedIngredients = un_ing


#DATABASE SEARCH

    #query_vectors = sentence_model.encode(" ".join(user_ingredients))
    
    response = openai.embeddings.create(
        input=" ".join(user_ingredients),
        model="text-embedding-3-small"
    )
    query_vectors = response.data[0].embedding
    
    results = client.search(

        collection_name="recipe_collection_13k_gpt",  
        data=[query_vectors], 
        filter=f'ingredients not in {userUnwantedIngredients}',
        limit=10,  
        output_fields=["ingredients", "name", "recipe"], 

    )
    '''
    for hits in results:
         for result in hits:
              entity = result["entity"]
              for i in entity:
                   print(i)
    '''
    results_text = f"Ingredients: {', '.join(user_ingredients)}\n"
    results_text += f"Unwanted Ingredients: {', '.join(userUnwantedIngredients)}\n"
    results_text = ""
    for hits in results:
            for result in hits:
                entity = result["entity"]
                results_text += f"\nRecipe Name: {entity['name']}\nIngredients: {entity['ingredients']}\nInstructions: {entity['recipe']}\n"

    #LLM RESPONSE
    
    messages.append({"role": "user", "content": results_text})

    response = openai.chat.completions.create(model = "gpt-4o-mini", response_format = { "type": "json_object" }, messages = messages)

    return jsonify(response.choices[0].message.content), 200


#EXTRACT RECIPE

def function_calling(string):
     
    response = openai.chat.completions.create(
    model="gpt-4o-mini",
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
                            "description": "The ingredients user has. If the ingredient is not specific, don't write it. (example: vegetables, various produce items, yellow fruits, greens)",
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
                            "description": "The ingredients user don't want to use.",
                        },
                    },
                    "difficulty": {
                        "type": "integer",
                        "description": "Scale the difficulty user wants between 1 and 5.",
                    },
                    "meal_type": {
                        "type": "string",
                        "description": "Determine what type of meal user wants to do (example: dinner, beverage, snack...)."
                    },
                    "budget": {
                        "type": "integer",
                        "description": "Determine the budget of the user and the currency its given. Convert to US dollars."
                    },
                    "nutrition_preferences": {
                        "type": "string",
                        "description": "Determine the nutrition preferences of the user (example: high protein, low calorie, no sugar).."
                    },
                    },
                },
                "required": ["ingredients"],
            },
        
    ],
    function_call={"name": "extract_recipe"},
)
    
    return response

@app.route('/extract_recipe', methods = ['POST'])
def extract_recipe():
    
    data = request.get_json()
    string = data.get("string")

    if not data:
        return jsonify({"error": "No data"}), 400
    
    response = function_calling(string).choices[0].message.function_call.arguments
    data = json.loads(response)

    print(data)

    data["difficulty"] = data.get("difficulty", 3)
    data["meal_type"] = data.get("meal_type", "any")

    #recipes = get_recipe(ing = data.get("ingredients"))


    return jsonify(data), 200


#EXTRACT INGREDIENTS FROM IMAGE


@app.route('/extract_ingredients_from_image', methods = ['POST'])
def extract_ingredients_from_image():
     
    data = request.get_json()
    image = data.get("image")

    response = openai.chat.completions.create(
         
        model="gpt-4o",
        messages=[
            {
            "role": "user",
            "content": [
                 {
                    "type": "text",
                    "text": "What ingredients do you see in this image? Specify every ingredient. Give stable answers unlike (vegetables, various produce items, yellow fruits, greens, cabbage or lettuce). Present ingredients all in lowercase."
                 },
                 {
                    "type": "image_url",
                    "image_url": 
                        { 
                            "url": f"data:image/jpeg;base64,{image}",
                            "detail": "low"
                        },
                    }
                ]
            }
        ]
    )
    print(response.choices[0].message.content)
    arguments = function_calling(response.choices[0].message.content).choices[0].message.function_call.arguments
    ingredients = json.loads(arguments)["ingredients"]

    print(ingredients)

    recipes = get_recipe(ing = ingredients, un_ing = [""])

    return recipes
      
    

@app.route('/get_relevant_recipe_titles', methods = ['POST'])
def get_relevant_recipe_titles():

    data = request.get_json()
    title = data.get("title")


    return jsonify(), 200




if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5001, processes = 1, threaded = False)

    