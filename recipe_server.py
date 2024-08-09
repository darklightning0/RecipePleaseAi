import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from flask import Flask, request, jsonify, render_template
from pymilvus import MilvusClient
from groq import Groq
from openai import OpenAI
import json
import pandas as pd
import base64

app = Flask(__name__)

client = MilvusClient("recipe_database_13k_gpt.db")
client.load_collection(collection_name="recipe_collection_13k_gpt", replica_number=1)

groq = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

gmodel = "llama3-groq-70b-8192-tool-use-preview" #llama-3.1-70b-versatile   llama-3.1-8b-instant

openai = OpenAI(api_key=os.environ.get('GPT_TOKEN'))


file_path = '/Users/efekapukulak/Desktop/Hobbies/Coding/ML/RecipeAi/recipes_dataset/cluster_dataset_13k.csv'
df = pd.read_csv(file_path)

#GET RECIPE

def get_recipe(data):


    user_ingredients = data["ingredients"]
    #user_unwanted_ingredients = data["unwanted_ingredients"]


#DATABASE SEARCH

    #query_vectors = sentence_model.encode(" ".join(user_ingredients))
    
    response = openai.embeddings.create(
        input=" ".join(user_ingredients),
        model="text-embedding-3-small"
    )
    query_vectors = response.data[0].embedding
    
    results = client.search(

        collection_name="recipe_collection_13k_gpt",  
        anns_field="vector",
        data=[query_vectors], 
        #filter=f'budget <= {data["budget"]} AND difficulty <= {data["difficulty"]} AND cuisine == "{data["cuisine"]}" AND NOT contains(allergens, "{data["unwanted_ingredients"][0]}")',
        limit=50,  
        output_fields=["ingredients", "name", "recipe", "image", "type", "cuisine", "difficulty", "budget", "servings", "cooking_time", "calories", "fat", "carbs", "protein", "allergens", "overall"], 

    )

    recipes = []
    for hits in results:
            for result in hits:
                entity = result["entity"]
                recipe = {
                    "name": entity['name'],
                    "ingredients": entity['ingredients'],
                    "instructions": entity['recipe'],
                    "image": entity['image'],
                    "type": entity['type'],
                    "cuisine": entity['cuisine'],
                    "difficulty": entity['difficulty'],
                    "budget": entity['budget'],
                    "cooking_time": entity['cooking_time'],
                    "servings": entity['servings'],
                    "calories": entity['calories'],
                    "fat": entity['fat'],
                    "carbs": entity['carbs'],
                    "protein": entity['protein'],
                    "allergens": entity['allergens'],
                    "overall": entity['overall'],
                }

                recipes.append(recipe)
  
    return recipes, user_ingredients

#EXTRACT RECIPE

def function_calling(string):
     
    messages=[
        {
            "role": "user",
            "content": string,
        }
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_recipe",
                "description": "Extract required information. If no information for specific field is given, return None.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ingredients": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "description": "The ingredients user has. If the ingredient is not specific, don't write it. (example: vegetables, various produce items, yellow fruits, greens)"
                            }
                        },
                        "cuisine": {
                            "type": "string",
                            "description": "The cuisine user wants to cook from."
                        },
                        "unwanted_ingredients": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "description": "The ingredients user don't want to use."
                            }
                        },
                        "difficulty": {
                            "type": "integer",
                            "description": "Scale the difficulty user wants between 1-5, min: 1 max: 5"
                        },
                        "budget": {
                            "type": "integer",
                            "description": "Rate the budget user has in integer between 1-5, min: 1 max: 5."
                        },
                        "meal_type": {
                            "type": "string",
                            "description": "Determine what type of meal user wants to do (example: dinner, beverage, snack...)."
                        },
                        "cooking_time": {
                            "type": "string",
                            "description": "Scale the time user has for the recipe between 1-5(1 = little time, 5 = a lot of time)."
                        },
                        "nutrition_preferences": {
                            "type": "string",
                            "description": "Determine the nutrition preferences of the user (example: high protein, low calorie, no sugar)."
                        }
                    }
                },
                "required": ["ingredients"]
            }
        }
    ]

    response = groq.chat.completions.create(
        model=gmodel,
        messages=messages,
        tools=tools,
        tool_choice="required"
    )

    return response

@app.route('/extract_recipe', methods = ['POST'])
def extract_recipe():
    
    string = request.form["query"]
    
    response = function_calling(string).choices[0].message.tool_calls[0].function.arguments
    returned = json.loads(response)

    recipes, ingredients = get_recipe(returned)
    return render_template('results.html', recipes=recipes, user_ingredients=ingredients)


#EXTRACT INGREDIENTS FROM IMAGE


@app.route('/extract_ingredients_from_image', methods = ['POST'])
def extract_ingredients_from_image():
     
    if 'avatar' not in request.files:
        return "No file part", 400

    file = request.files['avatar']
    if file.filename == '':
        return "No selected file", 400

    if file:
        image_data = base64.b64encode(file.read()).decode('utf-8')

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
                        "image_url": { 
                            "url": f"data:image/jpeg;base64,{image_data}",
                            "detail": "low"
                        },
                    }
                ]
            }
        ]
    )

    arguments = function_calling(response.choices[0].message.content).choices[0].message.tool_calls[0].function.arguments
    ingredients = json.loads(arguments)["ingredients"]

    recipes = get_recipe({"ingredients": ingredients})
    return render_template('results.html', recipes=recipes)
      
    

@app.route('/get_relevant_recipe_titles', methods = ['POST'])
def get_relevant_recipe_titles():

    data = request.get_json()
    title = data.get("title")

    if not title:
        return jsonify({"error": "No title provided"}), 400

    recipe = df[df['Title'].str.lower() == title.lower()]

    if recipe.empty:
        return jsonify({"error": "Recipe not found"}), 404

    cluster_no = recipe.iloc[0]['Cluster_No']

    cluster_recipes = df[df['Cluster_No'] == cluster_no].sample(n=5)

    main_recipe = recipe.iloc[0].to_dict()
    recommended_recipes = cluster_recipes.to_dict(orient='records')
    print(recommended_recipes)

    return render_template('recipe.html', main_recipe=main_recipe, recommended_recipes=recommended_recipes)




@app.route('/ai_chat', methods = ['POST'])
def ai_chat():

    data = request.get_json()
    title = data.get("title")
    user_input = data.get("query")
    
    recipe = (df[df['Title'].str.lower() == title.lower()]).iloc[0]

    messages = [{"role": "system", "content": "You are answering user's questions about this recipe with a frindly tone. Write in one paragraph. Recipe Ingredients: " + recipe["Ingredients"] + "Instructions: " + recipe["Instructions"]}, {"role": "user", "content": user_input}]

    response = groq.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=messages,
    )
    

    return (response.choices[0].message.content)



#HTML-----------------------------------------------------------------------------------------------------------------------------


@app.route('/')
def render():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5001, processes = 1, threaded = False)

    