import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from flask import Flask, request, jsonify, render_template
from pymilvus import MilvusClient
from groq import Groq
import google.generativeai as genai
import json
import pandas as pd

app = Flask(__name__)

client = MilvusClient("recipe_database_13k_gemini.db")
client.load_collection(collection_name="recipe_collection_13k_gemini", replica_number=1)

groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))

gmodel = "llama3-groq-70b-8192-tool-use-preview" 

gemini = genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

model = genai.GenerativeModel('gemini-1.5-flash')

file_path = './recipes_dataset/cluster_dataset_13k.csv'
df = pd.read_csv(file_path)


def get_recipe(data):

    user_ingredients = data["ingredients"]

    response = genai.embed_content(
        model="models/text-embedding-004",
        content=" ".join(user_ingredients),
        task_type="retrieval_query",
    )
    query_vectors = response["embedding"]
    results = client.search(
        collection_name="recipe_collection_13k_gemini",
        anns_field="vector",
        data=[query_vectors],
        limit=200,
        output_fields=["ingredients", "name", "recipe", "image", "type", "cuisine",
                       "difficulty", "budget", "servings", "cooking_time", "calories",
                       "fat", "carbs", "protein", "allergens", "overall", "type"],
        consistency_level="Strong"
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
        
            ranked_recipes = []
        
        for recipe in recipes:
            ranking_points = 0

            if recipe["name"] == "nan" or recipe["name"] == None or recipe["name"] == "None":
                recipes[recipe] = None

            if data.get("budget") is not None:
                if data["budget"] - 1 <= recipe["budget"] <= data["budget"] + 1:
                    ranking_points += 1

            if data.get("calories") is not None:
                if data["calories"] - 100 <= recipe["calories"] <= data["calories"] + 100:
                    ranking_points += 1

            if data.get("carbs") is not None:
                if data["carbs"] - 20 <= recipe["carbs"] <= data["carbs"] + 20:
                    ranking_points += 1

            if data.get("fat") is not None:
                if data["fat"] - 10 <= recipe["fat"] <= data["fat"] + 10:
                    ranking_points += 1

            if data.get("protein") is not None:
                if data["protein"] - 20 <= recipe["protein"] <= data["protein"] + 20:
                    ranking_points += 1

            if data.get("difficulty") is not None:
                if data["difficulty"] - 1 <= recipe["protein"] <= data["difficulty"] + 1:
                    ranking_points += 1

            if data.get("type") is not None:
                if data["type"] == recipe["type"]:
                    ranking_points += 3

            if data.get("cuisine") is not None:
                if data["cuisine"] == recipe.get("cuisine"):
                    ranking_points += 3

            if data.get("unwanted_ingredients") is not None:
                recipe_ingredients = " ".join(recipe["ingredients"]).lower().split()
                for ingredient in data["unwanted_ingredients"]:
                    if ingredient.lower() in recipe_ingredients or f"{ingredient.lower()}s" in recipe_ingredients or f"{ingredient.lower()}es" in recipe_ingredients:
                        ranking_points -= 10

            if data.get("allergens") is not None:
                recipe_allergens = " ".join(recipe["allergens"]).split()
                for allergen in data.get("allergens"):
                    if allergen in recipe_allergens:
                        ranking_points -= 10

            recipe["ranking_points"] = ranking_points
            ranked_recipes.append(recipe)
        
        ranked_recipes = sorted(ranked_recipes, key=lambda x: x["ranking_points"], reverse=True)
        
    return ranked_recipes, user_ingredients


# EXTRACT RECIPE


def function_calling(string):

    messages = [
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
                            "description": "The cuisine user wants to cook from. Return the inital letter capitalized."
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
                            "description": "Scale the difficulty user wants between 1-5, min: 1 max: 5."
                        },
                        "budget": {
                            "type": "integer",
                            "description": "Rate the budget user has in integer between 1-5, min: 1 max: 5."
                        },
                        "type": {
                            "type": "string",
                            "description": "Determine what type of meal user wants to do (example: Dinner, Beverage, Snack, Side Dish...). Return the inital letter capitalized."
                        },
                        "calories": {
                            "type": "integer",
                            "description": "Estimate how much calorie user wants."
                        },
                        "fat": {
                            "type": "integer",
                            "description": "Estimate how much fat user wants."
                        },
                        "carbs": {
                            "type": "integer",
                            "description": "Estimate how much carbs user wants."
                        },
                        "protein": {
                            "type": "integer",
                            "description": "Estimate how much protein user wants."
                        },
                        "allergens": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "description": "The allergens user has. (exp: Diary, Gluten...) Return the inital letter capitalized."
                            }
                        },
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


@app.route('/extract_recipe', methods=['POST'])
def extract_recipe():

    string = request.form["query"]

    response = function_calling(string).choices[0].message.tool_calls[0].function.arguments
    returned = json.loads(response)

    recipes, ingredients = get_recipe(returned)
    return render_template('results.html', recipes=recipes, user_ingredients=ingredients)


# EXTRACT INGREDIENTS FROM IMAGE


@app.route('/extract_ingredients_from_image', methods=['POST'])
def extract_ingredients_from_image():
    print(request.files)
    if 'avatar' not in request.files:
        return "No file part", 400

    returned_file = request.files['avatar']
    if returned_file.filename == '':
        return "No selected file", 400

    returned_file.save(f"./images/{returned_file.filename}")

    file = genai.upload_file(path=f"./images/{returned_file.filename}")

    response = model.generate_content([file, "What ingredients do you see in this image? Specify every ingredient. Give stable answers unlike (vegetables, various produce items, yellow fruits, greens, cabbage or lettuce). Present ingredients all in lowercase."])
    file.delete()
    os.remove(f"./images/{returned_file.filename}")

    arguments = function_calling(response.text).choices[0].message.tool_calls[0].function.arguments
    data = json.loads(arguments)

    recipes, ingredients = get_recipe(data)
    return render_template('results.html', recipes=recipes, user_ingredients=ingredients)


@app.route('/get_relevant_recipe_titles', methods=['POST'])
def get_relevant_recipe_titles():

    data = request.get_json()
    title = data.get("title")

    if not title:
        return jsonify({"error": "No title provided"}), 400

    recipe = df[df['Title'].str.lower() == title.lower()]

    if recipe.empty:
        return jsonify({"error": "Recipe not found"}), 404

    cluster_no = recipe.iloc[0]['Cluster_No']

    cluster_recipes = df[df['Cluster_No'] == cluster_no].sample(n=6)

    main_recipe = recipe.iloc[0].to_dict()
    main_recipe["Cleaned_Ingredients"] = eval(main_recipe["Cleaned_Ingredients"])
    main_recipe["Instructions"] = main_recipe["Instructions"].split("\n")
    main_recipe["allergens"] = eval(main_recipe["allergens"])
    recommended_recipes = cluster_recipes.to_dict(orient='records')

    return render_template('recipe.html', main_recipe=main_recipe, recommended_recipes=recommended_recipes)


@app.route('/ai_chat', methods=['POST'])
def ai_chat():

    data = request.get_json()
    title = data.get("title")
    user_input = data.get("query")

    recipe = (df[df['Title'].str.lower() == title.lower()]).iloc[0]

    messages = [
        {
            "role": "system",
            "content": "You are answering user's questions about this recipe with a friendly tone. Write in one paragraph. Recipe Ingredients: " +
                       recipe["Ingredients"] + "Instructions: " + recipe["Instructions"]
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    response = groq.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=messages,
    )

    return response.choices[0].message.content


# HTML-----------------------------------------------------------------------------------------------------------------------------


@app.route('/')
def render():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5001, processes=1, threaded=False)
