import requests
import base64


headers = {"Content-Type": "application/json"}

def get_recipe():

    url = "http://localhost:5001/get_recipe"

    payload = {

    "ingredients": ["tomato, onion, garlic, salt, pepper, olive oil"],
    "unwantedIngredients": [""]

}
    
    return url, payload
    




def extract_recipe():

    url = "http://localhost:5001/extract_recipe"

    payload = {

        "string": "I want to make a dessert without dairy. I have bananas, oats, and almond milk. Also, it shouldn't be too difficult"

    }

    return url, payload
    




def extract_from_image():

    url = "http://localhost:5001/extract_ingredients_from_image"

    image_path = "/Users/efekapukulak/Desktop/Hobbies/Coding/ML/RecipeAi/recipes_dataset/test/img2.jpg"
    
    with open(image_path, "rb") as image_file:
        image_code = base64.b64encode(image_file.read()).decode('utf-8')

    payload = {
        "image": image_code
    }

    return url, payload




def get_relevant_recipes():

    url = "http://localhost:5001/get_relevant_recipe_titles"

    payload = {

        "title": "Caesar Salad Roast Chicken"

    }

    return url, payload





url, payload = get_relevant_recipes()




headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)

if response.status_code == 200:

    results = response.json()
    print(results)

else:
    print(f"Error: {response.status_code}, {response.text}")
