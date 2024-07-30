import requests
import base64

headers = {"Content-Type": "application/json"}

url = ""
payload = ""

def get_recipe():

    url = "http://localhost:5001/get_recipe"
    payload = {

    "ingredients": ["chicken breast", "curry", "pepper", "milk"],
    "unwantedIngredients": ["onion"]

}
    




def extract_recipe():

    url = "http://localhost:5001/extract_recipe"

    payload = {

        "string": "I am preparing breakfast for the evening. I have pepper, salst and 1/2 salnon. I don't like pepper."

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


url, payload = extract_from_image()

headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)

if response.status_code == 200:

    results = response.json()
    print(results)

else:
    print(f"Error: {response.status_code}, {response.text}")
