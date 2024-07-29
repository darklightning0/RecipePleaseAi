import requests

url = "http://localhost:5001/get_recipe"
url2 = "http://localhost:5001/extract_recipe"

payload = {
    "ingredients": ["chicken breast", "curry", "pepper", "milk"],
    "unwantedIngredients": ["onion"]
}

payload2 = {

    "string": "I'm looking for something easy to cook with pasta, tomatoes, and cheese."

}

headers = {"Content-Type": "application/json"}

response = requests.post(url2, json=payload2, headers=headers)

if response.status_code == 200:
    results = response.json()
    print(results)

else:
    print(f"Error: {response.status_code}, {response.text}")
