import requests

url = "http://localhost:5001/get_recipe"

payload = {
    "ingredients": ["tomato", "potato", "onion", "oil"]
}

#for i in payload["ingredients"]:
#    print(i)

headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)

if response.status_code == 200:
    results = response.json()
    print(results)

else:
    print(f"Error: {response.status_code}, {response.text}")
