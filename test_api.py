import requests

#the server URL running first terminal
url="http://127.0.0.1:8000/predict"

#sample workout payload 
data = {
    "Gender": "female",
    "Age": 25,
    "Height": 165.0,
    "Weight": 60.0,
    "Duration": 30.0,
    "Heart_Rate": 140.0,
    "Body_Temp": 39.5
}

#send the automated POST request
response = requests.post(url, json=data)
print(response.json())