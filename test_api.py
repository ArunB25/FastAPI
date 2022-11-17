import base64
import requests

url = 'http://127.0.0.1:8000/predict'
with open("API_model_image.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
    
payload ={"filename": "API_model_image.jpg", "filedata": encoded_string}
resp = requests.get(url=url, data=payload) 
print(resp.text)