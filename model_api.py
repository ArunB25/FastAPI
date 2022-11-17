from fastapi import FastAPI, Form
import model
import base64
import uvicorn
from PIL import Image

api = FastAPI()

@api.get("/predict")
def predict(filename: str = Form(...), filedata: str = Form(...)):
    image_as_bytes = str.encode(filedata)  # convert string to bytes
    img_recovered = base64.b64decode(image_as_bytes)  # decode base64string
    try:
        with open("uploaded_" + filename, "wb") as f:
            f.write(img_recovered)
    except Exception:
        return {"message": "There was an error uploading the file"}
    
    localclassifier = model.Classifier()
    # sample execution (requires torchvision)
    input_image = Image.open("uploaded_" + filename)    
    
    predicitons =  localclassifier.predict(input_image)    
    return {"message": f"Top 5 Predictions for {filename}, {predicitons}"} 

uvicorn.run(api, port=8000, host='127.0.0.1') # use uvicorn to run the api so that it processes requests asynchronously