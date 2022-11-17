import torch
import urllib
from PIL import Image
from torchvision import transforms
import requests

class Classifier:
    def __init__(self) -> None:
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.model.eval()

        # Read the categories
        self.categories = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt").text
        self.categories = self.categories.splitlines()

    def predict(self,input_image):

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model.to('cuda')

        with torch.no_grad():
            output = self.model(input_batch) # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes

        probabilities = torch.nn.functional.softmax(output[0], dim=0)# The output has unnormalized scores. To get probabilities, you can run a softmax on it.

        # Show top categories per image
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        top_categories = {}
        for i in range(top5_prob.size(0)):
            top_category = self.categories[top5_catid[i]]
            top_prob = round(top5_prob[i].item(),4)
            top_categories[top_category] = top_prob

        return top_categories

if __name__ == "__main__":
    url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
    filename = "API_model_image.jpg"
   
    # Download an example image from the pytorch website
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)   

    # sample execution (requires torchvision)
    input_image = Image.open(filename)  

    localclassifier = Classifier()
    print(localclassifier.predict(input_image))
    