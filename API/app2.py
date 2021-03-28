from flask import Flask, request
from flask_cors import CORS
import PIL
import torch
from torchvision import datasets, transforms, models
from torch import nn
from collections import OrderedDict


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


# Load saved model
def load_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(512, 400)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(400, 2)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.load_state_dict(ckpt, strict=False)
    return model


SAVE_PATH = 'res18_10.pth'

# load model
model = load_ckpt(SAVE_PATH)


test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an torch Tensor
    '''
    im = PIL.Image.open(image)
    return test_transforms(im)


def predict(image_path, model):
    # Predict the class of an image using a trained deep learning model.
    model.eval()
    img_pros = process_image(image_path)
    img_pros = img_pros.view(1, 3, 224, 224)
    with torch.no_grad():
        output = model(img_pros)
    return output


@app.route('/', methods=['GET'])
def home():
    return 'API is not running'


@app.route('/pred', methods=['POST'])
def pred():
    img = request.files['img']
    ps = torch.exp(predict(img, model))
    cls_score = int(torch.argmax(ps))
    if cls_score == 0:
        return 'Cat'
    else:
        return 'Dog'


if __name__ == '__main__':
    app.run(port=8000, debug=True)
