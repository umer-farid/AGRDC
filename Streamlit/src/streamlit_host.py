import numpy as np
import streamlit as st
import torch, time
from torchvision import models
from collections import OrderedDict
import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import Variable
from PIL import Image

uploaded_file = st.file_uploader("Choose a image file", type="jpg")
# Write a function that loads a checkpoint and rebuilds the model
import streamlit as st

# Using object notation
add_selectbox = st.sidebar.selectbox(
    "Image Classification",
    ("Realtime", "Snapshot", "Select Image")
)

# Using "with" notation
with st.sidebar:
    add_radio = st.radio(
        "Choose a shipping method",
        ("Standard (5-15 days)", "Express (2-5 days)")
    )



import json


with open('categories.json', 'r') as f:
    cat_to_name = json.load(f)

@st.cache(allow_output_mutation=True)
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model = models.resnet152()
    
    # Our input_size matches the in_features of pretrained model
    input_size = 2048
    output_size = 39
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048, 512)),
                          ('relu', nn.ReLU()),
                          #('dropout1', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(512, 39)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

# Replacing the pretrained model classifier with our classifier
    model.fc = classifier
    
    
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, checkpoint['class_to_idx']

# Get index to class mapping
loaded_model, class_to_idx = load_checkpoint('D:\Research-work-2022\Streamlit\models\plant_disease_resnet156.hdf5')
idx_to_class = { v : k for k,v in class_to_idx.items()}

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model

    size = 256, 256
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((128 - 112, 128 - 112, 128 + 112, 128 + 112))
    
    npImage = np.array(image)
    
    npImage = npImage/255.
    
        
    imgA = npImage[:,:,0]
    imgB = npImage[:,:,1]
    imgC = npImage[:,:,2]
    
    imgA = (imgA - 0.485)/(0.229) 
    print(imgA)
    imgB = (imgB - 0.456)/(0.224)
    imgC = (imgC - 0.406)/(0.225)
        
    npImage[:,:,0] = imgA
    npImage[:,:,1] = imgB
    npImage[:,:,2] = imgC
    
    npImage = np.transpose(npImage, (2,0,1))
    
    
    
    
    
    
    return npImage

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    #ax.imshow(image)
    
    return ax
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    
    image = torch.FloatTensor([process_image(Image.open(image_path))])
    model.eval()
    output = model.forward(Variable(image))
    pobabilities = torch.exp(output).data.numpy()[0]
    

    top_idx = np.argsort(pobabilities)[-topk:][::-1] 
    top_class = [idx_to_class[x] for x in top_idx]
    top_probability = pobabilities[top_idx]

    return top_probability, top_class

# Display an image along with the top 5 classes
def view_classify(img, probabilities, classes, mapper):
    ''' Function for viewing an image and it's predicted classes.
    '''
    #img_filename = img.split('\\')[-1]
    #print("img filename:", uploaded_file)
    #st.image(img)
    
    flower_name = mapper[classes[0]]
    arr = np.arange(len(probabilities))
    fig, (ax1, ax2) = plt.subplots(figsize=(3,3), ncols=1, nrows=2)
    #ax1.set_title(flower_name)
    ax1.axis('off')
    ax2.barh(arr, probabilities)
    ax2.set_yticks(arr)
    ax2.set_yticklabels([mapper[x] for x in classes])
    ax2.invert_yaxis()
    st.pyplot(fig)

Genrate_pred = st.button("Generate Prediction")    
if Genrate_pred:
    p, c = predict(uploaded_file, loaded_model)
    st.image(uploaded_file, width=300)
    st.write(int(p[0]*100),'%',c[0])
    view_classify(uploaded_file, p, c, cat_to_name)
    