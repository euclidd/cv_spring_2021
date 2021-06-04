import argparse
import logging
import copy
import os

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

from PIL import Image


"""

To run this script:
1. first command line argument is the directory with source images: 
    - content image should be called "content.jpg"
    - style image should be called "style.jpg"
2. second command line argument is the directory where the output of the model will be saved
3. thirs command line argument is the number of steps for optimization

So you must be inside the nst_project directory and run something like this:
$ python3 nst.py van_gogh results 300

!!! The results folder must exist otherwise python will throw an error !!!

"""


# Create argument parser
parser = argparse.ArgumentParser()
parser.add_argument("img_dir")
parser.add_argument("res_dir")
parser.add_argument("total_steps")
args = parser.parse_args()

total_steps = int(args.total_steps)


# Initialize global constants
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512
img_folder = args.img_dir
results_folder = args.res_dir

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# These transformations are used to resize an image and convert it to Pytorch tensor
loader = transforms.Compose([transforms.Resize(imsize),  transforms.ToTensor()])


def load_img(image_name):
    """ Load an image into device """
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def save_tensor_as_img(tensor, name):
    """ Save image to local folder """
    img = tensor[0]
    save_image(img, os.path.join(results_folder, name))


# Loading of images
content_img = load_img(os.path.join(img_folder, "content.jpg"))
style_img = load_img(os.path.join(img_folder, "style.jpg"))

assert content_img.size() == style_img.size(), "Content and Style images should be of same shape"

# Saving style and content images
save_tensor_as_img(content_img, "content.png")
save_tensor_as_img(style_img, "style.png")

# Use content image as input image
input_img = content_img.clone()

# We can use white noise as input image, but the results will be different
# input_img = torch.randn(content_img.data.size(), device=device)

# Save input image
save_tensor_as_img(input_img, "input.png")


def compute_gram_matrix(input):
    """ Compute gram matrix for Style Loss
    
    a = batch size
    b = num of feature maps
    c, d = dimensions of a feature map
    """
    a, b, c, d = input.size()  
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    # Normalize gram matrix
    return G.div(a * b * c * d)


class ContentLoss(nn.Module):
    """ Layer to compute content loss and pass the data forward without modifying it """
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    """ Layer to compute style loss and pass the data forward without modifying it """
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = compute_gram_matrix(target_feature).detach()

    def forward(self, input):
        G = compute_gram_matrix(input)
        self.loss = mse_loss(G, self.target)
        return input


# Here we choose which layers will be used to compute content and style loss
content_layers_list = ['conv_4']
style_layers_list = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


# We use Limited-memory BFGS optimizer here since it was used by the author of the original paper: https://arxiv.org/abs/1508.06576
def set_optimizer(img):
    """ Tell our model that img is the parameter that requires a gradient"""
    return optim.LBFGS([img.requires_grad_()])


# Import pretrained vgg19 model (from the paper) and set it to eval mode
vgg_net = models.vgg19(pretrained=True).features.to(device).eval()

# VGG network are normalized with special values for the mean and std
# In order to work with pretrained VGG network we need to normalize our images 
# with the mean and std computed on ImageNet dataset. Here we use these values
# since they're openly available on the internet
imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
imagenet_std  = torch.tensor([0.229, 0.224, 0.225]).to(device)


# Normalize an image in order for it to be usable by the pretrained vgg19 model
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


def get_style_model_and_losses(vgg_model,
                               normalization_mean,
                               normalization_std,
                               style_img,
                               content_img,
                               content_layers=content_layers_list,
                               style_layers=style_layers_list):
    
    # Make a copy of pretrained vgg19
    cnn = copy.deepcopy(vgg_model)
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []

    # Here we constructing new Sequential model base on VGG19 architecture with addition of content and style loss layers
    model = nn.Sequential(normalization)

    cx = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            cx += 1
            name = 'conv_{}'.format(cx)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(cx)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(cx)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(cx)
        else:
            raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

        model.add_module(name, layer)

        # Add content loss layer
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(cx), content_loss)
            content_losses.append(content_loss)

        # Add style loss layer
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(cx), style_loss)
            style_losses.append(style_loss)

    # Dispose from layers after the content and loss layers
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def run_style_transfer(pretrained_model,
                       normalization_mean,
                       normalization_std,
                       content_img,
                       style_img,
                       input_img,
                       num_steps=100,
                       style_weight=1000000,
                       content_weight=1):
    
    """ Main inference function """
    model, style_losses, content_losses = get_style_model_and_losses(pretrained_model, normalization_mean, normalization_std, style_img, content_img)
    optimizer = set_optimizer(input_img)
    logging.info("Starting inference")
    
    curr_step = [0]
    while curr_step[0] <= num_steps:

        def closure():
            # Unlike some other librarise Pytorch does not use [0, 255] range, but rather [0, 1] range for tensor values, so we need to use clamp
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for cl in content_losses:
                content_score += cl.loss

            for sl in style_losses:
                style_score += sl.loss

            # Weights decide how important style and content losses are
            style_score *= style_weight
            content_score *= content_weight

            # Compute total loss
            loss = style_score + content_score
            loss.backward()

            # Log info about optimization process
            curr_step[0] += 1
            if curr_step[0] % 50 == 0:
                logging.info(f"Step #{curr_step[0]}")
                logging.info(f"Style Loss: {style_score.item():.4f} | Content Loss: {content_score.item():.4f}")

            return content_score + style_score

        optimizer.step(closure)

    # Pytorch needs tensor values to be between 0 and 1
    input_img.data.clamp_(0, 1)

    return input_img


# Run NST
output = run_style_transfer(vgg_net,
                            imagenet_mean,
                            imagenet_std ,
                            content_img,
                            style_img,
                            input_img,
                            num_steps=total_steps)

# Save result
save_tensor_as_img(output, "result.png")