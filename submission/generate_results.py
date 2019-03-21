import torch
import numpy as np
from torchvision.transforms import transforms

class GenerateFinalDetections():
    def __init__(self):
        # load saved model
        model_file = 'model.pth'
        self.model = torch.load(model_file)

        # put into evaluation mode
        self.model.eval()
        # define pre-process transforms
        imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transformation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((432, 648)),
            transforms.ToTensor(),
            transforms.Normalize((imagenet_stats[0]), (imagenet_stats[1]))])

    def convert_coords(self, input_coords):
        # copy input to new tensor
        coords = input_coords.clone().detach()
        # scale based on real image size (1296x864)
        coords = coords.reshape(4,2)
        coords[:,1] = ((coords[:,1] + 1) * (1296 / 2 ))  # x
        coords[:,0] = ((coords[:,0] + 1) * (864 / 2 ))  # y
        # swap x (second column) and y (first column)
        coords = torch.stack([coords[:,1], coords[:,0]], dim=1)
        # flatten
        coords = coords.flatten()
        # assume 100% confidence
        confidence = 1.0
        # return as list of floats
        return [[float(c) for c in coords] + [confidence]]

    def predict(self, img):
        img_tensor = self.transformation(img).unsqueeze_(0).to('cuda')
        preds = self.model(img_tensor)
        return self.convert_coords(preds)
