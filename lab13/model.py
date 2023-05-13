# model.py

import torch
from torchvision.models import resnet50, ResNet50_Weights

def predict(image):
        # TODO 1 - Build a model
        weights = ResNet50_Weights.DEFAULT # YOUR CODE HERE
        model = resnet50(weights=weights) # YOUR CODE HERE
        model.eval()

        # TODO 2 - Obtain the corresponding transforms to preprocess an image
        # YOUR CODE HERE
        preprocess = weights.transforms()

        # TODO 3 - Apply inference preprocessing transforms
        # YOUR CODE HERE
        batch = preprocess(image).unsqueeze(0)
        # TODO 4 - Add a batch dimension
        # YOUR CODE HERE
        logits = model(batch)
        # TODO 5 - Predict the probability of each class
        # Note: DON'T forget to change the model to prediction model
    
        probas = logits.softmax(1)# YOUR CODE HERE - probabilities
        preds = probas.argmax(1)# YOUR CODE HERE - predictions

        # Extract the corresponding class name
        class_id = preds[0].item()
        score = probas[0][class_id].item()
        category_name = weights.meta["categories"][class_id]

        # Output dict
        output = {category_name: 100*score}

        return output

