# Tracking the Surfer Population through Webcams

The aim of this project is to count the number of surfers in an image, so the population of surfer throughout the day and seasons can be automatically captured for further analysis.

This is a Final Project in the Deep Learning Class at the University of San Fransicso.

# Inspiration

We drew inspiration from the [Dino Repository](https://github.com/facebookresearch/dino) which presents a paper on [Self-Supervised Learning with Vision Transformers](https://arxiv.org/abs/2104.14294).

The repo includes impressive visualizations of the Vision Transformer (ViT) attention weights and how they are quite good at attending to objects in a scene.  

# Our Work

Our work primarily focuses on applying pretrained Vision Transfomers to Surfline Webcams.

# Data

We built a script to automatically pull rewind data from the webcam.  That code is [here](https://gist.github.com/tukavic/da2238ed28eeb00b97d9acecf29c2076). Premium membership is required for access.


# Pipeline

Our pipeline from Raw mp4 video to image counts is as follows:

1. Read mp4 videos and split by frame. Grab 1 frame per minute.
2. Pass frame through ViT model, return attention output
3. Isolate attention from `CLS` token and sum heads' attention together.
4. Run blob detection on each image and log result.
5. Loop through for each image in set.
