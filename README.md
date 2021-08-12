# Tracking the Surfer Population through Webcams

The aim of this project is to count the number of surfers in an image, so the population of surfer throughout the day and seasons can be automatically captured for further analysis.

This is a Final Project in the Deep Learning Class at the University of San Fransicso.

# Inspiration

We drew inspiration from the [Dino Repository](https://github.com/facebookresearch/dino) which presents a paper on [Self-Supervised Learning with Vision Transformers](https://arxiv.org/abs/2104.14294).

The repo includes impressive visualizations of the Vision Transformer (ViT) attention weights and how they are quite good at attending to objects in a scene.  

# Our Work

Our work primarily focuses on applying pretrained Vision Transfomers to Surfline Webcams. The primary goal of our project was to be able to count the number of surfers present in the webcams' frame without labels. To achieve this we passed frames of the videos through the ViT and created a heatmap image from the attention mask output. This heatmap image was then passed through a blob detector to predict the number of surfers. Below is an example heatmap output from a preliminary test with a short cropped video.

![Insert test gif here]()

# Data

For this task we needed to gather and create our own novel dataset. We built a script to automatically pull rewind data from the webcams at Surfline.  That code is [here](https://gist.github.com/tukavic/da2238ed28eeb00b97d9acecf29c2076). Premium membership is required for access. In order to reduce the amount and size of data we decided to keep very few frames from each video. Additionally, since the webcam we accessed was not zoomed in, we need to cut out the possible distractions such as land and fences.

### Preprocessing Pipeline:

1. Download a day’s worth of videos (5:30am - 8:30pm) in 10 minute increments
2. Take 1 frame per minute of video and save timestamp
3. Crop each frame to only focus on surfing area

Here is an example of what an uncropped image looked like:

![Example image from webcam](dino/surf.png)

And here is an example frame from after preprocessing:

![Example cropped frame (doesn't work because of space in filename)](dino/data/images/2021-08-11-10:44:00-07:00.png)

# Pipeline

Our pipeline from Raw mp4 video to image counts is as follows:

1. Read mp4 videos and split by frame. Grab 1 frame per minute.
2. Pass frame through ViT model, return attention output
3. Isolate attention from `CLS` token and sum heads' attention together.
4. Run blob detection on each image and log result.
5. Loop through for each image in set.
