# importing necessary libraries

import os
import pandas as pd
import numpy as np
import json
from PIL import Image, ImageDraw
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

# Setting the base directories

data_directory = "/your/base/directory/where/the/data/is"
input_directory = os.path.join(data_directory, "Input")
output_directory = os.path.join(data_directory, "Output")

def bounding_box_viz(image_folder, which_image):
    folder_path = os.path.join(input_directory, image_folder)
    output_json_path = os.path.join(output_directory, image_folder, "results.json")
    
    image_dir = f'all-000{str(which_image)}-grayscale.png'
    image_path = os.path.join(folder_path, image_dir)

    # Load the grayscale image and convert to color
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    
    # Open the JSON file with the bounding box data
    with open(output_json_path, 'r') as file:
        objects = json.load(file)
    
    # Predefined colors for each characteristic in BGR format
    color_map = {
        "HighMoistureSimplexPage": (255, 0, 0), 
        "HighMoistureSimplexVerticalBars": (0, 255, 0),     
        "HighMoistureSimplexObject1.3cm": (0, 0, 255), 
        "HighMoistureSimplexObject2.5cm": (255, 224, 32),
        "HighMoistureDuplexObject2.2cm":(0,0,100),
        "HighMoistureDuplexPageDelta": (100,0,0)
        # Add more characteristics and colors as needed
    }

    # Prepare a legend - a list of patches and labels
    legend_patches = [Patch(color=[c/255.0 for c in color[::-1]], label=char) for char, color in color_map.items()]

    for entries in objects['algorithm_results'][which_image - 1]['results']:
        characteristic = entries['characteristic']
        
        # Draw boxes and text
        for bbox_result in entries['boundingBoxResults']:
            boxes = bbox_result.get('corners') or bbox_result.get('corners: ', [])
            color = color_map.get(characteristic, (127, 127, 127))  # Default to a gray color if not specified
            pt1, pt2 = (boxes[1][1] * 5, boxes[1][0] * 5), (boxes[3][1] *5, boxes[3][0] *5)
            cv2.rectangle(color_image, pt1, pt2, color, 5)
    
    # Convert color image to RGB for Matplotlib
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # Display the image with Matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(color_image)
    
    # Hide the axes
    ax.axis('off')

    # Add the legend outside of the image
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.show()

if __name__=='__main__':
    #example usage

    bounding_box_viz('2012-summer', 1) # folder name and page number to visualize
