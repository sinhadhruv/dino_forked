from venv import create
import numpy as np
from os import listdir
import glob
import csv 

import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import torch
from PIL import Image
from PIL import ImageFile
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.cluster.vq import vq

import os
import sys
import shutil
import argparse 


def create_features(image_names):
    print("----Starting Feature Extraction----")
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # Utilize pretrained ResNet 18
    model = models.resnet18(pretrained=True)
    # Convert fully connected output layer to identity layer (strip so output is feature map)
    model.fc = nn.Identity()
    # Set model to eval mode
    model.eval()


    # Turn off gradient calculation
    with torch.no_grad():
        # Create transformation for input images to resnet model
        # Source: https://learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        START_INDEX = 0
        END_INDEX = len(image_names)

        # Run each image through the model to obtain feature maps
        image_features = []
        for i in range(START_INDEX, END_INDEX):
            if i%3==0:
                if i % 250 == 0:
                    print("{}% Complete".format(100*round(i / len(image_names), 3)))
                filename = image_names[i]
                try:
                    img = Image.open(filename)
                except:
                    print('Bad image found')
                    continue

                # Transform image to match model training input
                img_t = transform(img)
                batch_t = torch.unsqueeze(img_t, 0)

                # Run tensor through model and add features to list
                output_features = model(batch_t)
                image_features.append(output_features)

        return image_features

def get_keyframe_indices(features_list, summary_proportion=0.01):
    '''
    Get the indices of keyframes from within the dive
    Inputs:
        image_features: (list) of pytorch tensors. Each tensor is 512 dimension
                        vector of the the features extracted by CNN
        summary_proportion: (float) proportion of images that should be selected
                            as keyframes
    Returns:
        kf_indices: (list) of indices of keyframes from within the dive
    '''
    print("----Start Clustering----")
    num_frames = len(features_list)
    features_array = np.array(torch.cat(features_list))
    n_clusters = round(num_frames * summary_proportion)

    gmm = GaussianMixture(n_components=n_clusters)
    gmm.fit(features_array)
    centroids = gmm.means_
    kf_indices, distances = vq(centroids, features_array)
    kf_indices.sort()
    print("----End Clustering----")

    return kf_indices


def make_summary(kf_indices, image_names):
    kf_filenames = np.array(image_names)[kf_indices]
    kf_filenames.sort() 
    return kf_filenames

if __name__ == '__main__':


    # mypath = 'result-SB0318/attention'
    # image_names = glob.glob("result-SB0318/attention/*")
    # main_images_path = "/net/projects/soirov/mounting_root/images/SB0318/"
    # main_images = glob.glob("/net/projects/soirov/mounting_root/images/SB0318/*")
    # parent_dir= "/home/dhruvs/dino/"
    # directory = "result-SB0318/summary_frames_new/"
    #key_frames_output_path = "/home/dhruvs/dino/result-SB0318/summary_frames_new/""

    parser = argparse.ArgumentParser(description='Inputs to get key frames')
    parser.add_argument('--attention_path', type=str, required=True, help='Path to folder containing attention images for a dive')
    parser.add_argument('--main_images_path', type=str, required=True, help='Path to folder containing main images for the dive')
    parser.add_argument('--key_frames_output_path', type=str, required=True, help='Path to folder where the key frame will be stored')
    args = parser.parse_args()
    
    main_images_path = args.main_images_path
    main_images = glob.glob(main_images_path+"/*")
    image_names = glob.glob(args.attention_path+"/*")
    path_new = args.key_frames_output_path

    features_list = create_features(image_names)
    kf_indices = get_keyframe_indices(features_list)
    final_paths = make_summary(kf_indices, image_names)

    # corresponding to each attention key frame, retrieve the main images     
    main_paths = [os.path.join(main_images_path,s.split('-')[-1]) for s in final_paths]

    
    #path_new = os.path.join(parent_dir, directory)
    path_new = args.key_frames_output_path
    if os.path.exists(path_new):
        shutil.rmtree(path_new)
    os.mkdir(path_new)
    for path in main_paths:
        shutil.copy(path, path_new)
