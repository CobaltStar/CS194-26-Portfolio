# -*- coding: utf-8 -*-
"""Training_Kaggle.ipynb

Automatically generated by Colaboratory.
"""

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import matplotlib.pyplot as plt

import skimage as sk
import skimage.io as skio
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray
from skimage import data

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
import math

import matplotlib.patches as patches
import random
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm


"""PART 3 BEGINS"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

data_set_path = "/content/drive/MyDrive/Colab Notebooks/ibug_300W_large_face_landmark_dataset"

tree = ET.parse(data_set_path + '/labels_ibug_300W_train.xml')  
root = tree.getroot()
root_dir = '/content/drive/MyDrive/Colab Notebooks/ibug_300W_large_face_landmark_dataset'

face_boxes = [] 
img_filenames = []
landmark_set = []

for filename in root[2]:
  box = filename[0].attrib
  img_filenames.append(os.path.join(root_dir, filename.attrib['file']))

  box = filename[0].attrib
  # add corners with some extra padding
  new_w = int(float(box['width']) * 1.3)
  new_h = int(float(box['height'])* 1.3)
  diff_x = int((new_w - float(box['width']))/2)
  diff_y = int((new_h - float(box['height']))/2)

  face_boxes.append([float(box['left']) - diff_x, float(box['top']) - diff_y, new_w, new_h])

  # populate landmark_set
  landmarks = []
  for num in range(68):
    x_coords = int(filename[0][num].attrib['x'])
    y_coords = int(filename[0][num].attrib['y'])
    landmarks.append([x_coords, y_coords])
  landmark_set.append(landmarks)

landmark_set = np.array(landmark_set).astype('float32')   
face_boxes = np.array(face_boxes).astype('float32')

"""# Data Loading"""

from torch.utils.data import Dataset, DataLoader

def crop_img(im, box):
  # Make sure crop indices don't go out of bounds
  # for ind in box:
  
  if box[0] < 0:
    box[2] = box[2] + box[0]
    box[0] = 0
  if box[1] < 0:
    box[3] = box[3] + box[1]
    box[1] = 0
  if box[1] + box[3] > im.shape[0]:
    box[3] = im.shape[0] - box[1]
  if box[0] + box[2] > im.shape[1]:
    box[2] = im.shape[1] - box[0]
  box[3] = min(box[3], im.shape[0])
  box[2] = min(box[2], im.shape[1])
  # Crop
  im = im[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
  return im

def resize_img(im, resize_dim):
  im = resize(im, (resize_dim, resize_dim), order=0)
  return im

def crop_resize_landmarks(landmarks, box, resize_dim, imshape):
  xs = landmarks[:, 0] - box[0]
  ys = landmarks[:, 1] - box[1]

  # resize points
  scalar_x = resize_dim * 1.0 / imshape[0]
  scalar_y = resize_dim * 1.0 / imshape[1]
  xs = xs * scalar_x
  ys = ys * scalar_y
  return xs, ys

def show_im_w_pts(im, landmarks):
  plt.scatter(landmarks[:, 0], landmarks[:, 1], c='blue')
  plt.imshow(im, cmap='gray')

def save_im_w_pts(im, landmarks, fname, extra_landmarks=None):
  plt.scatter(landmarks[:, 0], landmarks[:, 1],c='red')
  if extra_landmarks is not None:
    plt.scatter(extra_landmarks[:, 0], extra_landmarks[:, 1],c='green')
  plt.imshow(im, cmap='gray')
  plt.savefig(fname)
  plt.close()

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

class LandmarkDataset(Dataset):
  def __init__(self, start_index, end_index, images_name_set, landmarks_set, face_boxes, image_dimensions=244, augment=False):   
    self.image_dimensions = image_dimensions

    self.images_name_set = images_name_set[start_index: end_index]
    self.landmarks_set = landmarks_set[start_index: end_index]
    self.face_boxes = face_boxes[start_index: end_index]

    self.augment = augment

    if augment:
      self.jitter = transforms.ColorJitter(brightness = .5, contrast=.5, saturation=.5, hue=.2)

  def __getitem__(self, i):
    if torch.is_tensor(i):
            i = i.tolist()
    landmark = np.copy(self.landmarks_set[i])
    image = sk.color.rgb2gray(skio.imread(self.images_name_set[i]))
    box = self.face_boxes[i].astype(int)

    # Crop
    image = crop_img(image, box)

    # Resize
    landmark[:, 0], landmark[:, 1] = crop_resize_landmarks(landmark, box, self.image_dimensions, image.shape)
    image = resize_img(image, self.image_dimensions)

    sample = {'image': image.astype('float32'), 'landmarks': landmark.astype('float32')}
    if self.augment:
      sample = self.augment_transform(sample)

    return sample

  def __len__(self):
    return len(self.landmarks_set)

  def augment_transform(self, sample):
    # Source: https://imgaug.readthedocs.io/en/latest/source/examples_basics.html

    # Set random transfrom params and define transform function
    img, landmarks = sample['image'], sample['landmarks']

    # Multiply Blending
    to_multiply = random.random() + 0.5
    
    # Sometimes function to drop out (tear holes in)
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential([
        iaa.Multiply(to_multiply),
        iaa.LinearContrast((0.75, 1.5)),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05)),
        sometimes(iaa.Crop(percent=(0, 0.1))),
        # Scale/zoom, translate/move, rotate, shear      
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        ),
        sometimes(iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout(
                        (0.03, 0.15), size_percent=(0.02, 0.05),
                        per_channel=0.2
                    ),
                ])),
    ])
    # Set keypoints to transform
    keypoints = KeypointsOnImage([Keypoint(x=keypt[0],y=keypt[1]) for keypt in landmarks], shape = img.shape)
    im_aug, landmarks_aug = seq(image=img.astype(np.float32), keypoints=keypoints)
    landmarks = np.array([[keypt.x, keypt.y] for keypt in landmarks_aug.keypoints])
    return {'image': im_aug, 'landmarks': landmarks}

train_dataset = LandmarkDataset(0, 6000, img_filenames, landmark_set, face_boxes, augment=True) # imgs 0-5999
print("loaded training set")
print(len(train_dataset))
validation_dataset = LandmarkDataset(6000, 6665, img_filenames, landmark_set, face_boxes, augment=False) # imgs 6000-6665
print("loaded validation set")
print(len(validation_dataset))

sample = train_dataset.__getitem__(630)
im = sample['image']
keypts = sample['landmarks']
show_im_w_pts(im, keypts)
plt.scatter(keypts[:, 0], keypts[:, 1], c='red')
plt.imshow(im, cmap='gray')

"""# Training"""

def train(dataloader, model, loss_function, optimizer):
    size = len(dataloader.dataset)
    model.train()
    
    total_loss = 0
    
    for i, batch in tqdm(enumerate(dataloader)):
        if (i % 100 == 0):
          print("processed another 100 batches")
        image, keypoint = batch['image'], batch['landmarks']
        # image = image.unsqueeze(0)
        image = image.float().unsqueeze(1)

        image, keypoint = image.to(device), keypoint.to(device)
        
        # Zero your gradients for every batch!
        model.zero_grad()
        # Make predictions for this batch
        print(image.shape)
        output = model(image)
        output = output.flatten()
        keypoint = keypoint.flatten()
        loss = loss_function(output, keypoint)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        del image
        del keypoint
        torch.cuda.empty_cache()
        
    mean_loss = total_loss / (i + 1)
        
    return mean_loss

def validate(dataloader, model, loss_function, optimizer):
    size = len(dataloader.dataset)
    model.eval()
    
    total_loss = 0
    
    for i, batch in tqdm(enumerate(dataloader)):
        image, keypoint = batch['image'], batch['landmarks']
        # image = image.unsqueeze(0)
        image = image.float().unsqueeze(1)
        image, keypoint = image.to(device), keypoint.to(device)
        
        # Zero your gradients for every batch!
        model.zero_grad()
        # Make predictions for this batch
        output = model(image)
        output = output.flatten()
        keypoint = keypoint.flatten()

        loss = loss_function(output, keypoint)     
        total_loss += loss.item()

        del image
        del keypoint
        torch.cuda.empty_cache()
        
    mean_loss = total_loss / (i + 1)
        
    return mean_loss

BATCH_SIZE = 128
NUM_WORKERS = 2
NUM_EPOCHS = 15

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

import torchvision.models as models
model = models.resnet18(pretrained=True)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(2,2), bias=False)
model.fc = torch.nn.Linear(512 * model.layer1[0].expansion, 136)
model = model.to(device, dtype=torch.float)

# Loss function
loss_fn = torch.nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # learning rate=1e-3
model

torch.nn.Linear(512 * model.layer1[0].expansion, 136)

train_losses = []
val_losses = []
import time

for epoch in range(NUM_EPOCHS):
    train_loss = train(train_loader, model, loss_fn, optimizer)
    print("Epoch Training Loss:", train_loss)
    torch.save(model.state_dict(), '/content/drive/MyDrive/Colab Notebooks/models/model'+time.strftime('%H%M%S')+'.pt')  
    val_loss = validate(val_loader, model, loss_fn, optimizer)
    print("Epoch Validation Loss:", val_loss)  
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print("finished epoch: ", epoch)

train_losses

val_losses

train_losses = [16912.487200797874,
                9921.070624168882,
                5071.137736868351,
                2394.319876163564,
                1374.3398411527594,
                1068.0849050968252,
                931.0446556578291,
                788.5899242644614,
                657.2046443858045,
                563.9719777208694,
                507.57640108149104,
                457.0535674399518,
                382.17111335916724,
                278.1155155263049,
                223.55444822920128,
                203.30848206865028,
                187.73126285634143]

val_losses = [11610.314127604166,
              6512.843668619792,
              2726.730997721354,
              844.8290303548177,
              452.859135945638,
              279.5690409342448,
              257.27233632405597,
              357.0324350992839,
              219.48041534423828,
              202.8890177408854,
              205.31771341959634,
              175.07744598388672,
              145.66884104410806,
              133.22111129760742,
              197.55113220214844,
              138.6497014363607,
              158.5707753499349]

plt.figure(figsize = (10, 7))
plt.plot(range(len(train_losses)), train_losses, label = 'training loss')
plt.plot(range(len(val_losses)), val_losses, label = 'validation loss')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.title('train and val loss over epochs for facial network')
plt.legend()
plt.savefig("more_points_train_val_loss.jpg")

"""Predict"""
data_set_path = "/content/drive/MyDrive/Colab Notebooks/ibug_300W_large_face_landmark_dataset"
tree = ET.parse(data_set_path +'/labels_ibug_300W_test_parsed.xml')
root = tree.getroot()
root_dir = '/content/drive/MyDrive/Colab Notebooks/ibug_300W_large_face_landmark_dataset'

face_boxes_test = [] 
img_filenames_test = [] 

for filename in root[2]:
  img_filenames_test.append(os.path.join(root_dir, filename.attrib['file']))
  box = filename[0].attrib
  new_w = int(float(box['width']) * 1.2)
  new_h = int(float(box['height'])* 1.2)
  diff_x = int((new_w - float(box['width']))/2)
  diff_y = int((new_h - float(box['height']))/2)
  face_boxes_test.append([float(box['left'])-diff_x, float(box['top'])-diff_y, new_w, new_h]) 
face_boxes_test = np.array(face_boxes_test).astype('float32')

class TestDataset(Dataset):
  def __init__(self, images_name_set, face_boxes, image_dimensions=244):   
    self.image_dimensions = image_dimensions

    self.images_name_set = images_name_set
    self.face_boxes = face_boxes

  def __getitem__(self, i):
    if torch.is_tensor(i):
            i = i.tolist()

    image = sk.color.rgb2gray(skio.imread(self.images_name_set[i]))
    box = self.face_boxes[i].astype(int)

    # Crop
    image = crop_img(image, box)

    # Resize
    image = resize_img(image, self.image_dimensions)

    return image

  def __len__(self):
    return len(self.images_name_set)

def predict(dataloader, model):
    outputs = []
    model.eval()
    
    for batch in dataloader:
        image = batch
        image = image.float().unsqueeze(1)
        image = image.to(device)
        
        output = model(image)

        output = output.cpu()
        outputs.append(output.detach().numpy().reshape((68, 2)))

        del image
    
    output_vector = np.stack(outputs, axis=0)
    return output_vector

NUM_WORKERS = 2
test_dataset = TestDataset(img_filenames_test, face_boxes_test)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=NUM_WORKERS)

test_predict = predict(test_loader, model)
torch.save(test_predict, 'test_predict.pt')

test_predict = torch.load("/content/test_predict.pt")

im = test_dataset[70]
keypts = test_predict[70]
show_im_w_pts(im, keypts)
plt.savefig("TestPrediction1.jpg")
keypts.shape

im = test_dataset[69]
keypts = test_predict[69]
show_im_w_pts(im, keypts)
plt.savefig("TestPrediction2.jpg")

def get_original_image_w_pts(keypoint_set, dataset, dataset_ind, im_filenames_lst, boxes_lst):
  im_ind = dataset_ind
  im = sk.color.rgb2gray(skio.imread(im_filenames_lst[im_ind]))
  box = boxes_lst[im_ind].astype(int)
  keypoints = np.copy(keypoint_set[dataset_ind])
  ret_pts = np.copy(keypoint_set[dataset_ind])
  # / dataset.image_dimensions
  ret_pts[:,0] = (keypoints[:,0] / dataset.image_dimensions)*box[2] + box[0]
  ret_pts[:,1] = (keypoints[:,1] / dataset.image_dimensions)*box[3] + box[1]
  return im, ret_pts

a, b = get_original_image_w_pts(test_predict, test_dataset, 0, img_filenames_test, face_boxes_test)
show_im_w_pts(a, b)
plt.savefig("TestPrediction3.jpg")

"""# Saving to CSV"""

def get_original_pts(keypoint_set, dataset, dataset_ind, im_filenames_lst, boxes_lst):
  box = boxes_lst[dataset_ind].astype(int)
  keypoints = np.copy(keypoint_set[dataset_ind])
  ret_pts = np.copy(keypoint_set[dataset_ind])
  # / dataset.image_dimensions
  ret_pts[:,0] = (keypoints[:,0] / dataset.image_dimensions)*box[2] + box[0]
  ret_pts[:,1] = (keypoints[:,1] / dataset.image_dimensions)*box[3] + box[1]
  return ret_pts

def to_csv(keypoints_lst, fname="/content/predictions.csv"):
  ids = np.arange(len(keypoints_lst))
  csv_array = np.zeros((len(keypoints_lst), 2))
  csv_array[:,0] = ids.astype(int)
  csv_array[:,1] = keypoints_lst
  np.savetxt(fname, csv_array, delimiter=',')

test_keypt_predictions_formatted = []

for i, keypoints in enumerate(test_predict):
  keypoints = get_original_pts(test_predict, test_dataset, i, img_filenames_test, face_boxes_test)

  for point in keypoints:
    test_keypt_predictions_formatted.append(point[0])
    test_keypt_predictions_formatted.append(point[1])

print(len(test_keypt_predictions_formatted))
to_csv(test_keypt_predictions_formatted)

import csv
with open('/content/predictions.csv',newline='') as f:
    r = csv.reader(f)
    data = [line for line in r]

"""# Try my own photos"""

Archit_box = np.array([122, 255, 401, 412])
David_box = np.array([39, 96, 154, 105])
Facebook_box = np.array([296, 281, 382, 278])
personal_boxes = np.array([Archit_box, David_box, Facebook_box])
personal_boxes

personal_img_list_names = np.array(["/content/personal_images/Archit.jpg", "/content/personal_images/David_weeb.png", "/content/personal_images/Facebook.jpg"])

personal_dataset = TestDataset(personal_img_list_names, personal_boxes)
personal_loader = DataLoader(personal_dataset, batch_size=1, num_workers=2)

import torchvision.models as models
model = models.resnet18(pretrained=True)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(2,2), bias=False)
model.fc = torch.nn.Linear(512 * model.layer1[0].expansion, 136)

model.load_state_dict(torch.load("/content/drive/MyDrive/Colab Notebooks/models/model070658.pt"))
model.to(device)

personal_predictions = predict(personal_loader, model)

a, b = get_original_image_w_pts(personal_predictions, personal_dataset, 0, personal_img_list_names, personal_boxes)
show_im_w_pts(a, b)
plt.savefig("Model3Personal1.jpg")

a, b = get_original_image_w_pts(personal_predictions, personal_dataset, 1, personal_img_list_names, personal_boxes)
show_im_w_pts(a, b)
plt.savefig("Model3Personal2.jpg")

a, b = get_original_image_w_pts(personal_predictions, personal_dataset, 2, personal_img_list_names, personal_boxes)
show_im_w_pts(a, b)
plt.savefig("Model3Personal3.jpg")




"""# Part 4 BEGINS





## Pre-process the images
"""

# Necessary functions
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import cv2

def apply_gaussian(img, kernel_size, sigma):
    gauss_1d = cv2.getGaussianKernel(kernel_size, sigma)
    gauss_kernel = np.outer(gauss_1d.T, gauss_1d)
    gauss_blurred = signal.convolve2d(img, gauss_kernel, boundary='symm', mode='same')
    return gauss_blurred

def create_heatmap(landmarks, imgsize=224, kernel_size=15, sigma=5):
  mask = np.zeros((68, 224, 224))

  filter = cv2.getGaussianKernel(kernel_size, sigma)
  kernel = np.outer(filter.T, filter)
  
  rows = landmarks[:, 0].astype(int)
  cols = landmarks[:, 1].astype(int)

  mask[np.arange(68), rows, cols] = 1.0

  for i in range(68):
    signal.convolve2d(mask[i], kernel, boundary='symm', mode='same')
  
  return mask

def is_valid_points_in_box(landmarks, box):
  xs = np.array(landmarks[:, 0])
  ys = np.array(landmarks[:, 1])

  if (not (xs >= 0).sum() == xs.size):
    # print("x too small")
    return False
  if (not (xs < 224).sum() == xs.size):
    # print("x too big")
    return False
  if (not (ys >= 0).sum() == xs.size):
    # print("y too small")
    return False
  if (not (ys < 224).sum() == ys.size):
    # print("y too big")
    return False
  return True

"""# Data Loader"""

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

class UNetDataset(Dataset):
  def __init__(self, start_index, end_index, images_name_set, landmarks_set, face_boxes, image_dimensions=224, augment=False):   
    self.image_dimensions = image_dimensions

    self.images_name_set = images_name_set
    self.landmarks_set = landmarks_set
    self.face_boxes = face_boxes
    self.augment = augment

    self.imgs = []
    self.landmarks = []
    self.heatmaps = []

    if augment:
      self.jitter = transforms.ColorJitter(brightness = .5, contrast=.5, saturation=.5, hue=.2)

    for i in range(start_index, end_index):
      landmark = np.copy(self.landmarks_set[i])
      image = skio.imread(self.images_name_set[i])
      if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=-1)
      box = self.face_boxes[i].astype(int)

      # Crop
      image = crop_img(image, box)

      # Resize
      landmark[:, 0], landmark[:, 1] = crop_resize_landmarks(landmark, box, self.image_dimensions, image.shape)
      image = resize_img(image, self.image_dimensions)

      if (not is_valid_points_in_box(landmark, box)):
        continue

      sample = {'image': image.astype('float32'), 'landmarks': landmark}

      self.imgs.append(sample['image'].astype('float32'))
      self.landmarks.append(sample["landmarks"])

    

  def __getitem__(self, i):
    if torch.is_tensor(i):
            i = i.tolist()

    im = self.imgs[i]

    sample = {'image': im, 'landmarks': self.landmarks[i]}

    if self.augment:
      sample = self.augment_transform(sample)
      
    heatmap = create_heatmap(sample["landmarks"])
    
    result = {'image': im, 'landmarks': heatmap}

    # return {'image': sample["image"].astype('float32'), 'heatmap': heatmap}
    return result

  def __len__(self):
    return len(self.imgs)

  def augment_transform(self, sample):
    # Source: https://imgaug.readthedocs.io/en/latest/source/examples_basics.html

    # Set random transfrom params and define transform function
    img, landmarks = sample['image'], sample['landmarks']

    # Multiply Blending
    to_multiply = random.random() + 0.5
    
    # Sometimes function to drop out (tear holes in)
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential([
        iaa.Multiply(to_multiply),
        iaa.LinearContrast((0.75, 1.5)),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05)),
        sometimes(iaa.Crop(percent=(0, 0.1))),
        sometimes(iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout(
                        (0.03, 0.15), size_percent=(0.02, 0.05),
                        per_channel=0.2
                    ),
                ])),
    ])
    # Set keypoints to transform
    keypoints = KeypointsOnImage([Keypoint(x=keypt[0],y=keypt[1]) for keypt in landmarks], shape = img.shape)
    im_aug, landmarks_aug = seq(image=img.astype(np.float32), keypoints=keypoints)
    landmarks = np.array([[keypt.x, keypt.y] for keypt in landmarks_aug.keypoints])
    return {'image': im_aug, 'landmarks': landmarks}

train_UNet_dataset = UNetDataset(0, 6000, img_filenames, landmark_set, face_boxes, augment=True) # imgs 0-5999
print("loaded training set")
print(len(train_UNet_dataset))

validation_UNet_dataset = UNetDataset(6000, 6665, img_filenames, landmark_set, face_boxes, augment=False) # imgs 6000-6665
print("loaded validation set")
print(len(validation_UNet_dataset))

b = validation_UNet_dataset[70]
plt.imshow(b["image"])
plt.show()
plt.imshow(b["heatmap"])
plt.show()

a = train_UNet_dataset[70]
plt.imshow(a["image"])
plt.show()
plt.imshow(a["heatmap"])
plt.show()

BATCH_SIZE = 16
train_UNet_dataloader = DataLoader(train_UNet_dataset, batch_size=BATCH_SIZE)
val_UNet_dataloader = DataLoader(validation_UNet_dataset, batch_size=BATCH_SIZE)

"""## Model"""

import torch
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
model.conv = torch.nn.Conv2d(32, 68, kernel_size=(7,7), padding=(3,3), bias=False)
model = model.to(device, dtype=torch.float)
# Loss function
loss_fn = torch.nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # learning rate=1e-3
model

test_img = torch.tensor(a["image"]).unsqueeze(0).permute((0, 3, 1, 2)).float()
test_img.to(device)
test_output = model(test_img)
# plt.imshow(test_output[0][0].cpu().detach().numpy())
# torch.tensor(a["heatmap"]).unsqueeze(0).unsqueeze(0).shape
test_img.shape
test_output.shape

def train_UNet(dataloader, model, loss_function, optimizer):
    size = len(dataloader.dataset)
    model.train()
    
    total_loss = 0
    
    for i, batch in tqdm(enumerate(dataloader)):
        if (i % 100 == 0):
          print("processed another 100 batches")
        image, heatmap = batch['image'], batch['heatmap']
        image = torch.tensor(image).permute((0, 3, 1, 2)).float()
        heatmap = torch.tensor(heatmap).float()

        print(image.shape)
        print(heatmap.shape)

        image, heatmap = image.to(device), heatmap.to(device)
        
        # Zero your gradients for every batch!
        model.zero_grad()
        # Make predictions for this batch

        output = model(image)
        output = torch.sum(output, dim=1)

        loss = loss_function(output, heatmap)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        del image
        del heatmap
        torch.cuda.empty_cache()
        
    mean_loss = total_loss / (i + 1)
        
    return mean_loss

def validate_UNet(dataloader, model, loss_function, optimizer):
    size = len(dataloader.dataset)
    model.eval()
    
    total_loss = 0
    
    for i, batch in tqdm(enumerate(dataloader)):
        if (i % 10 == 0):
          print("processed another 100 batches")
        image, heatmap = batch['image'], batch['heatmap']
        image = torch.tensor(image).permute((0, 3, 1, 2)).float()
        heatmap = torch.tensor(heatmap).float()

        image, heatmap = image.to(device), heatmap.to(device)
        
        # Zero your gradients for every batch!
        model.zero_grad()
        # Make predictions for this batch

        output = model(image)
        output = torch.sum(output, dim=1)

        loss = loss_function(output, heatmap)  
        total_loss += loss.item()

        del image
        del heatmap
        torch.cuda.empty_cache()
        
    mean_loss = total_loss / (i + 1)
        
    return mean_loss

train_losses = []
val_losses = []
import time
torch.cuda.empty_cache()

NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):
    train_loss = train_UNet(train_UNet_dataloader, model, loss_fn, optimizer)
    print("Epoch Training Loss:", train_loss)
    torch.save(model.state_dict(), '/content/drive/MyDrive/Colab Notebooks/unet_models/model'+time.strftime('%H%M%S')+'.pt')  
    val_loss = validate_UNet(val_UNet_dataloader, model, loss_fn, optimizer)
    print("Epoch Validation Loss:", val_loss)  
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print("finished epoch: ", epoch)

print(train_losses)
print(val_losses)

train_UNet_losses = [0.0009323367858305574,
                     0.0005495530650950968,
                     0.0004129043225354205,
                     0.00034287025399195654,
                     0.0002957495450973511,
                     0.00025953111347431937,
                     0.0002318613382910068,
                     0.0002098458172986284,
                     0.0001924612702568993,
                     0.00017771712275377164,
                     0.00016612694420230884,
                     0.00015641454347254088,
                     0.00014886678161565213,
                     0.0001422819464545076,
                     0.00013670139592917015,
                     0.00013180800674793622,
                     0.0001274980801390484,
                     0.0001235874731404086,
                     0.0001200716205057688,
                     0.000116895027187032]

val_UNet_losses = [0.0007420947409367987,
                   0.0005037070458499892,
                   0.00039931746210814233,
                   0.0003343314317698103,
                   0.00029565268847537007,
                   0.0002638335264193648,
                   0.00023310217788786671,
                   0.00021095971897011623,
                   0.00019316608536644795,
                   0.000178143748988597,
                   0.00016749872711010365,
                   0.00016018743640632325,
                   0.00015361362845093632,
                   0.00014749956227162677,
                   0.00014143740277393677,
                   0.000136756361392881,
                   0.00013211774023061263,
                   0.0001287120702770716,
                   0.00012530421268560791,
                   0.00012252123200423305]


assert(len(train_UNet_losses) == len(val_UNet_losses))

plt.figure(figsize = (10, 7))
plt.plot(range(len(train_UNet_losses)), train_UNet_losses, label = 'training loss')
plt.plot(range(len(val_UNet_losses)), val_UNet_losses, label = 'validation loss')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.title('train and val loss over epochs for facial network')
plt.legend()
plt.savefig("UNet_train_val_loss.jpg")

data_set_path = "/content/drive/MyDrive/Colab Notebooks/ibug_300W_large_face_landmark_dataset"
tree = ET.parse(data_set_path +'/labels_ibug_300W_test_parsed.xml')
root = tree.getroot()
root_dir = '/content/drive/MyDrive/Colab Notebooks/ibug_300W_large_face_landmark_dataset'

face_boxes_test = [] 
img_filenames_test = [] 

for filename in root[2]:
  img_filenames_test.append(os.path.join(root_dir, filename.attrib['file']))
  box = filename[0].attrib
  new_w = int(float(box['width']) * 1.2)
  new_h = int(float(box['height'])* 1.2)
  diff_x = int((new_w - float(box['width']))/2)
  diff_y = int((new_h - float(box['height']))/2)
  face_boxes_test.append([float(box['left'])-diff_x, float(box['top'])-diff_y, new_w, new_h]) 
face_boxes_test = np.array(face_boxes_test).astype('float32')

class TestDatasetColor(Dataset):
  def __init__(self, images_name_set, face_boxes, image_dimensions=224):   
    self.image_dimensions = image_dimensions

    self.images_name_set = images_name_set
    self.face_boxes = face_boxes

  def __getitem__(self, i):
    if torch.is_tensor(i):
            i = i.tolist()

    image = skio.imread(self.images_name_set[i])
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=-1)
    box = self.face_boxes[i].astype(int)

    # Crop
    image = crop_img(image, box)

    # Resize
    image = resize_img(image, self.image_dimensions)

    return image

  def __len__(self):
    return len(self.images_name_set)

NUM_WORKERS = 2
test_color_dataset = TestDatasetColor(img_filenames_test, face_boxes_test)
test_color_loader = DataLoader(test_color_dataset, batch_size=1, num_workers=NUM_WORKERS)

def find_keypoint(a, ksize):
    """
    Finds keypoint in a heatmap
    """
    xsize=a.shape[1]
    ysize=a.shape[0]
    b=np.array([[np.mean(a[y-ksize:y+ksize,x-ksize:x+ksize]) for y in range(ksize,ysize-ksize)]for x in range(ksize,xsize-ksize)])
    

    max_x_centers = []
    max_y_centers = []

    # for i in ind:
    maxcenterx = np.unravel_index(b.argmax(), b.shape)[0]+ksize
    maxcentery = np.unravel_index(b.argmax(), b.shape)[1]+ksize
    max_x_centers = maxcenterx
    max_y_centers = maxcentery

    return [max_x_centers, max_y_centers]

def predict_UNet(dataloader, model):
    heatmaps = []
    model.eval()
    keypoint_set = []
    
    for batch in dataloader:
        image = batch
        image = image.permute((0, 3, 1, 2)).float().to(device)
        
        output = model(image)

        output = output.cpu()
        output = output.detach()

        heatmaps.append(torch.sum(output, dim=1))

        output = output.numpy()

        keypoints = []

        for im in output[0]:
          keypoints.append(find_keypoint(im, 4))

        keypoint_set.append(keypoints)
        del image
        del keypoints
    
    # output_vector = np.stack(outputs, axis=0)
    keypoint_set_vector = np.stack(keypoint_set, axis=0)
    heatmap_vector = np.stack(heatmaps, axis=0)
    return keypoint_set, heatmaps

import torch
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
model.conv = torch.nn.Conv2d(32, 68, kernel_size=(7,7), padding=(3,3), bias=False)
model = model.to(device, dtype=torch.float)
# Loss function
loss_fn = torch.nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # learning rate=1e-3

model.load_state_dict(torch.load("/content/drive/MyDrive/Colab Notebooks/unet_models/model113332.pt"))
model.to(device)

UNet_test_predict = predict_UNet(test_color_loader, model)
torch.save(UNet_test_predict, 'UNet_test_predict.pt')

plt.imshow(UNet_test_predict[69][0])
UNet_test_predict.shape
UNet_test_predict[69][0]

"""Predictions on test images"""

UNet_test_predict_points, UNet_test_predict_heatmaps = predict_UNet(test_color_loader, model)

plt.imshow(UNet_test_predict_heatmaps[69])
plt.savefig("test_predict_heatmap1.jpg")

plt.imshow(UNet_test_predict_heatmaps[72])
plt.savefig("test_predict_heatmap2.jpg")

plt.imshow(UNet_test_predict_heatmaps[80])
plt.savefig("test_predict_heatmap3.jpg")

a, b = get_original_image_w_pts(UNet_test_predict_points, test_dataset, 69, img_filenames_test, face_boxes_test)
show_im_w_pts(a, b)
plt.savefig("part4_test_predict1.jpg")

a, b = get_original_image_w_pts(UNet_test_predict_points, test_dataset, 72, img_filenames_test, face_boxes_test)
show_im_w_pts(a, b)
plt.savefig("part4_test_predict2.jpg")

a, b = get_original_image_w_pts(UNet_test_predict_points, test_dataset, 80, img_filenames_test, face_boxes_test)
show_im_w_pts(a, b)
plt.savefig("part4_test_predict3.jpg")

"""Predictions on personal images"""

UNet_personal_predict_points, UNet_personal_predict_heatmaps = predict_UNet(personal_loader, model)

plt.imshow(UNet_personal_predict_heatmaps[0])
plt.savefig("archit_heatmap.jpg")

plt.imshow(UNet_personal_predict_heatmaps[1])
plt.savefig("david_weeb_heatmap.jpg")

plt.imshow(UNet_test_predict_heatmaps[2])
plt.savefig("facebook_heatmap.jpg")

a, b = get_original_image_w_pts(UNet_personal_predict_points, personal_dataset, 0, personal_img_list_names, personal_boxes)
show_im_w_pts(a, b)
plt.savefig("part4_archit_prediction.jpg")

a, b = get_original_image_w_pts(UNet_personal_predict_points, personal_dataset, 1, personal_img_list_names, personal_boxes)
show_im_w_pts(a, b)
plt.savefig("part4_david_weeb_prediction.jpg")

a, b = get_original_image_w_pts(UNet_personal_predict_points, personal_dataset, 2, personal_img_list_names, personal_boxes)
show_im_w_pts(a, b)
plt.savefig("part4_facebook_prediction.jpg")

"""#Saving to csv"""

def to_csv(keypoints_lst, fname="/content/predictions_unet.csv"):
  ids = np.arange(len(keypoints_lst))
  csv_array = np.zeros((len(keypoints_lst), 2))
  csv_array[:,0] = ids.astype(int)
  csv_array[:,1] = keypoints_lst
  np.savetxt(fname, csv_array, delimiter=',')

test_keypt_predictions_formatted = []

for i, keypoints in enumerate(UNet_test_predict_points):
  keypoints = get_original_pts(UNet_test_predict_points, test_dataset, i, img_filenames_test, face_boxes_test)

  for point in keypoints:
    test_keypt_predictions_formatted.append(point[0])
    test_keypt_predictions_formatted.append(point[1])

print(len(test_keypt_predictions_formatted))
to_csv(test_keypt_predictions_formatted)

import csv
with open('/content/predictions_unet.csv',newline='') as f:
    r = csv.reader(f)
    data = [line for line in r]