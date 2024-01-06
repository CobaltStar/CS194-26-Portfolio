# -*- coding: utf-8 -*-

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

data_set_path = "/content/drive/MyDrive/Colab Notebooks/ibug_300W_large_face_landmark_dataset"

tree = ET.parse(data_set_path + '/labels_ibug_300W_train.xml')  
root = tree.getroot()
root_dir = '/content/drive/MyDrive/Colab Notebooks/ibug_300W_large_face_landmark_dataset'


"""Part 3"""

face_boxes = [] 
img_filenames = []
landmark_set = []

for filename in root[2]:
  box = filename[0].attrib
  img_filenames.append(os.path.join(root_dir, filename.attrib['file']))

  box = filename[0].attrib
  # add corners with some extra padding
  new_w = int(float(box['width']) * 1.2)
  new_h = int(float(box['height'])* 1.2)
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
  im = resize(im, (resize_dim, resize_dim))
  return im

def crop_resize_landmarks(landmarks, box, resize_dim, imshape):
  xs = landmarks[:, 0] - box[0]
  ys = landmarks[:, 1] - box[1]

  # resize points
  scalar = resize_dim * 1.0 / imshape[0]
  xs = xs * scalar
  ys = ys * scalar
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
# model = model.float()
model = model.to(device, dtype=torch.float)

# Loss function
loss_fn = torch.nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # learning rate=1e-3

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




"""END OF PART 3"""






"""# Part 4"""

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

def create_heatmap(landmarks, imgsize=224, kernel_size=7, sigma=1):
  # heatmaps = []

  A = np.zeros((imgsize, imgsize))

  for kp in landmarks:
    if (0 < kp[1] < imgsize and 0 < kp[0] < imgsize):
      A[int(kp[1]), int(kp[0])] = 1    # random
      # heatmaps.append(A)

  A = apply_gaussian(A, kernel_size, sigma)
  
  # F = np.zeros((imgsize, imgsize))

  # for heatmap in heatmaps:
  #   F = F + heatmap

  return A


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


      sample = {'image': image.astype('float32'), 'landmarks': landmark}
      if self.augment:
        sample = self.augment_transform(sample)

      heatmap = create_heatmap(sample["landmarks"])
      self.imgs.append(sample['image'].astype('float32'))
      self.heatmaps.append(heatmap)
    

    

  def __getitem__(self, i):
    if torch.is_tensor(i):
            i = i.tolist()


    heatmap = self.heatmaps[i]
    im = self.imgs[i]
    sample = {'image': im, 'heatmap': heatmap}

    # return {'image': sample["image"].astype('float32'), 'heatmap': heatmap}
    return sample

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

BATCH_SIZE = 64
train_UNet_dataloader = DataLoader(train_UNet_dataset, batch_size=BATCH_SIZE)
val_UNet_dataloader = DataLoader(validation_UNet_dataset, batch_size=BATCH_SIZE)

"""## Model"""

import torch
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(2,2), bias=False)
model.fc = torch.nn.Linear(128, 68)
model = model.to(device, dtype=torch.float)
# Loss function
loss_fn = torch.nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # learning rate=1e-3
model

test_img = torch.tensor(np.transpose(a["image"])).unsqueeze(0).float().to(device)
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
        heatmap = torch.tensor(heatmap).float().unsqueeze(0).unsqueeze(0)

        image, heatmap = image.to(device), heatmap.to(device)
        
        # Zero your gradients for every batch!
        model.zero_grad()
        # Make predictions for this batch

        output = model(image)

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
        if (i % 100 == 0):
          print("processed another 100 batches")
        image, heatmap = batch['image'], batch['heatmap']
        image = torch.tensor(image).permute((0, 3, 1, 2)).float().to(device)
        heatmap = torch.tensor(heatmap).float().unsqueeze(0).unsqueeze(0)

        image, heatmap = image.to(device), heatmap.to(device)
        
        # Zero your gradients for every batch!
        model.zero_grad()
        # Make predictions for this batch

        output = model(image)
        loss = loss_function(output, heatmap).float()   
        total_loss += loss.item()

        del image
        del heatmap
        torch.cuda.empty_cache()
        
    mean_loss = total_loss / (i + 1)
        
    return mean_loss

train_losses = []
val_losses = []
import time

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

train_UNet_losses =  [0.0006934505876647853,
                      9.616404473645355e-05,
                      9.584655863557388e-05,
                      9.5744622933822e-05,
                      9.569654441215693e-05,
                      9.567026938113602e-05,
                      9.565271938435159e-05,
                      9.56382636820521e-05,
                      9.562456398271024e-05]

val_UNet_losses = [0.00010407903490969065,
                   0.00010335411207051948,
                   0.00010319770411694084,
                   0.00010314236384477805,
                   0.00010311679480682042,
                   0.00010309987820536745,
                   0.00010308498365868053,
                   0.00010306985495844856,
                   0.00010305351911070333]

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

def predict_UNet(dataloader, model):
    outputs = []
    model.eval()
    
    for batch in dataloader:
        image = batch
        image = image.permute((0, 3, 1, 2)).float().to(device)
        
        output = model(image)

        output = output.cpu()
        outputs.append(output.detach().numpy()[0][0])

        del image
    
    output_vector = np.stack(outputs, axis=0)
    return output_vector

import torch
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(2,2), bias=False)
model.fc = torch.nn.Linear(128, 68)
model = model.to(device, dtype=torch.float)
# Loss function
loss_fn = torch.nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # learning rate=1e-3

model.load_state_dict(torch.load("/content/drive/MyDrive/Colab Notebooks/unet_models/model094652.pt"))
model.to(device)

UNet_test_predict = predict_UNet(test_color_loader, model)
torch.save(UNet_test_predict, 'UNet_test_predict.pt')

plt.imshow(UNet_test_predict[1])


def find_keypoint(img, ksize, n):
    xsize=a.shape[1]
    ysize=a.shape[0]
    b=np.array([[np.mean(a[y-ksize:y+ksize,x-ksize:x+ksize]) for y in range(ksize,ysize-ksize)]for x in range(ksize,xsize-ksize)])
    
    flat = b.flatten()
    ind = np.argpartition(flat, -n)[-n:]
    ind[np.argsort(flat[ind])]

    max_x_centers = []
    max_y_centers = []

    for i in ind:
      maxcenterx = np.unravel_index(ind, b.shape)[0]+ksize
      maxcentery = np.unravel_index(ind, b.shape)[1]+ksize
      max_x_centers = maxcenterx
      max_y_centers = maxcentery

    return max_x_centers, max_y_centers

  
x_centers, y_centers = find_keypoint(hm, 40, 68)
plt.imshow(hm)
plt.scatter(x_centers, y_centers)

