"""Conversion formulas for labeling.
centre_x = [(xmin+xmax)/2]/width of image
centre_y = [(ymin+ymax)/2]/height of image
width = (xmax-xmin)/width of image
height = (ymax-ymin)/height of image"""

import os
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split

csvpath= r'D:\python\ID CARD DETECTION\ID_RESTARTED\data\annotations.csv'
data = pd.read_csv(csvpath)

#labelling 
data['centre_x'] = ((data['xmin'] + data['xmax']) / 2) / data['width']
data['centre_y'] = ((data['ymin'] + data['ymax']) / 2) / data['height']
data['label_width'] = (data['xmax'] - data['xmin']) / data['width']
data['label_height'] = (data['ymax'] - data['ymin']) / data['height']
data = data[['filename', 'class_name', 'centre_x', 'centre_y', 'label_width', 'label_height']]

#splitting into train and test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_data.to_csv(r'D:\python\ID CARD DETECTION\ID_RESTARTED\data\train_labels.csv', index=False)
test_data.to_csv(r'D:\python\ID CARD DETECTION\ID_RESTARTED\data\test_labels.csv', index=False)

print("Training data:")
print(train_data.head())
print("\nValidation data:")
print(test_data.head())

print("\nTraining data shape:", train_data.shape)
print("Validation data shape:", test_data.shape)
print("\nTraining data class distribution:")
print(train_data['class_name'].value_counts())
print("\nValidation data class distribution:")
print(test_data['class_name'].value_counts())

#label encoding- assigning a number to each class name is not needed as we only have one class name
