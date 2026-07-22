# Save images and labels in separate train and test folders
import os
import pandas as pd
from shutil import copy

# Paths
train_csv = r'D:\python\ID CARD DETECTION\ID_RESTARTED\data\train_labels.csv'
test_csv = r'D:\python\ID CARD DETECTION\ID_RESTARTED\data\test_labels.csv'
images_folder = r'D:\python\ID CARD DETECTION\ID_RESTARTED\data\images'
train_folder = r'D:\python\ID CARD DETECTION\ID_RESTARTED\data\dataset\train'
test_folder = r'D:\python\ID CARD DETECTION\ID_RESTARTED\data\dataset\val'

# Create train and test folders if they don't exist
os.makedirs(os.path.join(train_folder, 'images'), exist_ok=True)
os.makedirs(os.path.join(train_folder, 'labels'), exist_ok=True)
os.makedirs(os.path.join(test_folder, 'images'), exist_ok=True)
os.makedirs(os.path.join(test_folder, 'labels'), exist_ok=True)

# Function to save images and labels
def save_files(csv_path, dest_folder):
    # Read the CSV file
    data = pd.read_csv(csv_path)
    
    for _, row in data.iterrows():
        image_name = row['filename']
        txt_name = os.path.splitext(image_name)[0] + '.txt'  # TXT file name
        
        # Copy image
        src_image_path = os.path.join(images_folder, image_name)
        dest_image_path = os.path.join(dest_folder, 'images', image_name)
        if os.path.exists(src_image_path):
            copy(src_image_path, dest_image_path)
        
        # Write label to TXT file
        dest_txt_path = os.path.join(dest_folder, 'labels', txt_name)
        with open(dest_txt_path, 'w') as f:
            # YOLO format: class_id centre_x centre_y label_width label_height
            class_id = 0  # Assuming a single class (id card)
            f.write(f"{class_id} {row['centre_x']} {row['centre_y']} {row['label_width']} {row['label_height']}\n")

# Save train files
save_files(train_csv, train_folder)

# Save test files
save_files(test_csv, test_folder)

print("Images and labels have been saved to train and test folders.")

