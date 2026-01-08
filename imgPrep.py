import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from typing import List
# pip install -r requirements.txt

def imgPrepSVM(folder_path, categories: List[str], img_size=(64, 64)) -> tuple:
    data = []
    labels = []
    # ['cats', 'dogs'] = categories
    for category in categories:
        path = os.path.join(folder_path, category)
        class_num = categories.index(category)

        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img_arr = cv2.imread(img_path)

                new_arr = cv2.resize(img_arr, img_size)
                hog = cv2.HOGDescriptor(_winSize=(64, 64),
                        _blockSize=(16, 16),
                        _blockStride=(8, 8),
                        _cellSize=(8, 8),
                        _nbins=9)

                # original image
                gray_arr = cv2.cvtColor(new_arr, cv2.COLOR_BGR2GRAY)
                hog_features = hog.compute(gray_arr)
                data.append(hog_features.flatten())
                labels.append(class_num)

                # flipped image
                flipped_gray = cv2.flip(gray_arr, 1) 
                flipped_features = hog.compute(flipped_gray)
                data.append(flipped_features.flatten())
                labels.append(class_num)

            except Exception as e:
                pass

    X = np.array(data)
    y = np.array(labels)
    
    X, y = shuffle(X, y, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test


def imgPrepCNN(folder_path, categories: List[str], img_size=(64, 64)) -> tuple:
    data = []
    labels = []
    # ['cats', 'dogs'] = categories
    for category in categories:
        path = os.path.join(folder_path, category)
        class_num = categories.index(category)

        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img_arr = cv2.imread(img_path)
                new_arr = cv2.resize(img_arr, img_size)
                
                gray_arr = cv2.cvtColor(new_arr, cv2.COLOR_BGR2GRAY)
                
                img_reshaped = gray_arr.reshape(img_size[0], img_size[1], 1)
                img_final = img_reshaped / 255.0                
                data.append(img_final)
                labels.append(class_num)

                # Data Augmentation (Flip)
                flipped = cv2.flip(img_final, 1).reshape(img_size[0], img_size[1], 1)
                data.append(flipped)
                labels.append(class_num)

            except Exception as e:
                pass

    X = np.array(data)
    y = np.array(labels)
    
    X, y = shuffle(X, y, random_state=42)
    
    return train_test_split(X, y, test_size=0.2, random_state=0)