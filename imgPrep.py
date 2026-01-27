import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from typing import List


def imgPrepSVM(path: str, img_name: str, hog: cv2.HOGDescriptor, img_size=(64, 64)) -> tuple:
    img_path = os.path.join(path, img_name)
    img_arr = cv2.imread(img_path)

    new_arr = cv2.resize(img_arr, img_size)
                
    gray_arr = cv2.cvtColor(new_arr, cv2.COLOR_BGR2GRAY)
    hog_features = hog.compute(gray_arr)
    res_original = hog_features.flatten()

    flipped_gray = cv2.flip(gray_arr, 1) 
    flipped_features = hog.compute(flipped_gray)
    res_flipped = flipped_features.flatten()

    return res_original, res_flipped


def imgPrepCNN(path: str, img_name: str, img_size=(64, 64)) -> tuple:
    img_path = os.path.join(path, img_name)
    img_arr = cv2.imread(img_path)
    new_arr = cv2.resize(img_arr, img_size)
                
    gray_arr = cv2.cvtColor(new_arr, cv2.COLOR_BGR2GRAY)
                
    img_reshaped = gray_arr.reshape(img_size[0], img_size[1], 1)
    res_original = img_reshaped / 255.0                

    res_flipped = cv2.flip(res_original, 1).reshape(img_size[0], img_size[1], 1)

    return res_original, res_flipped


def procImg(folder_path: str, categories: List[str], procType: str, img_size=(64, 64)) -> tuple:
    data = []
    labels = []
    hog = cv2.HOGDescriptor(_winSize=img_size,
                        _blockSize=(16, 16),
                        _blockStride=(8, 8),
                        _cellSize=(8, 8),
                        _nbins=9)

    for category in categories:
        path = os.path.join(folder_path, category)
        class_num = categories.index(category)

        for img_name in os.listdir(path):
            try:
                if procType == "svm":
                    res_original, res_flipped = imgPrepSVM(path, img_name, hog, img_size)                    
                elif procType == "cnn":
                    res_original, res_flipped = imgPrepCNN(path, img_name, img_size)

                data.append(res_original)
                labels.append(class_num)

                data.append(res_flipped)
                labels.append(class_num)
            except Exception as e:
                print(f"Error processing image {img_name}: {e}")
                continue

    X = np.array(data)
    y = np.array(labels)
    
    X, y = shuffle(X, y, random_state=42)
    
    return train_test_split(X, y, test_size=0.2, random_state=0)