import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from typing import List


def imgPrepSVM(folder_path: str, categories: List[str], img_size=(64, 64)) -> tuple:
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
                img_path = os.path.join(path, img_name)
                img_arr = cv2.imread(img_path)

                if img_arr is None:
                    continue

                new_arr = cv2.resize(img_arr, img_size)
                
                gray_arr = cv2.cvtColor(new_arr, cv2.COLOR_BGR2GRAY)
                hog_features = hog.compute(gray_arr)
                res_original = hog_features.flatten()

                data.append(res_original)
                labels.append(class_num)
            except Exception as e:
                print(f"Error processing image {img_name}: {e}")
                continue

    X = np.array(data)
    y = np.array(labels)
    
    X, y = shuffle(X, y, random_state=42)

    return train_test_split(X, y, test_size=0.2, random_state=0)