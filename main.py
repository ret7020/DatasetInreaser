from dataclasses import field
import albumentations as A
import cv2
import os
from pathlib import Path



#Config
dataset_path = "/home/step/Documents/DatasetInreaser/examples" # Full path
Path(os.path.join(dataset_path, "inreased")).mkdir(parents=True, exist_ok=True) # Create dir for transformed images
transformations = [0, 1]
'''
0 - RandomBrightnessContrast 
1 - Clahe
2 - RandomRain
3 - RandomBrightness
4 - 
'''
#Init alumentation transformer
trns = []
for method in transformations:
    if method == 0:
        trns.append(A.RandomBrightnessContrast(p=0.2))
    elif method == 1:
        trns.append(A.CLAHE(clip_limit=5))
    elif method == 2:
        trns.append(A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1))
    elif method == 3:
        trns.append(A.RandomBrightness(limit=0.3))
    elif method == 4:
        trns.append(A.ISONoise(intensity=0.1))
transform = A.Compose(trns)



#Get dataset images
files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
print(f"Found {len(files)} files in dataset. We will generate {len(files) * len(transformations)} extra images")
for curr_image in files:
    image = cv2.imread(os.path.join(dataset_path, curr_image))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    print(transformed_image.shape)
    cv2.imwrite(os.path.join(os.path.join(dataset_path, "inreased"), curr_image), transformed_image)
    
    
        