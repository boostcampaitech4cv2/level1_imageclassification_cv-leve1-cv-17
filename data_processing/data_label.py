import os
import pandas as pd
import numpy as np


'''
# -- 여기서 train.csv를 읽어서, 각 이미지의 경로 및 Label 번호를 만들어주는 코드입니다. -- #
    
    function: get_img_dir(data_dir, train=True)
    return: image_dir_path, image_label
    
    function: get_image_path(image_dir_path, num2class)
    return: image_path
    
    function: mask_label(image_path, img_label)
    return : label
'''

num2class = ['incorrect_mask', 'mask1', 'mask2', 'mask3',
             'mask4', 'mask5', 'normal']

def mask_label(image_path, img_label):
    '''
    Labeling from 0 to 17
    0 : wearing mask, male, young(<30)
    1: wearing mask, male, middle(30~60)
    2: wearing mask, male, old(>60)
    3: wearing mask, female, young(<30)
    4: wearing mask, female, middle(30~60)
    5: wearing mask, female, old(>60)
    ...
    
    Args:
        img_path: image path to get info mask, incorrect and no mask.
        img_label: image label from mask_label to get info age and gender
        num2class: ['incorrect_mask', 'mask1', 'mask2', 'mask3',
                    'mask4', 'mask5', 'normal']
                    
    Returns:
        label: label from 0 to 17
    '''
    mask = 0
    gender = 0
    age = 0
    
    img_info = img_label.split('_')
    # print(img_info)
    
    # mask check
    mask_path = image_path.split('/')[-1]
    # print((mask_path))
    if 'normal' in mask_path:
        mask = 2
    elif 'incorrect' in mask_path:
        mask = 1
    else:
        mask = 0
    
    # gender check
    if img_info[1] == 'male':
        gender = 0
    else:
        gender = 1
    
    # age check
    if int(img_info[3]) < 30:
        age = 0
    elif 30 <= int(img_info[3]) < 60:
        age = 1
    else:
        age = 2
        
    return 6*mask + 3*gender + age

def get_img_dir(data_dir, train=True):
    '''
    get image dir path from (data_dir=/opt/ml/project)/input/data/train/train.csv
    Args:
        data_dir: /opt/ml/project/
        train: True
    
    Returns:
        image_dir_path: image dir path from train['path']
        image_label: if train, return image label from train['path']
    '''
    if train: # /input/data/train/train.csv
        image_dir_path = []
        image_label = []
        df = pd.read_csv(os.path.join(data_dir, "input", "data", "train", "train.csv"))
        for i in range(len(df)):
            image_dir_path.append(os.path.join(data_dir, "input", "data/" "train", "images", df.iloc[i, -1]))
            image_label.append(df.iloc[i, -1])
        return image_dir_path, image_label

    else: # input/data/eval/images
        image_dir_path = []
        for i in os.listdir(os.path.join(data_dir, 'input', 'data', 'eval', 'images')):
            image_dir_path.append(os.path.join(data_dir, 'input', 'data', 'eval', 'images', i))
        return image_dir_path
 


def get_image_path(image_dir_path, num2class):
    '''
    get image path from input/data/train/images/df['path']/num2class
    
    args:
        image_dir_path: image path from get_img_dir
                        see get_img_dir(data_dir, train=True)
        num2class: ['incorrect_mask', 'mask1', 'mask2', 'mask3',
                    'mask4', 'mask5', 'normal']
                    
    returns:
        image_path: image path, input/data/train/images/df['path']/num2class
    '''
    image_path = []
    for i in num2class:
         image_path.append(image_dir_path + '/' + i)
    return image_path



if __name__ == '__main__':
    label_df = pd.DataFrame(columns=['image_path', 'incorrect_mask', 'mask', 'normal'])
    
    image_dir_paths, image_labels = get_img_dir(os.getcwd())
    # print(image_dir_paths[0])
    image_path = get_image_path(image_dir_paths[0], num2class)
    print(len(image_path))
    print(image_labels[100])
    print(mask_label(image_path[0], image_labels[100]))
    # print(image_dir_paths)
    
    label_df['image_path'] = image_dir_paths
    if label_df['mask'].any() == False:
        for i in range(len(image_dir_paths)):
            image_path = get_image_path(image_dir_paths[i], num2class)
            for j in range(len(image_path)):
                label_df.loc[i, num2class[j]] = int(mask_label(image_path[j], image_labels[i]))
        
        label_df['mask'] = label_df['mask1']
        # print(label_df.loc[241].mask1)
        
        for i in range(1, len(image_path)+2):
            label_df.iloc[:, i] = label_df.iloc[:,i].astype(int)
        label_df.to_csv('/opt/ml/project/input/data/train/label.csv', index=False)
    
    print(label_df['mask'].dtypes)
    # print(label_df.iloc[:,0])
    
    