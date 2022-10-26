import pandas as pd
import glob
from collections import defaultdict
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image-dir-path', help='assign image data dir path')
parser.add_argument('-c', '--csv-path', help='assign csv data path')
parser.add_argument('-s', '--save-path', help='assign csv save path')
args = parser.parse_args()

class DataAddlabel():
    def __init__(self, image_dir_path : str, csv_path : str, save_path : str):
        self.image_dir_path = image_dir_path
        self.csv_path = csv_path
        self.save_path = save_path

        orig_data = pd.read_csv(self.csv_path)
        self.orig_columns = orig_data.columns.tolist()
        self.orig_data = orig_data.values.tolist()

        self.split_age_gender = defaultdict(list)
        self.labeled_datas = []
            
    def data_split_age_gender(self):
        for data in self.orig_data:
            age = int(data[3])
            gender = data[1]
            age_type = ''
            label_sqeunce = 0
            
            if age < 30:
                age_type = 'young'
            elif 30 <= age < 60:
                age_type = 'middle'
                label_sqeunce = 1
            else:
                age_type = 'old'
                label_sqeunce = 2
                
            self.split_age_gender[f'{label_sqeunce}_{age_type}_{gender}'].append(data)

    def update_labels_dict(self, data_list : list, labels : list):
        add_label_data = []
        label_list = []

        for data in data_list:
            image_paths = glob.glob(f'{self.image_dir_path}/{data[-1]}/*.jpg')

            for image_path in image_paths:
                status = image_path.split('/')[-1].split('.')[0]
                status = re.sub('[0-9]', '', status)

                if status == "mask":
                    label = labels[0]
                elif status == "incorrect_mask":
                    label = labels[1]
                else:
                    label = labels[2]

                if label not in label_list:
                    label_list.append(label)

            label_list.sort()
            add_label_data = data + label_list
            self.labeled_datas.append(add_label_data)

    def data_add_label(self):  
        keys = list(self.split_age_gender.keys())
        keys.sort()

        male_labels, female_labels = [0, 6, 12], [3, 9, 15]

        for idx, key in enumerate(keys):
            if idx % 2 == 0:
                self.update_labels_dict(self.split_age_gender[key], female_labels)
            else:
                self.update_labels_dict(self.split_age_gender[key], male_labels)
                male_labels = [label + 1 for label in male_labels]
                female_labels = [label + 1 for label in female_labels]

    def save_labeld_data(self):
        self.labeled_datas = sorted(self.labeled_datas, key=lambda x: x[0])
        labeld_data_df = pd.DataFrame(self.labeled_datas, columns=self.orig_columns+['mask', 'incorrect', 'normal'])
        labeld_data_df.to_csv(self.save_path, index=False)


if __name__ == "__main__":
    image_dir_path = '/opt/ml/input/data/train/images' if args.image_dir_path == None else args.image_dir_path
    train_csv_path = '/opt/ml/input/data/train/train.csv' if args.csv_path == None else args.csv_path
    save_path = '/opt/ml/input/data/train/train_added_label.csv' if args.save_path == None else args.save_path

    dataAddlabel = DataAddlabel(image_dir_path, train_csv_path, save_path)
    
    dataAddlabel.data_split_age_gender()
    dataAddlabel.data_add_label()
    dataAddlabel.save_labeld_data()

    print('Done.')