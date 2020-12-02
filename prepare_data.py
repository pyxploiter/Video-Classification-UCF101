import os
import argparse

def prepare_data(data_path, label_path):
    
    with open(os.path.join(label_path, "classInd.txt"), "r") as f:
        classes = [x.split(" ")[1].replace("\n","") for x in f.readlines()]

    for video_file_id in os.listdir(data_path):
        class_id = video_file_id.split("_")[1]
        if class_id in classes:
            os.makedirs(os.path.join("filtered_data", class_id), exist_ok=True)
            os.system("cp '"+os.path.join(data_path, video_file_id)+"' '"+os.path.join("filtered_data", class_id)+"'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", dest="data_dir", help="Path to data videos directory", required=True)
    parser.add_argument("-l","--label_dir", dest="label_dir", help="path to action recognition labels directory", required=True)

    configs = parser.parse_args()
    print(configs)
    # prepare data for training
    prepare_data(
        data_path=configs.data_dir,
        label_path=configs.label_dir
    )