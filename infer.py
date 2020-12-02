import os
import argparse

import torch
from torchvision import transforms
from torchvision.datasets.video_utils import VideoClips

def get_transforms():
    tfs = transforms.Compose([
        # scale in [0, 1] of type float
        transforms.Lambda(lambda x: x / 255.),
        # reshape into (T, C, H, W) for easier convolutions
        transforms.Lambda(lambda x: x.permute(3, 0, 1, 2)),
        # rescale to the most common size
        transforms.Lambda(lambda x: torch.nn.functional.interpolate(x, (240, 320))),
    ])

    return tfs

def get_classes(label_dir):
    with open(os.path.join(label_dir, "classInd.txt"), "r") as f:
        classes = [x.split(" ")[1].replace("\n","") for x in f.readlines()]
    return classes

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--video_file", dest="video_file", help="path to video file")
    parser.add_argument("-m","--saved_model", dest="saved_model", help="path to saved model.")
    parser.add_argument("-l","--label_dir", dest="label_dir", help="path to action recognition labels directory")
    parser.add_argument("-fc","--frames_per_clip", type=int, dest="frames_per_clip", help="framers per video clip", default=5)
    parser.add_argument("-sc","--step_between_clips", type=int, dest="step_between_clips", help="steps between video clips", default=2)

    configs = parser.parse_args()

    video_file = configs.video_file
    MODEL_PATH = configs.saved_model
    LABEL_DIR = configs.label_dir

    frames_per_clip = configs.frames_per_clip
    step_between_clips = configs.step_between_clips

    classes = get_classes(LABEL_DIR)

    # check if cuda available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_clips = VideoClips(
                        [video_file],
                        frames_per_clip,
                        step_between_clips,
                    )

    total_clips = video_clips.num_clips()

    # load model
    model = torch.load(MODEL_PATH, map_location=device)
    model.to(device)

    model.eval()
    with torch.no_grad():
        preds, outputs = [],[]
        # load clips from videos and classify them
        for clip_no in range(max(total_clips//4,1)):
            video, audio, info, idx = video_clips.get_clip(clip_no)
            video = get_transforms()(video)
            inputs = video.unsqueeze(0).to(device)
            logps = model.forward(inputs)
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            outputs.append([classes[top_class],top_p.item()])
            preds.append(classes[top_class])

        final_pred = max(preds, key=preds.count)
        confs = [conf for pred, conf in outputs if pred == final_pred]
        print("predicted class:", final_pred)
        print("confidence:", sum(confs)/len(confs))