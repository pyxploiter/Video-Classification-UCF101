import os
import argparse

import torch
from torch.nn import NLLLoss
from torch import optim

from dataloader import load_data

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--data", dest="data_dir", help="path to training data videos")
parser.add_argument("-l","--labels", dest="label_dir", help="path to action recognition labels")
parser.add_argument("-i","--saved_model", dest="saved_model", help="path to saved model.")
parser.add_argument("-bs","--batch_size", type=int, dest="batch_size", help="Batch size of data", default=8)
parser.add_argument("-fc","--frames_per_clip", type=int, dest="frames_per_clip", help="framers per video clip", default=5)
parser.add_argument("-sc","--step_between_clips", type=int, dest="step_between_clips", help="steps between video clips", default=1)

configs = parser.parse_args()

ucf_data_dir = configs.data_dir
ucf_label_dir = configs.label_dir
MODEL_PATH = configs.saved_model

frames_per_clip = configs.frames_per_clip
step_between_clips = configs.step_between_clips
batch_size = configs.batch_size

# check if cuda available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the data into train test splits
test_loader = load_data(data_dir=ucf_data_dir,
                        label_dir=ucf_label_dir,
                        batch_size=batch_size,
                        frames_per_clip=frames_per_clip,
                        step_between_clips=step_between_clips,
                        test_only=True
                    )

# load model
model = torch.load(MODEL_PATH, map_location=device)
model.to(device)

# define loss function
criterion = NLLLoss()

steps = 0
max_acc = -100
test_acc, test_losses = [], []

running_test_accuracy = 0
running_test_loss = 0

model.eval()
with torch.no_grad():
    print("testing...")
    for tb_no, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.permute(0, 2, 1, 3, 4)
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)
        t_loss = batch_loss.item()
        running_test_loss += t_loss

        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        t_acc = torch.mean(equals.type(torch.FloatTensor)).item()
        running_test_accuracy += t_acc

        print("[Testing] "
                +str(tb_no+1)
                +"/"
                +str(len(test_loader))
                +" - Test loss: "
                +str(round(t_loss,4))
                +" - Test accuracy:"
                +str(round(t_acc,4))
            )    
    
test_losses.append(running_test_loss/len(test_loader))
test_acc.append(running_test_accuracy/len(test_loader))

print(f".. "
    f"Average Test loss: {running_test_loss/len(test_loader):.3f}.. "
    f"Average Test accuracy: {running_test_accuracy/len(test_loader):.3f}.. ")

        
