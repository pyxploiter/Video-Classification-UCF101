import os
import argparse

import torch
from torch.nn import NLLLoss
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from model import prepare_model
from dataloader import load_data

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--data", dest="data_dir", help="Path to training data videos")
parser.add_argument("-l","--labels", dest="label_dir", help="path to action recognition labels")
parser.add_argument("-o","--output", dest="output_dir", help="Output path for saving model.")
parser.add_argument("-ne","--num_epochs", type=int, dest="num_epochs", help="Number of epochs.", default=1)
parser.add_argument("-bs","--batch_size", type=int, dest="batch_size", help="Batch size of data", default=8)
parser.add_argument("-fc","--frames_per_clip", type=int, dest="frames_per_clip", help="framers per video clip", default=5)
parser.add_argument("-sc","--step_between_clips", type=int, dest="step_between_clips", help="steps between video clips", default=1)

configs = parser.parse_args()

ucf_data_dir = configs.data_dir
ucf_label_dir = configs.label_dir
OUTPUT_PATH = configs.output_dir

frames_per_clip = configs.frames_per_clip
step_between_clips = configs.step_between_clips
batch_size = configs.batch_size

EPOCHS = configs.num_epochs

# create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

# load the data into train test splits
train_loader, test_loader = load_data(data_dir=ucf_data_dir,
                                    label_dir=ucf_label_dir,
                                    batch_size=batch_size,
                                    frames_per_clip=frames_per_clip,
                                    step_between_clips=step_between_clips
                                )

num_classes = len(train_loader.dataset.classes)
# prepare the resnet 2+1 model
model = prepare_model(num_classes, pretrained=True)

# create the tensorboard summary writer
writer = SummaryWriter(comment="-r2plus1d_18")
# check if cuda available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define loss function
criterion = NLLLoss()

model_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(model_params, lr=0.01)

model.to(device)

steps = 0
max_acc = -100
test_acc, test_losses = [], []

model.train()
print("Starting the training...")
for epoch in range(EPOCHS):
    train_accuracy = []
    train_loss = []
    for batch_no, (inputs, labels) in enumerate(train_loader):
        # print("Epoch "+str(epoch+1)+" - Batch "+str(batch_no+1)+"/"+str(len(train_loader)))
        # reshape input tensor
        inputs = inputs.permute(0, 2, 1, 3, 4)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        # perform forward pass
        logps = model.forward(inputs)
        # calculate loss
        loss = criterion(logps, labels)
        # backpropagate the loss
        loss.backward()
        # update weights
        optimizer.step()
        train_loss.append(loss.item())

        # calculating training accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim = 1)
        equals = top_class == labels.view(*top_class.shape)
        train_accuracy.append(torch.mean(equals.type(torch.FloatTensor)).item())

        # update tensorboard after every 5 batches
        print_every = 5
        if ((batch_no+1) % print_every == 0):
            print(f"Batch {batch_no+1}/{len(train_loader)}.. "
                  f"Step {steps+1}.. "
                    f"Train loss: {sum(train_loss)/len(train_loss):.3f}.. "
                        f"Train accuracy: {sum(train_accuracy)/len(train_accuracy):.3f}.. ")

            writer.add_scalar('train/loss: ', sum(train_loss)/len(train_loss), steps+1)
            writer.add_scalar('train/accuracy: ', sum(train_accuracy)/len(train_accuracy), steps+1)

            train_accuracy = []
            train_loss = []
          
        # perform validation after every half epoch
        valid_step = len(train_loader)/2
        if (steps+1) % valid_step == 0:
            running_test_loss = 0
            running_test_accuracy = 0

            model.eval()
            with torch.no_grad():
                print("Validating...")
                for tb_no, (inputs, labels) in enumerate(test_loader):
                    inputs = inputs.permute(0, 2, 1, 3, 4)
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    t_loss = batch_loss.item()
                    running_test_loss += t_loss

                    # calculating test accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    t_acc = torch.mean(equals.type(torch.FloatTensor)).item()
                    running_test_accuracy += t_acc

                    print("[Validation] Epoch "
                            +str(epoch+1)
                            +" - Batch "
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

            # save the best model
            if running_test_accuracy > max_acc:
                max_acc = running_test_accuracy
                best_model = model
                torch.save(best_model, os.path.join(OUTPUT_PATH,"best_model.pth"))
              
            writer.add_scalar('test/loss: ', running_test_loss/len(test_loader), steps)
            writer.add_scalar('test/accuracy: ', running_test_accuracy/len(test_loader), steps)

            print(f"Epoch {epoch+1}/{EPOCHS}.. "
                f"Test loss: {running_test_loss/len(test_loader):.3f}.. "
                f"Test accuracy: {running_test_accuracy/len(test_loader):.3f}.. ")

            f = open(os.path.join(OUTPUT_PATH,"log.txt"), "a")
            f.write(f"Epoch {epoch+1}/{EPOCHS}.. "
                f"Test loss: {running_test_loss/len(test_loader):.3f}.. "
                f"Test accuracy: {running_test_accuracy/len(test_loader):.3f}.. \n")

            f.close()
            model.train()

        steps += 1

torch.save(model, os.path.join(OUTPUT_PATH,"last_model.pth"))        
print("Training is completed.")