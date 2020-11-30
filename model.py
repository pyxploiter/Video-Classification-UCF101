import torch.nn as nn

from torchvision import models


def prepare_model(num_classes, pretrained=True):
    # loading pretrained model
    model = models.video.r2plus1d_18(pretrained=pretrained)
    # freeze parameters so we don't backprop through them
    for name, param in model.named_parameters():
        if name == "layer4.1.conv1.0.0.weight":
            break
        param.requires_grad = False        
    
    # modifying the classifier
    model.fc = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes),
                    nn.LogSoftmax(dim=1)
                )

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

    return model