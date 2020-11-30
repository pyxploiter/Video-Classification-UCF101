## Video classification using Resnet (2+1)D on UCF101 Dataset

### Installations

* Python 3.8
	- numpy==1.19.2
	- tensorboard==2.4.0
	- torch==1.7.0+cu92
	- torchvision==0.8.1+cu92


or you can install above packages using pip:

`pip install -r requirements`


### Data

UCF101 Action Recognition dataset is used.

[Click here to download the dataset](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar)

[Click here to download the labels](https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip)

### Pretrained Model

[Click here to download the pretrained model](https://drive.google.com/file/d/10mEgt-jXg51eF39di2zgGSvQp2CKSCYS/view?usp=sharing)

Model is trained on following 15 classes from UCF101: 
['ApplyEyeMakeup', 'BabyCrawling', 'CleanAndJerk', 'Diving', 'Fencing', 'GolfSwing', 'Haircut', 'IceDancing', 'JavelinThrow', 'Kayaking', 'LongJump', 'MilitaryParade', 'Nunchucks', 'ParallelBars', 'Rafting']


### Training

```
usage: train.py [-h] [-d DATA_DIR] [-l LABEL_DIR] [-o OUTPUT_DIR]
                [-ne NUM_EPOCHS] [-bs BATCH_SIZE] [-fc FRAMES_PER_CLIP]
                [-sc STEP_BETWEEN_CLIPS]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_DIR, --data DATA_DIR
                        Path to training data videos
  -l LABEL_DIR, --labels LABEL_DIR
                        path to action recognition labels
  -o OUTPUT_DIR, --output OUTPUT_DIR
                        Output path for saving model.
  -ne NUM_EPOCHS, --num_epochs NUM_EPOCHS
                        Number of epochs.
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size of data
  -fc FRAMES_PER_CLIP, --frames_per_clip FRAMES_PER_CLIP
                        framers per video clip
  -sc STEP_BETWEEN_CLIPS, --step_between_clips STEP_BETWEEN_CLIPS
                        steps between video clips
```

Sample command:
```
python train.py \
	--data data \
	--labels ucfTrainTestlist \
	--output saved_model \
	--num_epochs 1 \
	--batch_size 8 \
	--frames_per_clip 5 \
	--step_between_clips 1
```

### Testing

```
usage: infer.py [-h] [-d DATA_DIR] [-l LABEL_DIR] [-i SAVED_MODEL]
                [-bs BATCH_SIZE] [-fc FRAMES_PER_CLIP]
                [-sc STEP_BETWEEN_CLIPS]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_DIR, --data DATA_DIR
                        path to training data videos
  -l LABEL_DIR, --labels LABEL_DIR
                        path to action recognition labels
  -i SAVED_MODEL, --saved_model SAVED_MODEL
                        path to saved model.
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size of data
  -fc FRAMES_PER_CLIP, --frames_per_clip FRAMES_PER_CLIP
                        framers per video clip
  -sc STEP_BETWEEN_CLIPS, --step_between_clips STEP_BETWEEN_CLIPS
                        steps between video clips
```

Sample command:
```
python infer.py \
	--data data \
	--labels ucfTrainTestlist \
	--saved_model saved_model/best_model.pth \
	--batch_size 16 \
	--frames_per_clip 5 \
	--step_between_clips 1
```
