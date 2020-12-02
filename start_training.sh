python train.py \
	--data filtered_data \
	--labels ucfTrainTestlist \
	--output saved_model \
	--num_epochs 1 \
	--batch_size 8 \
	--frames_per_clip 5 \
	--step_between_clips 1
