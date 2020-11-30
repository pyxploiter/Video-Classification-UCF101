import torch
from torchvision import datasets, transforms


def custom_collate(batch):
	filtered_batch = []
	for video, _, label in batch:
		filtered_batch.append((video, label))
	return torch.utils.data.dataloader.default_collate(filtered_batch)

def get_transforms():
	tfs = transforms.Compose([
		# scale in [0, 1] of type float
		transforms.Lambda(lambda x: x / 255.),
		# reshape into (T, C, H, W) for easier convolutions
		transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
		# rescale to the most common size
		transforms.Lambda(lambda x: torch.nn.functional.interpolate(x, (240, 320))),
	])

	return tfs

def load_data(data_dir, 
			  label_dir, 
			  batch_size,
			  frames_per_clip, 
			  step_between_clips,
			  test_only=False,
			):

	tfs = get_transforms()

	if not test_only:
		print("Loading train data...")
		train_dataset = datasets.UCF101(
									data_dir, 
									label_dir, 
									frames_per_clip=frames_per_clip,
									step_between_clips=step_between_clips, 
									fold=1,
									train=True, 
									transform=tfs
								  )

		train_loader = torch.utils.data.DataLoader(train_dataset, 
												  batch_size=batch_size, 
												  shuffle=True,
												  collate_fn=custom_collate
												)

	print("Loading test data...")
	test_dataset = datasets.UCF101(
							  data_dir, 
							  label_dir, 
							  frames_per_clip=frames_per_clip,
							  step_between_clips=step_between_clips, 
							  fold=1,
							  train=False, 
							  transform=tfs
							)

	test_loader = torch.utils.data.DataLoader(test_dataset, 
										  batch_size=batch_size, 
										  shuffle=False,
										  collate_fn=custom_collate
										)

	if not test_only:
		print("Training classes:", train_dataset.classes)
		print("Total training samples:", len(train_dataset))
	print("Total testing samples:", len(test_dataset))
	print()

	if not test_only:
		return train_loader, test_loader

	return test_loader