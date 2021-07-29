from home_scenes_dataset import HomeScenesDataset
from training_utils import collate_fn, get_transform, get_instance_model_default, get_instance_model_modified
import torch
import argparse
from references.detection import utils
from references.detection.engine import train_one_epoch, evaluate

parser = argparse.ArgumentParser(description='training maskrcnn')
parser.add_argument('-mdf', '--modified', type=str,
                    help='Specify if modified model must be trained', required=True)

args = parser.parse_args()
ismodified = args.modified
if ismodified not in ['True', 'False']:
    raise ValueError('argument must be \'True\' or \'False\'')

root = '../dataset_ade20k_filtered'

# use our dataset and defined transformations
dataset = HomeScenesDataset(root, get_transform(train=True))
dataset_test = HomeScenesDataset(root, get_transform(train=False))

# split the dataset in train and test set
batch_size_train = 2
batch_size_test = 2
train_percentage = 0.8
test_percentage = 1 - train_percentage
train_size = int(train_percentage * len(dataset))
test_size = len(dataset) - train_size

indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[0:train_size])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-test_size:])

# define training and validation data loaders
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size_train, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn)

test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size_test, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)

# get the model using our helper function
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 1324 #1323 interesting objects plus background

if ismodified == 'True':
    model = get_instance_model_modified(num_classes)
else:
    model = get_instance_model_default(num_classes)

# move model to the right device
model.to(device)
# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
if ismodified == 'True':
    PATH = 'model_mask_modified.pt'
else:
    PATH = 'model_mask_default.pt'
    
# let's train it for 15 epochs
num_epochs = 15
running_losses = torch.zeros((num_epochs, len(train_loader)))
mask_losses = torch.zeros((num_epochs, len(train_loader)))
classifier_losses = torch.zeros((num_epochs, len(train_loader)))


for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        result = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        running_losses[epoch,:] = result

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        evaluate(model, test_loader, device=device)

        #save losses
        torch.save(running_losses, 'loss.pt')
        torch.save(running_losses, 'loss_mask.pt')
        torch.save(running_losses, 'loss_classifier.pt')

        #save model
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        #'loss': running_losses
        }, PATH)