import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

cudnn.benchmark = True
plt.ion() # interactive mode


#data augmentation and normalization for training
#Just normalization for validation

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])
    ]),

    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),   #image is converted into Pytorch tensors
        transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])  #first 0.4 numbers are mean of R,G,B channels respectively while the other 3 numbers are Std Devs
    ]),
}


data_dir = '/home/ibab/Documents/Deep_Learning/Pre-trained model tutorial /hymenoptera_data'

image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x])
                  for x in ['train','val']
                  }   #for each x in ['train','val'], create an ImageFolder dataset
                      #os.path.join(data_dir,x) this constructs the path
                      #when x = 'train' we get data/hymenoptera_data/train
                      #when x = 'val' we get data/hymenoptera_data/val
                      #Go to this folder, find the images, assign their labels based on their folders, and apply the appropriate preprocessing.
# image_datasets
# │
# ├── train → ImageFolder(training images)
# │
# └── val   → ImageFolder(validation images)


dataloaders = {x:torch.utils.data.DataLoader(image_datasets[x],batch_size=4,shuffle=True,num_workers = 4)
               for x in ["train","val"]}
# torch.utils.data.DataLoader(
#     image_datasets[x],this Creates a DataLoader using the dataset we just created.
#batch_size=4 means the model will get 4 images at a time , shuffle randomly rearranges the images between epochs
#what does num_workers=4  do ? so it creates 4 worker processes to load/preprocess images  (load image → preprocess → load image → preprocess...) 4 workers will work in parallel

# dataloaders
# │
# ├── train → DataLoader
# │             └── batches of 4
# │
# └── val   → DataLoader
#               └── batches of 4

# A Dataset stores the images and labels.
# A DataLoader actually helps feed those images to the model in batches.
dataset_sizes = {x:len(image_datasets[x]) for x in ['train','val']}
# This finds the number of images in each dataset.

class_names = image_datasets['train'].classes
#this gets the names of the classes that is ants,bees


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


def imshow(inp,title=None):
    """Display image for Tensor"""
    inp = inp.numpy().transpose((1,2,0))
    #pytorch stores an image as Channels × Height × Width

    #but matplotlib.pyplot.imshow() expects Height × Width × Channels
    # inp.numpy()
    #converts the Pytorch tensor into Numpy array

    # Before:
    #
    # 3 × 224 × 224
    # │   │    │
    # C   H    W
    #

    # After:
    # 224 × 224 × 3
    # │    │    │
    # H    W    C
    #
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean   #undo normalization
    inp = np.clip(inp, 0, 1)
    # After reversing normalization, some values might be slightly below 0 or above 1.
    #but an image is displayed by Matplotlib shld generally have the values between 0 and 1 , so np.clip() forces them into that range

    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


#Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

#MAke a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
plt.ioff()
plt.show()


#Training the model
def train_model(model,criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    #Create a temporary directory to save training checkpoints
    with  TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir,'best_model_params.pt') #creating path where the best model will be saved

        torch.save(model.state_dict(),best_model_params_path)  #model.state_dict() contains the learned parameters/weights of the model
        #while torch.save() saves those weights to temporary file

        best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs -1 }')
        print('-' * 10)

        #Each epoch has a training and validation phase
        for phase in ['train','val']:
            # Epoch
            #  0
            #  ├── train
            #  └── val
            #
            # Epoch
            #  1
            #  ├── train
            #  └── val
            #
            # Epoch
            #  2
            #  ├── train
            #  └── val
            if phase in ['train','val']:
                if phase == 'train':
                    model.train()  #set model to training mode
                else:
                    model.eval() #set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                #iterate over data  basically looping through batches
                for inputs, labels in dataloaders[phase]:
                    # inputs:
                    # [image1, image2, image3, ...]
                    #
                    # labels:
                    # [cat, dog, dog, ...]  every batch contains smthing like this
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    #zero the parameter gradients
                    optimizer.zero_grad()   #we clear the gradients after each batch so that it is ready for the next batch


                    #forward
                    #track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        #if we are in training phase it will  be True the gradients will be calculated if in validation phase it will be false and gradients wont be calculated
                        outputs = model(inputs)   #inputs are passed thru the neural network
                        _, preds = torch.max(outputs,1)  #This finds which class has the highest output.
                        #why we are using _, because torch.max returns two things max. value and index but we only want index, which is the predicted class so we ignore maximum value

                        loss = criterion(outputs, labels)

                        # for eg.
                        # Predicted → Cat
                        # Actual    → Dog
                        #   ↓
                        # LOSS

                        #backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()   #calculates the gradients of the loss with respect to the model's parameters
                            optimizer.step()   #update weights , optimizer uses the gradients calculated above to update the model's weights

                    #statistics
                    running_loss += loss.item() * inputs.size(0)  #it keeps track of the loss in the entire dataset ,
                    #loss.item() converts PyTorch tensor containing the loss into a normal Python number
                    # tensor(0.5321)-> 0.5321
                    #input.size(0) gets the sample in the batch if batch size = 32 we will get 32
                    #thus running_loss calculate the total loss contribution of that batch

                    running_corrects += torch.sum(preds == labels.data)
                    #this compares predictions with actual labels

                    # #Example:
                    # preds: [0, 1, 1, 0]
                    # labels: [0, 1, 0, 0]
                    #
                    # Comparison:
                    # [True, True, False, True]

                    #torch.sum() counts the True values

                if phase == 'train':
                    scheduler.step()

                #this happens after thet training phase of each epoch
                #the scheduler changes the learning rate according to whatever schedule we have defined

                epoch_loss = running_loss / dataset_sizes[phase]  #gives the avg loss for each epoch
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                #calculates the accuracy
                # 900
                # correct
                # 1000
                # total
                #
                # accuracy = 900 / 1000
                # = 0.9

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:  #Are we in validation AND is this the best validation accuracy we've seen so far?
                    best_acc = epoch_acc   #updating best accuracy
                    torch.save(model.state_dict(), best_model_params_path) #saves the best model

            print()


        time_elapsed = time.time() - since #gives total training duration in seconds

        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        # converts time into mins and secs

        print(f'Best val Acc: {best_acc:4f}')
        #gives the highest validation accuracy

        #load the best model weights
        model.load_state_dict(torch.load(best_model_params_path,weights_only =True))
        #even after the best epoch the training kept on going but we only need the best weights which we are loading into the model
        # model.load_state_dict() - puts the best weights in the model
    return model

def visualize_model(model, num_images = 6):  #6 is the number of validation images we want to display
    was_training = model.training
    #model.training tells us whether the model is currently in:
    # True → training mode
    # False → evaluation mode

    #if model was in training mode it model.training would return True and we save that value in was_training because we will
    # change the model to evaluation mode
    #After visualization we want to put the model back into whatever mode it was originally in

    model.eval() #switching model into evaluation/testing mode

    images_so_far = 0
    fig = plt.figure() #This creates an empty Matplotlib figure where the images will be displayed.

    with torch.no_grad(): #Normally, PyTorch keeps track of operations so that it can calculate gradients during backpropagation.
    # But we're not training the model here. We're only making predictions.
         for i,(inputs,labels) in enumerate(dataloaders['val']):
             #dataloader consists of:
    # dataloaders = {
    #     'train': train_loader,
    #     'val': val_loader
    # }
             #each iteration gives inputs,labels
             #inputs = contains images
             #labels = contains class of each image
             inputs = inputs.to(device)
             labels = labels.to(device)

             outputs = model(inputs)   #this is where the Neural Network actually processes the images
             #outputs =[
                       #   [2.1, 0.3, 4.7, 1.2],
                       #   [5.2, 1.1, 0.4, 0.7],
                       #   ...
                       # ]

             #in which each row corresponds to one image
             #each column corresponds to one class
             #These values are usually logits, not probabilities.
             #the highest value is taken for each image

             _, preds = torch.max(outputs,1)  #torch.max() finds the maximum value.
             #1 means find the maximum across the class dimension

             #outputs =
             # [
             #    [2.1, 0.3, 4.7, 1.2],
             #    [5.2, 1.1, 0.4, 0.7]
             # ]
             # then , torch.max(outputs, 1) gives -> maximum values → [4.7, 5.2] indices        → [2, 0]
             #_     → maximum scores
             # preds → class indices
             #preds = [2, 0]

             #Image 1 → predicted class 2
             # Image 2 → predicted class 0


             for j in range(inputs.size()[0]):  #looping through each image in the batch
                 images_so_far += 1
                 ax = plt.subplot(num_images//2,2,images_so_far)
                 #This determines where each image will appear.

                 #if num_images = 6
                 #then, num_images //2 = 3
                 #which means there will be 3 rows and 2 cols
                 # #┌─────────┬─────────┐
                 #  │ Image 1 │ Image 2 │
                 #  ├─────────┼─────────┤
                 #  │ Image 3 │ Image 4 │
                 #  ├─────────┼─────────┤
                 #  │ Image 5 │ Image 6 │
                 #  └─────────┴─────────┘

                 # images_so_far specifies which position we're currently in '
                 ax.axis('off')  #since its image we remove x and y axis
                 ax.set_title(f'predicted: {class_names[preds[j]]}')

                 imshow(inputs.cpu().data[j])  #diplays the current image

                 if images_so_far == num_images: #so once all the images are displayed we stop
                     model.train(mode=was_training) #we restore original model mode thus now the model is in training mode
                     #This is important because we don't want the visualization function to permanently change the model's state.
                     return

         model.train(mode=was_training) #So this final line ensures that the model is still restored to its original mode.

































