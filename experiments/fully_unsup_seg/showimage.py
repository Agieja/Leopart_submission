import os
import sys

import torchvision

import numpy as np
import matplotlib.pyplot as plt 

from torchvision.datasets.vision import StandardTransform
from torchvision.transforms import ToTensor, Compose, Resize, Normalize


## Code to fix the data module errors
# Set the root folder to the current working directory
root_folder = os.getcwd()

# Add the root folder to the Python path
sys.path.append(root_folder)

from data.VOCdevkit.vocdata import VOCDataModule

def visualize_features(image, mask):
    """
    Visualize an image-mask pair.
    
    Args:
        image (torch.Tensor): Image tensor.
        mask (torch.Tensor): Mask tensor.
    """
    # Convert tensors to numpy arrays, transpose image and normalize image values
    image = np.transpose(image.numpy(), (1, 2, 0))
    mask = mask.numpy().squeeze()
    image = (image - image.min()) / (image.max() - image.min())

    # Create subplots
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Plot image
    ax[0].imshow(image)
    ax[0].set_title("Image")

    # Plot mask
    ax[1].imshow(mask)
    ax[1].set_title("Mask")

    # Plot mask on image
    overlaid_image = np.copy(image)
    overlaid_image[mask > 0] = 0.5 * overlaid_image[mask > 0] + 0.5 * np.array([1, 0, 0])  # Red color for mask
    ax[2].imshow(overlaid_image)
    ax[2].set_title("Image with overlaid mask")

    # Show plot
    plt.show()

def main():
    # Load data module and get data loader
    imgloader = data_module.val_dataloader()

    while True:
            # Get the number of samples in the dataset
            num_samples = len(imgloader.dataset)

            # Prompt for index selection
            index = input(f"Enter an index between 0 and {num_samples - 1} (or 'exit' to quit): ")

            # Exit condition -> ctrl+c or 'exit'
            if index.lower() == 'exit':
                break

            # Convert index to integer
            try:
                index = int(index)
            except ValueError:
                print("Invalid input. Please enter a valid index or 'exit' to quit.")
                continue

            # Check if the index is within the valid range
            if index < 0 or index >= num_samples:
                print("Invalid index. Please enter a valid index.")
                continue

            # Get the image-mask pair from the selected index
            selected_img, selected_mask = imgloader.dataset[index]

            # Visualize selection
            visualize_features(selected_img, selected_mask)


if __name__ == "__main__":

    data_dir = "data/VOCdevkit/VOC_data/"
    batch_size = 4
    input_size = 448
        # Init data module
    val_image_transforms = Compose([Resize((input_size, input_size)),
                                    ToTensor(),
                                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_target_transforms = Compose(
        [Resize((input_size, input_size), interpolation=torchvision.transforms.InterpolationMode.NEAREST),
         ToTensor()])
    data_module = VOCDataModule(batch_size=batch_size,
                                num_workers=6,
                                train_split="train",
                                val_split="val",
                                data_dir=data_dir,
                                train_image_transform=StandardTransform(val_image_transforms, val_target_transforms),
                                val_image_transform=val_image_transforms,
                                val_target_transform=val_target_transforms,
                                return_masks=True,
                                drop_last=False,
                                shuffle=False)
    data_module.setup()
    
    
    main()
