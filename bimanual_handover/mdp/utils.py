import torch
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
import numpy as np
import os
import matplotlib.pyplot as plt

# Class to implement sequence storage
class TensorQueue:
    def __init__(self, max_size, element_shape):

        # Initialize the queue tensor with the max number of elements and given shape
        self.queue = torch.zeros((max_size, *element_shape))
        self.max_size = max_size
        self.current_size = 0
    
    def enqueue(self, new_element):
        # Shift elements to the left if the queue is full
        if self.current_size >= self.max_size:
            self.queue[:-1] = self.queue[1:].clone()
            self.queue[-1] = new_element
        else:
            # Add to the current end if the queue is not full
            self.queue[self.current_size] = new_element
            self.current_size += 1
    
    def get_queue(self):
        # Return only the filled part of the queue if not yet full
        return self.queue[:self.current_size]
    


# Reward computation --> TODO
@torch.jit.script
def compute_rewards(rew_position_tracking: float,
                    rew_position_tracking_fine_grained: float,
                    rew_orientation_tracking: float,
                    rew_dual_quaternion_error: float,
                    rew_action_rate: float,
                    rew_joint_vel: float,
                    device: str):
    
    
    return torch.tensor([0.0]).to(device)


# Function for image saving from IsaacLab --> Taken from source/standalone/demos/cameras.py
def save_images_grid(
    images: list[torch.Tensor],
    cmap: str | None = None,
    nrow: int = 1,
    subtitles: list[str] | None = None,
    title: str | None = None,
    filename: str | None = None,
):
    """Save images in a grid with optional subtitles and title.

    Args:
        images: A list of images to be plotted. Shape of each image should be (H, W, C).
        cmap: Colormap to be used for plotting. Defaults to None, in which case the default colormap is used.
        nrows: Number of rows in the grid. Defaults to 1.
        subtitles: A list of subtitles for each image. Defaults to None, in which case no subtitles are shown.
        title: Title of the grid. Defaults to None, in which case no title is shown.
        filename: Path to save the figure. Defaults to None, in which case the figure is not saved.
    """
    
    # Show images in a grid
    n_images = len(images)
    ncol = int(np.ceil(n_images / nrow))

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2)) # -> tiene que ser un numpy array
    print(filename)
    axes = np.array(axes)

    # Axes(0.125,0.11;0.775x0.77)
    axes = axes.flatten()

    # Plot images
    for idx, (img, ax) in enumerate(zip(images, axes)):
        img = img.detach().cpu().numpy()
        ax.imshow(img, cmap=cmap)
        ax.axis("off")
        if subtitles:
            ax.set_title(subtitles[idx])

    # Remove extra axes if any
    for ax in axes[n_images:]:
        fig.delaxes(ax)

    # Set title
    if title:
        plt.suptitle(title)

    # Adjust layout to fit the title
    plt.tight_layout()

    # Save the figure
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)

    # Close the figure
    plt.close()