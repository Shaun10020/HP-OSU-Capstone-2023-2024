import matplotlib.pyplot as plt
from config.config import labels
from utils.convert import convertBinary

def print_output(idx,masks, outputs):
    '''
    shows the ground truth masks and predicted masks
    
    param index: the index of the image to be shown
    param masks: the ground truth masks of the test set
    param outputs: the predicted outputs of the test set
    
    '''
    
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))


    for i in range(len(labels)):
        # Display each mask in the first row
        axs[0, i].imshow(masks[idx][i].cpu().detach().numpy(), cmap='gray')
        axs[0, i].axis('off')  # Hide axes for clarity
        axs[0, i].set_title(labels[i])

        # Display each output in the second row
        axs[1, i].imshow(outputs[idx][i].cpu().detach().numpy(), cmap='gray')
        axs[1, i].axis('off')  # Hide axes for clarity

    plt.tight_layout()  # Adjust subplots to fit into the figure area
    plt.show()  # Display the plot
    
    
