def print_output(index, masks, outputs):
    '''
    shows the ground truth masks and predicted masks
    
    param index: the index of the image to be shown
    param masks: the ground truth masks of the test set
    param outputs: the predicted outputs of the test set
    
    '''
    
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    titles = ['HighMoistureSimplexPage_bitmap', 'HighMoistureSimplexObject1.3cm_bitmap', 'HighMoistureSimplexObject2.5cm_bitmap']

    for i in range(3):
        # Display each mask in the first row
        axs[0, i].imshow(masks[1][i].cpu(), cmap='gray')
        axs[0, i].axis('off')  # Hide axes for clarity
        axs[0, i].set_title(titles[i])

        # Display each output in the second row
        axs[1, i].imshow(outputs[1][i].cpu(), cmap='gray')
        axs[1, i].axis('off')  # Hide axes for clarity
    #     axs[1, i].set_title(titles[i])

    plt.tight_layout()  # Adjust subplots to fit into the figure area
    plt.show()  # Display the plot
    
    
