import matplotlib.pyplot as plt
import os 
from PIL import Image


labels = ['HighMoistureSimplexPage_bitmap','HighMoistureSimplexObject1.3cm_bitmap','HighMoistureSimplexObject2.5cm_bitmap']
duplex_labels = ['HighMoistureDuplexObject2.2cm_bitmap','HighMoistureDuplexPageDelta_bitmap']

row_header = ['Truth','U-Net','DeepLabV3+','DeepLabV3','E-Net']
column_header = ['simplex_page','simplex_object_1.3','simplex_object_2.5','duplex_object_2.2','E-duplex_page']

def display_images(unet,deeplabv3p,deeplabv3,enet, truth_dir):
    all_labels = labels + duplex_labels
    num_labels = len(all_labels)
    
    # Create a figure with 2 rows and one column for each label plus one for the side labels
    fig, axs = plt.subplots(5, num_labels + 1, figsize=(18, 20), gridspec_kw={'width_ratios': [1] + [3] * num_labels})

    # Set up the labels on the left-hand side
    for i in range(5):
        axs[i, 0].text(0.5, 0.5, row_header[i], verticalalignment='center', horizontalalignment='center', fontsize=14, fontweight='bold', transform=axs[i, 0].transAxes)
        axs[i, 0].axis('off')  # Turn off axis for the label cell

    for i, label in enumerate(all_labels):
        # Load predicted image
        unet_path = os.path.join(unet, f"{label}.bmp")
        deeplabv3p_path = os.path.join(deeplabv3p, f"{label}.bmp")
        deeplabv3_path = os.path.join(deeplabv3, f"{label}.bmp")
        enet_path = os.path.join(enet, f"{label}.bmp")
        truth_path = os.path.join(truth_dir, f"{label}.bmp")

        try:
            unet_img = Image.open(unet_path).convert('L')
            deeplabv3p_img = Image.open(deeplabv3p_path).convert('L')
            deeplabv3_img = Image.open(deeplabv3_path).convert('L')
            enet_img = Image.open(enet_path).convert('L')
            truth_img = Image.open(truth_path).convert('L')
        except FileNotFoundError:
            # Placeholder for missing images
            unet_img = Image.new('L', (200, 200), color='grey')
            deeplabv3p_img = Image.new('L', (200, 200), color='grey')
            deeplabv3_img = Image.new('L', (200, 200), color='grey')
            enet_img = Image.new('L', (200, 200), color='grey')
            truth_img = Image.new('L', (200, 200), color='grey')

        # Display truth image
        axs[0, i + 1].set_title(column_header[i], fontsize=10, fontweight='bold')  # Show label above the predicted image
        axs[0, i + 1].imshow(truth_img, cmap='gray', aspect='auto')
        axs[0, i + 1].axis('off')
        
        
        # Display predicted image
        axs[1, i + 1].imshow(unet_img, cmap='gray', aspect='auto')
        axs[1, i + 1].axis('off')
        
        axs[2, i + 1].imshow(deeplabv3p_img, cmap='gray', aspect='auto')
        axs[2, i + 1].axis('off')
        
        axs[3, i + 1].imshow(deeplabv3_img, cmap='gray', aspect='auto')
        axs[3, i + 1].axis('off')
        
        axs[4, i + 1].imshow(enet_img, cmap='gray', aspect='auto')
        axs[4, i + 1].axis('off')
    
    # Adjust layout and show the plot
    plt.tight_layout(pad=1)
    plt.savefig(".\\output_viz.png",dpi=150)

# Call the function to display images
display_images(".\\2009_practicemodel_predict_unet",".\\2009_practicemodel_predict_deeplabv3+",".\\2009_practicemodel_predict_deeplabv3",".\\2009_practicemodel_predict_enet", ".\\2009_practicemodel")