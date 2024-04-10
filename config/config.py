'''
This script is a configuration file
'''

## Input configutions
img_extension = '.png'
features = ['baunderlay','black','cyan','yellow','magenta']

## Output configurations
label_extension = '.bmp'
labels = ['HighMoistureSimplexPage_bitmap','HighMoistureSimplexObject1.3cm_bitmap','HighMoistureSimplexObject2.5cm_bitmap']
detect_labels = ['HighMoistureSimplexPage','HighMoistureSimplexObject1.3cm','HighMoistureSimplexObject2.5cm']
detect_duplex_labels = ['HighMoistureDuplexObject2.2cm','HighMoistureDuplexPageDelta']
duplex_labels = ['HighMoistureDuplexObject2.2cm_bitmap','HighMoistureDuplexPageDelta_bitmap']

## Dataset split ratio
train_val_ratio = [0.9,0.1]
train_test_ratio = [0.7,0.3]

## Dataloader configution
pin_memory = True

## Image transformation
input_height = 300
input_width = 300
output_height = 300
output_width = 300

## Threshold for binary conversion
threshold = 0.5

## Dataloader filenames
train_dataloader_name = 'Train_loader.pth'
val_dataloader_name = 'Validation_loader.pth'
test_dataloader_name = 'Test_loader.pth'