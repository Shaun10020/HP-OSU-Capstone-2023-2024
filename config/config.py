
img_extension = '.png'
label_extension = '.bmp'
input_label = "input"
features = ['baunderlay','black','cyan','yellow','magenta']
labels = ['HighMoistureSimplexPage_bitmap','HighMoistureSimplexObject1.3cm_bitmap','HighMoistureSimplexObject2.5cm_bitmap']
duplex_labels = ['HighMoistureDuplexObject2.2cm_bitmap','HighMoistureDuplexPageDelta_bitmap']

pin_memory = True
train_val_ratio = [0.7,0.3]
random_seed = 42
batch_size = 10
input_height = 300
input_width = 300
output_height = 165
output_width = 165
