
img_extension = '.png'
label_extension = '.bmp'
input_label = "input"
features = ['baunderlay','black','cyan','yellow','magenta']
labels = ['HighMoistureSimplexPage_bitmap','HighMoistureSimplexObject1.3cm_bitmap','HighMoistureSimplexObject2.5cm_bitmap']
duplex_labels = ['HighMoistureDuplexObject2.2cm_bitmap','HighMoistureDuplexPageDelta_bitmap']


train_val_ratio = [0.9,0.1]
train_test_ratio = [0.7,0.3]

random_seed = 42
batch_size = 10
pin_memory = True

input_height = 300
input_width = 300
output_height = 165
output_width = 165

weight_decay = 2e-4
lr = 1e-5
lr_decay = 0.1
lr_decay_epochs = 100