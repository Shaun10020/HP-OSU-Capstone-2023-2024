
img_extension = '.png'
label_extension = '.bmp'
features = ['baunderlay','black','cyan','yellow','magenta']
labels = ['HighMoistureSimplexPage_bitmap','HighMoistureSimplexObject1.3cm_bitmap','HighMoistureSimplexObject2.5cm_bitmap']
duplex_labels = ['HighMoistureDuplexObject2.2cm_bitmap','HighMoistureDuplexPageDelta_bitmap']


train_val_ratio = [0.9,0.1]
train_test_ratio = [0.7,0.3]

batch_size = 10
pin_memory = True

input_height = 300
input_width = 300
output_height = 300
output_width = 300
threshold = 0.5
