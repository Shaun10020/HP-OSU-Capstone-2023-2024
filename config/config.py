
img_extension = '.png'
label_extension = '.bmp'

features = ['baunderlay','black','cyan','yellow','magenta']
labels = ['HighMoistureSimplexPage_bitmap','HighMoistureSimplexObject1.3cm_bitmap','HighMoistureSimplexObject2.5cm_bitmap']
duplex_labels = ['HighMoistureDuplexObject2.2cm_bitmap','HighMoistureDuplexPageDelta_bitmap']

train_val_ratio = [0.7,0.3]

random_seed = 42
batch_size = 10

height = 300
width = 300