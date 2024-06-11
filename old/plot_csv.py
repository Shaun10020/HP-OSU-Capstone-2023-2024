import csv 
import matplotlib.pyplot as plt
import seaborn as sns

def data_preparation():
    # Filepaths
    csv_file = "../csv_files/deeplabv3+-duplex-train-epoch.csv"
    train_loss = []
    valid_loss = []

    with open(csv_file,"r") as _csv_file:
        reader = csv.reader(_csv_file)
        index = next(reader).index('Train Loss')
        for line in reader:
            train_loss.append(float(line[index]))
        
    with open(csv_file,"r") as _csv_file:
        reader = csv.reader(_csv_file)
        index = next(reader).index('Validation Loss')
        for line in reader:
            valid_loss.append(float(line[index]))
    return train_loss,valid_loss

def plot_graph(train_loss,valid_loss):
    plt.clf()
    ## plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_context("talk")
    plt.figure(figsize = (12,6))
    plt.title("Loss over Epoch - DeepLabV3+")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0.025,0.10)
    plt.plot(range(len(train_loss)),train_loss,"b-",label="Training Loss",linewidth = 6,alpha=0.3)
    plt.plot(range(len(valid_loss)),valid_loss,"r-",label="Validation Loss", linewidth = 2,alpha = 1)
    plt.legend(frameon=True, loc='best')
    plt.grid(True)
    plt.savefig("DeepLabV3+.png",dpi=100)
    return

if __name__ == '__main__':
    t, v = data_preparation()
    plot_graph(t,v)
