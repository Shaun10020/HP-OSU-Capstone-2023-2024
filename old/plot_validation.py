import csv 
import matplotlib.pyplot as plt
import seaborn as sns

def data_preparation():
    # Filepaths
    csv_file1 = "../csv_files/unet-duplex-train-epoch.csv"
    csv_file2 = "../csv_files/enet-duplex-train-epoch.csv"
    csv_file3 = "../csv_files/deeplabv3-duplex-train-epoch.csv"
    csv_file4 = "../csv_files/deeplabv3+-duplex-train-epoch.csv"
    validation_loss1 = []
    validation_loss2 = []
    validation_loss3 = []
    validation_loss4 = []

    with open(csv_file1,"r") as _csv_file:
        reader = csv.reader(_csv_file)
        index = next(reader).index('Validation Loss')
        for line in reader:
            validation_loss1.append(float(line[index]))
        
    with open(csv_file2,"r") as _csv_file:
        reader = csv.reader(_csv_file)
        index = next(reader).index('Validation Loss')
        for line in reader:
            validation_loss2.append(float(line[index]))

    with open(csv_file3,"r") as _csv_file:
        reader = csv.reader(_csv_file)
        index = next(reader).index('Validation Loss')
        for line in reader:
            validation_loss3.append(float(line[index]))

    with open(csv_file4,"r") as _csv_file:
        reader = csv.reader(_csv_file)
        index = next(reader).index('Validation Loss')
        for line in reader:
            validation_loss4.append(float(line[index]))
    return validation_loss1,validation_loss2,validation_loss3,validation_loss4

def plot_graph(unet,enet,deeplabv3,deeplabv3plus):
    plt.clf()
    ## plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_context("talk")
    plt.figure(figsize = (10,6))
    plt.title("Validation Loss over Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0.025,0.10)
    plt.plot(range(len(unet)),unet,"r-",label="UNet",linewidth = 1.5)
    plt.plot(range(len(enet)),enet,"g-",label="ENet",linewidth = 1.5)
    plt.plot(range(len(deeplabv3)),deeplabv3,"b-",label="DeeplabV3",linewidth = 1.5)
    plt.plot(range(len(deeplabv3plus)),deeplabv3plus,"m-",label="DeeplabV3+",linewidth = 1.5)
    plt.legend(frameon=True, loc='best')
    plt.grid(True)
    plt.savefig("Validation_5.png",dpi=500)
    return

if __name__ == '__main__':
    u, e,dv3, dv3plus = data_preparation()
    plot_graph(u, e,dv3, dv3plus )
