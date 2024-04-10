from argparse import ArgumentParser

'''
This script is reponsible for setting up argument parser object, default value
'''

def get_arguments():
    """
    Defines command-line arguments, and parses them.

    """
    parser = ArgumentParser()

    parser.add_argument(
        "--input_folder",
        "-i",
        default="./data/cache/DP_a2200_xml_ff2c81d8ad6655f915cbaa558ee7bf9e878730a8",
        help=("Folder to get data from."))
    
    parser.add_argument(
        "--output_folder",
        "-o",
        default="./output",
        help=("Folder to save output to."))
    
    parser.add_argument(
        "--label_folder",
        "-l",
        default="./data/output/DP_a2200_xml_ff2c81d8ad6655f915cbaa558ee7bf9e878730a8",
        help=("Folder to get label from."))
    
    parser.add_argument(
        "--save_folder",
        "-s",
        default="./checkpoints",
        help=("Folder to save model weights"))
    
    parser.add_argument(
        "--model",
        choices=["unet","enet","deeplabv3","customunet1","customenet1"],
        default="unet",
        help=("Choose from 'unet', 'customunet1','enet','customenet1' or 'deeplabv3'"))

    parser.add_argument(
        "--mode",
        "-m",
        choices=["train","test","inference"],
        default="train",
        help=("Choose from 'train', 'test' or 'inference'"))
    
    parser.add_argument(
        "--epoch",
        default=5,
        help=("Epoch number, default is 5"))
    
    parser.add_argument(
        "--lr",
        default=5e-5,
        help=("Learning rate, default is 5e-5"))

    parser.add_argument(
        "--lr_decay",
        default=0.1,
        help=("Learning rate decay factor, default is 0.1"))

    parser.add_argument(
        "--lr_epoch",
        default=50,
        help="The number of epochs before adjusting the learning rate. "
        "Default: 50")
    
    parser.add_argument(
        "--batch",
        default=5,
        help=("Batch Size, default is 5"))
    
    parser.add_argument(
        "--dataset",
        choices=["simplex","duplex"],
        default="simplex",
        help=("Dataset, choose from 'simplex' or 'duplex'"))

    return parser.parse_args()