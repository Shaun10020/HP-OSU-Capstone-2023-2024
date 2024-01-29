from argparse import ArgumentParser


def get_arguments():
    """Defines command-line arguments, and parses them.

    """
    parser = ArgumentParser()

    parser.add_argument(
        "--input_folder",
        "-i",
        default="./data/dcdata/cache/DP_a2200_xml_ff2c81d8ad6655f915cbaa558ee7bf9e878730a8",
        help=("Folder to get data from."))
    
    parser.add_argument(
        "--label_folder",
        "-l",
        default="./data/dcdata/output/DP_a2200_xml_ff2c81d8ad6655f915cbaa558ee7bf9e878730a8",
        help=("Folder to get label from."))
    
    parser.add_argument(
        "--save_folder",
        "-s",
        default="./checkpoints",
        help=("Folder to save model weights"))
    
    parser.add_argument(
        "--model",
        choices=["unet","enet","deeplabv3"],
        default="unet",
        help=("Choose from 'unet', 'enet' or 'deeplabv3'"))
    
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
        default=1e-5,
        help=("Learning rate, default is 1e-5"))
    
    parser.add_argument(
        "--batch",
        default=10,
        help=("Batch Size, default is 10"))
    
    parser.add_argument(
        "--dataset",
        choices=["simplex","duplex"],
        default="simplex",
        help=("Dataset, choose from 'simplex' or 'duplex'"))

    return parser.parse_args()