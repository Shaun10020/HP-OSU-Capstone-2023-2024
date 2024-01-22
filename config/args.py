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
        default="./saved_weights",
        help=("Folder to save weights"))
    
    parser.add_argument(
        "--mode",
        "-m",
        choices=["train","test"],
        default="train",
        help=("Choice from 'train' or 'test'"))
    
    parser.add_argument(
        "--method",
        choices=["seg","bb"],
        default="seg",
        help=("Choice from 'seg'(segmentation) or 'bb'(bounding box)"))
    

    return parser.parse_args()