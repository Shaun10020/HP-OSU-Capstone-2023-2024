## 1. Setup Python
The version of python being used is 3.11.7, go to this [link](https://www.python.org/downloads/release/python-3117/) to download the python installer.

> [!NOTE]
> if you are using Windows, make sure the python bin folder is added to PATH environment variables. 

## 2. Install Packages
To install the required packages, open up a terminal at the same directory of this file. Run the commands below
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install -r requirements.txt
```
or 
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install matplotlib
pip3 install numpy
pip3 install pillow
pip3 install tqdm
pip3 install ptflops
```

## 3. Data Preparation
There is no fix location for where the data should be placed. For this guide, it's recommended to create and place them under the folder `data`. 

When running the script, you will need to specific the path to the input folders and label folders (if doing training).

```
 <data>
    |
    |---<Input Folder>  <----- specific this folder to run with the main.py
    |       |---<pdf_1 folder>
    |       |       |---<cyan....png>
    |       |       |---<black....png>
    |       |       ...
    |       |---<pdf_2 folder>
    |       |---<pdf_3 folder>
    |       |
    |       ...
    |
    |---<Label Folder>  <----- specific this folder to run with the main.py
            |---<pdf_1 folder>
            |       |---<results.json>
            |       |---<intermediate_results>
            |       |       |
            |       |       ...
            |       ...
            |---<pdf_2 folder>
            |---<pdf_3 folder>
            ...
```

> [!TIP]
> After identified the `Input folder` and `Label folder`, you can change the default value at `\config\args.py` so you won't need to specific folder when running the scripts.

## 4.  Run the script
Now the Data prepation is done, you can run the `main.py` by using python.

#### Example
For Training:
```
python main.py --mode=train --model=unet --batch=8 --epoch=120 --dataset=duplex --input_folder=INPUT_FOLDER --label_folder=LABEL _FOLDER --save_folder=checkpoints
```

For Testing:
```
python main.py --mode=test --model=unet --batch=8 --dataset=duplex --input_folder=INPUT_FOLDER --label_folder=LABEL _FOLDER --save_folder=checkpoints
```

For Inference (generating output from inputs):
```
python main.py --mode=inference --model=unet --batch=8 --input_folder=INPUT_FOLDER --save_folder=checkpoints
```

> [!TIP]
> If you want to run the script multiple time, you can modify the shell script `.\run.sh` and run it instead.