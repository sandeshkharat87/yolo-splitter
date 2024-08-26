# yolo-splitter
Tool to create,modify YOLO dataset.

## Installation
```bash
pip install yolosplitter
```

## Uses
```python
from yolosplitter import YoloSplitter

ys = YoloSplitter(imgFormat=['.jpg', '.jpeg', '.png'], labelFormat=['.txt'] )

# If you have yolo-format dataset already on the system
df = ys.from_yolo_dir(input_dir="yolo_dataset",ratio=(0.7,0.2,0.1),return_df=True)

# If you have mixed Images and Labels in the same directory
df = ys.from_mixed_dir(input_dir="mydataset",ratio=(0.7,0.2,0.1),return_df=True)

# To see train/test/val split size, total error files, all class names from annotation files
ys.info()

# !!! changed show_dataframe to get_dataframe()
# to see dataframe
ys.get_dataframe()
```
![2024-01-30_08-19](https://github.com/sandeshkharat87/yolo-splitter/assets/47347413/b2475cde-cbb7-410f-a4df-dd2622698ee1)



```python
ys.save_split(output_dir="potholes")
```

```bash
Saving New split in 'potholes' dir
100%|██████████| 118/118 [00:00<00:00, 1352.79it/s]
```

```python
# Use ys.show_show_errors  to show filename which have errors
ys.show_errors()

# Use ys.show_dataframe to see dataframe created on the dataset
ys.get_dataframe()

# To see train/test/val split size, total error files, all class names from annotation files
ys.info()
```


### Input Directory
```
MyDataset/
├── 02.png
├── 02.txt
├── 03.png
├── 03.txt
├── 04.png
├── 04.txt
├── 05.png
├── 05.txt
├── 06.png
├── 06.txt
├── 07.png
├── 07.txt
├── 08.png
├── 08.txt
├── 09.png
├── 09.txt
├── 10.png
├── 10.txt
├── 11.png
└── 11.txt
```

### Output Directory
```
MyDataset-splitted/
├── data.yaml
├── train
│   ├── images
│   │   ├── 03.png
│   │   ├── 04.png
│   │   ├── 05.png
│   │   ├── 07.png
│   │   ├── 08.png
│   │   ├── 09.png
│   │   └── 10.png
│   └── labels
│       ├── 03.txt
│       ├── 04.txt
│       ├── 05.txt
│       ├── 07.txt
│       ├── 08.txt
│       ├── 09.txt
│       └── 10.txt
└── val
    ├── images
    │   ├── 02.png
    │   ├── 06.png
    │   └── 11.png
    └── labels
        ├── 02.txt
        ├── 06.txt
        └── 11.txt
```
# Change Log
## Stable
* 2024-08-26 version 5.0.0
    * Optimize code and speedup execution. Thanks for incredible work [https://github.com/MarcelloCuoghi]
      
* 2023-04-25 version 4.9.1
    * Fixed. "Having a newline at the end of the file causes an error:
      ValueError('invalid literal for int() with base 10: ''')". Thanks to [https://github.com/Maxvgrad] for finding bug.

* 2023-01-30 version 4.9
    * Fixed Fixes Annotation Parse Error. Thanks to [https://github.com/Xiteed] 
       
* 2023-12-20 version 4.8
    * Changed yaml file style

* 2023-12-19 version 4.7
    * Fix output dir of `val` to `valid` thanks to [https://github.com/AndreasFridh]
    * Added `ys.info()` To see train/test/val split size, total error files, all class names from annotation files
    * Changed `ys.show_dataframe` to `ys.get_dataframe()`
    * small bug fixes
    
