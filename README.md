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
df = ys.from_yolo_dir(input_dir="yolo_dataset",ratio=(0.7,0.2,0.1))

# If you have mixed Images and Labels in the same directory
df = ys.from_mixed_dir(input_dir="mydataset",ratio=(0.7,0.2,0.1))

ys.show_dataframe
```
![2023-10-08_23-28](https://github.com/sandeshkharat87/yolo-splitter/assets/47347413/6e08285d-59c5-4856-8bb5-2eac5f1ec3da)



```python
ys.save_split(output_dir="potholes")
```

```bash
Saving New split in 'potholes' dir
100%|██████████| 118/118 [00:00<00:00, 1352.79it/s]
```

```python
# Use ys.show_show_errors  to show filename which have errors
ys.show_errors

# Use ys.show_dataframe to see dataframe created on the dataset
ys.show_dataframe

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
