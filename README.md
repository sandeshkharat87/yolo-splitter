# yolo-splitter
Tool that makes it easy to split YOLOs images and their associated labels into separate sets for training and testing.


## Installation
```bash
pip install yolosplitter
```

## Uses
```python
from yolosplitter import YoloSplitter

# give directory path containing Image and Labels
ys = YoloSplitter(input_dir="MyDataset/")

# creates the dataframe
df = ys.from_mixed_dir(main_dir="mydataset/")

# saves the Images and labels in "new_dataset" dir. with data.yaml
ys.split_and_save(DF=df,output_dir="new_dataset",train_size=0.70)

```
```python
df = ys.from_mixed_dir(main_dir="mydataset/")
df
```
![from_mixed_dir](https://github.com/sandeshkharat87/yolo-splitter/assets/47347413/93347b2a-c245-4509-ab15-ae169f3680b3)

```python
df = ys.from_yolo_dir(image_dir="mydataset-splitted/train/images/",label_dir="mydataset-splitted/train/labels/")
df
```
![Uploading from_yolo_dir.png…]()




```python
# Dataframe contains Image names, Label names, annoations and class names.
# In the dataframe, we can observe the number of classes present in each image. 
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
