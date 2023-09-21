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
df = ys.create_dataframe()

# saves the Images and labels in "new_dataset" dir. with data.yaml
ys.split_and_save_project(DF=df,output_dir="MyDataset-splitted",train_size=0.70)

```
```
!ls MyDataset-splitted
---------------------
data.yaml  train  val
```

```python
# Dataframe contains Image names, Label names, annoations and class names.
# In the dataframe, we can observe the number of classes present in each image. 
```
![2023-09-21_19-22](https://github.com/sandeshkharat87/yolo-splitter/assets/47347413/bba5a794-3f4b-47d2-bb77-65b06b135f15)




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
