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

# use this function if Image & Labels are in the same folder 
df = ys.from_mixed_dir(input_dir="mydataset")

# If folder contains train test valid set already (yolo dataset)
df = ys.from_yolo_dir("mydataset")

# saves the Images and labels in "new_dataset" dir. with data.yaml file.
# change save=True  if you want to create new dataset
ys.split_and_save(DF=df,output_dir="new_datset", ratio=(0.7, 0.2,0.1 ) ,save=False,shuffle=False)

```
![split_and_save](https://github.com/sandeshkharat87/yolo-splitter/assets/47347413/5e5dc779-f28b-4439-bdbe-17ed7761f407)


![mixed](https://github.com/sandeshkharat87/yolo-splitter/assets/47347413/9a0e7601-9912-4665-bfc3-35ab828491a3)


```python
# YOLO directory contaning (train,valid,test) 
df = ys.from_yolo_dir("pot_holes")
df
```

![from_yolo](https://github.com/sandeshkharat87/yolo-splitter/assets/47347413/98096445-d988-4818-bfd6-83bf7a8220ea)




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
