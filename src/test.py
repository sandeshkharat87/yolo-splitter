from yolosplitter import YoloSplitter
import logging


logging.getLogger().setLevel(logging.DEBUG)

dataset_path = 'xxxFolder'

ys = YoloSplitter(imgFormat=['.jpg', '.jpeg'], labelFormat=['.txt'] )

# If you have yolo-format dataset already on the system
df = ys.from_yolo_dir(input_dir=dataset_path,ratio=(0.8,0.2,0.0),return_df=True)

ys.save_split(dataset_path + '_splitted')