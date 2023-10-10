import pandas as pd
import os
import tqdm
import shutil as sh
import yaml


class YoloSplitter():
    """
    imgFormat:[".jpg", ".jpeg", ".png"]
    Give image format in the list

    lableFormat:[".txt"]
    Give label format in th list
    """
    def __init__(self,imgFormat= [".jpg", ".jpeg", ".png"],labelFormat=[".txt"]):
        self.imgFormat = imgFormat
        self.labelFormat = labelFormat
        self.__DATAFRAME = None
        self.__error_files = []
        self.__req_cols = ['images', 'labels',  'annots', 'cls_names', 'set', 'new_set']
        
    def from_mixed_dir(self,input_dir,ratio=(0.70,0.20,0.10)):
        """
        input_dir : Provide directory path
        ratio: rato of split train/val/test (0.70,0.20,0.10)
        """
        self.__DATAFRAME = None
        self.__error_files = []

        dataset = self.get_data(image_dir=input_dir, label_dir=input_dir)
        input_df = pd.DataFrame.from_dict(dataset).copy()
        input_df["set"] = ""
        
        
        if self.__error_files:
            print(f"Total Error Files: {len(self.__error_files)} ")
    
        # Splitted df (train/test/val)
        splitted_df = self.__make_split(input_df,ratio=ratio)
        
        
        self.__DATAFRAME = splitted_df
        
        return self.__DATAFRAME[self.__req_cols]
        
    
    def from_yolo_dir(self,input_dir,ratio=(0.70,0.20,0.10)):
        """
        input_dir : Provide directory path
        ratio: rato of split train/val/test (0.70,0.20,0.10)
        """
        self.__DATAFRAME = None
        self.__error_files = []
        self.__input_dir = input_dir
        
        set_dir_names = [i  for i in  os.listdir(input_dir) if i in  ["train","test","valid"]]        
        all_dataframes = []
        
        for folder_name in set_dir_names:
            os.path.join(input_dir,folder_name)
            image_dir = os.path.join(input_dir,folder_name,"images")
            label_dir = os.path.join(input_dir,folder_name,"labels")
            dataset = self.get_data(image_dir=image_dir, label_dir=label_dir)
            temp_df = pd.DataFrame.from_dict(dataset)
            temp_df["set"] = folder_name
            all_dataframes.append(temp_df)
            
        if self.__error_files:
            print(f"Total Error Files: {len(self.__error_files)} ")
        
        input_df = pd.concat(all_dataframes,ignore_index=True,sort=False)
        
        # splitted df (train/test/val)
        splitted_df = self.__make_split(input_df,ratio=ratio)
        
        self.__DATAFRAME = splitted_df
        
        return splitted_df[self.__req_cols]
        
    
    def get_data(self,image_dir,label_dir):
        dataset = {"images":[], "labels":[], "images_path":[], "labels_path":[] , "annots":[],"cls_names":[] }
        
        All_Images = [i for i in os.listdir(
            f"{image_dir}") if os.path.splitext(i)[-1] in self.imgFormat]
        All_Labels = [i for i in os.listdir(
            f"{label_dir}") if os.path.splitext(i)[-1] in self.labelFormat]

        for iname in All_Images:
            for lname in All_Labels:
                if os.path.splitext(iname)[0] == os.path.splitext(lname)[0]:
                    try:
                        annot_data,cls_names = self.__read_annot(os.path.join(label_dir,lname))
                        dataset["images_path"].append(os.path.join(image_dir,iname))
                        dataset["labels_path"].append(os.path.join(label_dir,lname))
                        dataset["images"].append(os.path.join(iname))
                        dataset["labels"].append(os.path.join(lname))
                        dataset["annots"].append(annot_data)
                        dataset["cls_names"].append(cls_names)
                    except Exception as e:
                        self.__error_files.append([os.path.join(label_dir,lname),e])
                else:
                    continue

        return dataset
    
    
    def __read_annot(self,path):
        annot_data = []
        all_cls_names = []
        with open(path,"r") as f:
            f_data = f.read().split("\n")
            for i in f_data:
                i = i.split(" ")
                cls_name = int(i[0])
                cls_annot = [float(i) for i in i[1:]]
                annot_data.append([cls_name,cls_annot])
                all_cls_names.append(cls_name)
        return annot_data,list(set(all_cls_names))
    
    
    def __make_split(self,input_df,ratio):
        
        if round(sum(ratio),5)!=1:
            raise ValueError("Ratio sum should be equal to 1")
        
        if len(ratio)==2:
            train_ratio,val_ratio = ratio
            test_ratio = 0
        
        elif len(ratio) == 3:
            train_ratio,val_ratio,test_ratio = ratio
        
        else:
            raise ValueError("ratio must be tuple length of 2 or 3")
            
        total_length = len(input_df)
        train_length = round(train_ratio * len(input_df))
        val_length = round(val_ratio * total_length)
        test_length = total_length-train_length-val_length

        train_df = input_df.iloc[:train_length].copy()
        val_df = input_df.iloc[train_length:train_length+val_length].copy()
        test_df = input_df.iloc[train_length + val_length:].copy()

        # Shuffle new_set column to new set
        set_names = ['train'] * train_length + ['valid'] * val_length + ['test'] * test_length
        random.shuffle(set_names)
        
        train_df["new_set"] = set_names[:train_length]
        val_df["new_set"] = set_names[train_length:train_length + val_length]
        test_df["new_set"] = set_names[train_length + val_length:]

        
        splitted_df = pd.concat([train_df,val_df,test_df],ignore_index=True,sort=False)
        print(f"\nTrain size:{train_length},Validation size:{val_length},Test size :{test_length}\n")

        
        return splitted_df
    
    @property
    def show_errors(self):    
        return self.__error_files
    
    @property
    def show_dataframe(self):
        return self.__DATAFRAME[self.__req_cols]
    
        


    def save_split(self,output_dir):
        """
        output_dir: Oupt dir path
        """
        if os.path.exists(output_dir):
            raise FileExistsError("Folder already exists ...")

        input_df = self.__DATAFRAME
        
        print(f"Saving New split in '{output_dir}' dir")

        if input_df is None:
            raise ValueError("Dataframe is not created. Plase ran from_yolo_dir or from_mixed_dir first")
        
        
        for idx,row in tqdm.tqdm(input_df.iterrows(),total=len(input_df)):
            os.makedirs(os.path.join(output_dir,row["new_set"],"images"), exist_ok=True)
            os.makedirs(os.path.join(output_dir,row["new_set"],"labels"), exist_ok=True)
            
            
            #Images
            input_image_name = os.path.join(row["images_path"])
            output_image_name = os.path.join(output_dir,row["new_set"],"images",row["images"])
            #labels
            input_label_name = os.path.join(row["labels_path"])
            output_label_name = os.path.join(output_dir,row["new_set"],"labels",row["labels"])
            
            #Copying Images --
            sh.copy2(input_image_name,output_image_name)
            #Copying labels --
            sh.copy2(input_label_name,output_label_name)
        
        # SAVING YAML FILE
        yamlFile = {"train": "", "nc": 0, "names": ""}
        flat_list = [item for sublist in input_df["cls_names"].to_list() for item in sublist]
        cls_names = list(set(flat_list))
        train_set,valid_set,test_set = [self.__DATAFRAME.new_set.value_counts().to_dict()[i] if i in self.__DATAFRAME.new_set.value_counts().to_dict()  else 0 for i in ["train","valid","test"]]

        yamlFile["nc"] = len(cls_names)
        yamlFile["names"] = cls_names
        yamlFile["train"] = os.path.join(output_dir, "train")
        if valid_set != 0:
            yamlFile["val"] = os.path.join(output_dir, "val")
        if test_set != 0:
            yamlFile["test"] = os.path.join(output_dir, "test")

        with open(os.path.join(output_dir, "data.yaml"), "w") as f:
            yaml.dump(yamlFile, f, indent=2)

