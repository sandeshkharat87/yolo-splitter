import pandas as pd
import os
import tqdm
import shutil as sh
import yaml



class YoloSplitter():
    """
    imgFormat = [".jpg", ".jpeg", ".png",]
    Give new extensions in list ex. [ ".tiff", ".jpg", ".jpeg", ".png",]

    labelFormat = [".txt"]
    """
    def __init__(self,imgFormat= [".jpg", ".jpeg", ".png"],labelFormat=[".txt"]):
        self.imgFormat = imgFormat
        self.labelFormat = labelFormat
        self.__image_dir = None
        self.__label_dir = None


    def from_mixed_dir(self,main_dir):
        """
        main_dir = Main dir path
        Path containing both Images and Labels in single directory.
        """
        self.__image_dir = main_dir
        self.__label_dir = main_dir
        dataset = self.get_data(image_dir=self.__image_dir, label_dir=self.__label_dir)
        df = pd.DataFrame.from_dict(dataset)
        return df


    def from_yolo_dir(self,image_dir,label_dir):
        """
        image_dir = image dir path
        label_dir = label dir path
        """
        self.__image_dir = image_dir
        self.__label_dir = label_dir
        dataset = self.get_data(image_dir=self.__image_dir, label_dir=self.__label_dir)
        df = pd.DataFrame.from_dict(dataset)
        return df

    def get_data(self,image_dir,label_dir):
        dataset = {"images":[], "labels":[], "annots":[],"cls_names":[] }


        All_Images = [i for i in os.listdir(
            f"{image_dir}") if os.path.splitext(i)[-1] in self.imgFormat]
        All_Labels = [i for i in os.listdir(
            f"{label_dir}") if os.path.splitext(i)[-1] in self.labelFormat]

        for iname in All_Images:
            for lname in All_Labels:
                if os.path.splitext(iname)[0] == os.path.splitext(lname)[0]:
                    annot_data,cls_names = self.__read_annot(label_file_name=lname)
                    dataset["images"].append(iname)
                    dataset["labels"].append(lname)
                    dataset["annots"].append(annot_data)
                    dataset["cls_names"].append(cls_names)
                else:
                    continue


        return dataset

   # Read annotation from label files
    def __read_annot(self,label_file_name):
        annotation_file_name = os.path.join(self.__label_dir,label_file_name)
        annot_data = []
        all_cls_names = []
        with open(annotation_file_name,"r") as f:
            f_data = f.read().split("\n")
            for i in f_data:
                i = i.split(" ")
                cls_name = int(i[0])
                cls_annot = [float(i) for i in i[1:]]
                annot_data.append([cls_name,cls_annot])
                all_cls_names.append(cls_name)
        return annot_data,list(set(all_cls_names))

    def split_and_save(self,DF,output_dir,train_size=0.70,overwrite=False):
        """
        DF: dataframe (pandas dataframe)
        Get datafrmae from 'from_mixed_dir' or 'from_yolo_dir'

        outputdir: Directory name
        New files will be saved to output dir

        train_size: 0.70
        Dataframe will be splitted in (70% + 30%) (train+val)

        overwrite: False
        True if you want to overwrite the existing files and folders
        """
        if not overwrite:
            if os.path.exists(output_dir):
                raise Exception("Folder Already Exixts !!!")
        else:
            pass

        # YAML file
        yamlFile = {"train": "", "val": "", "nc": 0, "names": ""}

        flat_list = [item for sublist in DF["cls_names"].to_list() for item in sublist]
        cls_names = list(set(flat_list))

        train_length = int(train_size * len(DF))
        val_length = len(DF) - train_length

        print(f"Train size: {train_length}, Validation size: {val_length}")

        train_df = DF.iloc[:train_length]
        val_df = DF.iloc[train_length:]


        for df_name, dir_name in [(train_df,"train"), (val_df,"val")]:
            image_dir = os.path.join(output_dir,dir_name,"images")
            label_dir = os.path.join(output_dir,dir_name,"labels")

            os.makedirs(image_dir,exist_ok=True)
            os.makedirs(label_dir,exist_ok=True)
            # saving in theire respective folders
            for idx,row in tqdm.tqdm(df_name.iterrows(),total=len(df_name),desc=dir_name):

                # Copying Images
                sh.copy2(src=os.path.join(self.__image_dir, row["images"])
                         ,dst = os.path.join(image_dir,row["images"]) )

                # Copying Labels
                sh.copy2(src=os.path.join(self.__label_dir, row["labels"])
                         ,dst = os.path.join(label_dir,row["labels"]))


        # Saving YAML File
        yamlFile["nc"] = len(cls_names)
        yamlFile["names"] = cls_names
        yamlFile["train"] = os.path.join(output_dir, "train")
        yamlFile["val"] = os.path.join(output_dir, "val")

        with open(os.path.join(output_dir, "data.yaml"), "w") as f:
            yaml.dump(yamlFile, f, indent=2)


