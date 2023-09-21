import pandas as pd
import shutil as sh
import os
import yaml
import  tqdm


class YoloSplitter():
    """
    root: Input Directory
    """
    def __init__(self,input_dir):
        self.input_dir = input_dir

    def create_dataframe(self,imgFormat=None,labelFormat=None):
        """
        Returns DataFrame containing images,labels,annotations,classes
        By default Image fromats are [".jpg", ".jpeg", ".png"]
        By default Label fromats are [".txt"]

        """
        if imgFormat is None:
            imgFormat = [".jpg", ".jpeg", ".png"]

        if labelFormat is None:
            labelFormat = [".txt"]





        # Collect all images and labels
        All_Images = [i for i in os.listdir(
            f"{self.input_dir}/") if os.path.splitext(i)[-1] in imgFormat]
        All_Labels = [i for i in os.listdir(
            f"{self.input_dir}/") if os.path.splitext(i)[-1] in labelFormat]

        # match image name with label name if matches collect there annotations
        # saved them in the dict.
        dataset = {"images":[], "labels":[], "annots":[],"cls_names":[] }
        for iname in All_Images:
            for lname in All_Labels:
                if os.path.splitext(iname)[0] == os.path.splitext(lname)[0]:
                    annot_data,cls_names = self.__read_annot(value=lname)
                    dataset["images"].append(iname)
                    dataset["labels"].append(lname)
                    dataset["annots"].append(annot_data)
                    dataset["cls_names"].append(cls_names)
                else:
                    continue
        return pd.DataFrame.from_dict(dataset)

    # Read annotation from label files
    def __read_annot(self,value):
        annot_data = []
        all_cls_names = []
        with open(self.input_dir+"/"+value,"r") as f:
            f_data = f.read().split("\n")
            for i in f_data:
                i = i.split(" ")
                cls_name = int(i[0])
                cls_annot = [float(i) for i in i[1:]]
                annot_data.append([cls_name,cls_annot])
                all_cls_names.append(cls_name)
        return annot_data,list(set(all_cls_names))



    def split_and_save_project(self,DF,output_dir,train_size=0.70,force=False):
        """
        DF = Dataframe created from `def create_dataframe()`
        output_dir = Output Dir || (train,val,data.yaml) will be saved in Output Dir
        train_size = 0.70 || split dataframe in to (train:0.70 + val:0.30) if train_size = 0.60 then (train:0.60 + val:0.40)
        force = True || To Overwrite the existing Output Dirs (train,val,data.yaml)
        """
        if not force:
            if os.path.exists(output_dir):
                raise FileExistsError("Folder already exists ... ")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Writing Data In -> {output_dir}")

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
                sh.copy2(src=os.path.join(self.input_dir, row["images"])
                         ,dst = os.path.join(image_dir,row["images"]) )

                # Copying Labels
                sh.copy2(src=os.path.join(self.input_dir, row["labels"])
                         ,dst = os.path.join(label_dir,row["labels"]))


        # Saving YAML File
        yamlFile["nc"] = len(cls_names)
        yamlFile["names"] = cls_names
        yamlFile["train"] = os.path.join(output_dir, "train/")
        yamlFile["val"] = os.path.join(output_dir, "val/")

        with open(os.path.join(output_dir, "data.yaml"), "w") as f:
            yaml.dump(yamlFile, f, indent=2)



