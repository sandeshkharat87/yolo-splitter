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

    def __init__(self, imgFormat=[".jpg", ".jpeg", ".png"], labelFormat=[".txt"]):
        self.imgFormat = imgFormat
        self.labelFormat = labelFormat
        self.__input_dir = None
        self.__overwrite = False

    def from_mixed_dir(self, input_dir):
        """
        input_dir = Input dir path
        Path containing both Images and Labels in single directory.
        """
        self.__input_dir = input_dir
        dataset = self.get_data(image_dir=input_dir, label_dir=input_dir)
        df = pd.DataFrame.from_dict(dataset)
        df["set"] = ""
        return df

    def from_yolo_dir(self, input_dir):
        """
        input_dir: folder path containing train,val,test subfolders
        """
        self.__input_dir = input_dir
        set_dir_names = [i for i in os.listdir(input_dir) if i in [
            "train", "test", "valid", "val"]]
        all_dataframes = []

        for folder_name in set_dir_names:
            os.path.join(input_dir, folder_name)
            image_dir = os.path.join(input_dir, folder_name, "images")
            label_dir = os.path.join(input_dir, folder_name, "labels")
            dataset = self.get_data(image_dir=image_dir, label_dir=label_dir)
            temp_df = pd.DataFrame.from_dict(dataset)
            temp_df["set"] = folder_name
            all_dataframes.append(temp_df)

        return pd.concat(all_dataframes, ignore_index=True, sort=False)

    def get_data(self, image_dir, label_dir):
        dataset = {"images": [], "labels": [], "annots": [], "cls_names": []}

        All_Images = [i for i in os.listdir(
            f"{image_dir}") if os.path.splitext(i)[-1] in self.imgFormat]
        All_Labels = [i for i in os.listdir(
            f"{label_dir}") if os.path.splitext(i)[-1] in self.labelFormat]

        for iname in All_Images:
            for lname in All_Labels:
                if os.path.splitext(iname)[0] == os.path.splitext(lname)[0]:
                    annot_data, cls_names = self.__read_annot(
                        os.path.join(label_dir, lname))
                    dataset["images"].append(iname)
                    dataset["labels"].append(lname)
                    dataset["annots"].append(annot_data)
                    dataset["cls_names"].append(cls_names)
                else:
                    continue

        return dataset

    @property
    def overwrite(self):
        # self.__overwrite = value
        return self.__overwrite

    @overwrite.setter
    def overwrite(self, value):
        """
        set value=True if you want to ovewrite `output_dir` while using `split_and_save method`
        """
        self.__overwrite = value
        print(self.__overwrite,
              "\nCaution: The code can now overwrite an existing folder.")
        return self.__overwrite

   # read annotation from label files
    def __read_annot(self, path):
        annot_data = []
        all_cls_names = []
        with open(path, "r") as f:
            f_data = f.read().split("\n")
            for i in f_data:
                i = i.split(" ")
                cls_name = int(i[0])
                cls_annot = [float(i) for i in i[1:]]
                annot_data.append([cls_name, cls_annot])
                all_cls_names.append(cls_name)
        return annot_data, list(set(all_cls_names))

    # split and save the dataframe into train, validation, and test sets.
    def split_and_save(self, DF, output_dir, ratio=(0.70, 0.20, 0.10), save=False, shuffle=False):
        """
        DF: dataframe (pandas dataframe)
        Get datafrmae from 'from_mixed_dir' or 'from_yolo_dir'

        outputdir: Directory name
        New folders and files will be saved to output dir

        ratio: (0.70,0.20,0.10) 
        ratio of train=0.70,val=0.20,test=0.10
        """

        if os.path.exists(output_dir) and self.__overwrite == False:
            raise FileExistsError(f"Folder:{output_dir} already exist")

        # reset
        self.__overwrite = False

        # YAML file
        yamlFile = {"train": "", "nc": 0, "names": ""}

        flat_list = [item for sublist in DF["cls_names"].to_list()
                     for item in sublist]
        cls_names = list(set(flat_list))

        # Ratio

        if round(sum(ratio), 5) != 1:
            raise ValueError("Ratio sum should be equal to 1")

        if len(ratio) == 2:
            train_ratio, val_ratio = ratio
            test_ratio = 0

        elif len(ratio) == 3:
            train_ratio, val_ratio, test_ratio = ratio

        else:
            raise ValueError("ratio must be tuple length of 2 or 3")

        if shuffle:
            DF = DF.sample(frac=1.0, random_state=42)

        total_length = len(DF)
        train_length = round(train_ratio * len(DF))
        val_length = round(val_ratio * total_length)
        test_length = total_length-train_length-val_length

        if test_ratio == 0:
            val_length += test_length
            test_length = 0

        train_df = DF.iloc[:train_length].copy()
        val_df = DF.iloc[train_length:train_length+val_length].copy()
        test_df = DF.iloc[train_length + val_length:].copy()

        train_df["set"] = "train"
        val_df["set"] = "valid"
        test_df["set"] = "test"

        new_df = pd.concat([train_df, val_df, test_df],
                           ignore_index=True, sort=False)

        print(
            f"Train size:{train_length} ,Validation size:{val_length} ,Test size :{test_length}")

        if save:
            for df_name, set_dir_name in [(train_df, "train"), (val_df, "valid"), (test_df, "test")]:
                if len(df_name) != 0:
                    output_image_dir = os.path.join(
                        output_dir, set_dir_name, "images")
                    output_label_dir = os.path.join(
                        output_dir, set_dir_name, "labels")

                    os.makedirs(output_image_dir, exist_ok=True)
                    os.makedirs(output_label_dir, exist_ok=True)
                    # saving in theire respective folders
                    for idx, row in tqdm.tqdm(df_name.iterrows(), total=len(df_name), desc=set_dir_name):
                        input_image_dir = os.path.join(
                            self.__input_dir, row["set"], row["images"])
                        input_label_dir = os.path.join(
                            self.__input_dir, row["set"], row["labels"])

                        # Copying Images
                        sh.copy2(src=input_image_dir, dst=os.path.join(
                            output_image_dir, row["images"]))

                        # Copying Labels
                        sh.copy2(src=input_label_dir, dst=os.path.join(
                            output_label_dir, row["labels"]))

            # Saving YAML File
            yamlFile["nc"] = len(cls_names)
            yamlFile["names"] = cls_names
            yamlFile["train"] = os.path.join(output_dir, "train")
            if len(val_df) != 0:
                yamlFile["val"] = os.path.join(output_dir, "val")
            if len(test_df) != 0:
                yamlFile["test"] = os.path.join(output_dir, "test")

            with open(os.path.join(output_dir, "data.yaml"), "w") as f:
                yaml.dump(yamlFile, f, indent=2)

        return new_df
