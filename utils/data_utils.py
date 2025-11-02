import json
import os
import random
from datasets import Dataset, load_dataset
from abc import ABC, abstractmethod

FIELD_DICT = {
    "MathInstruct": "math",
    "finance_alpaca": "finance",
    "code_alpaca_20k": "coding",
    "MedInstruct-52k": "medical",
}

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task in the field of {field}, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task in the field of {field}. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class DatasetPathFactory:
    def __init__(self, dataset_dir_path, dataset_name):
        self.dataset_dir_path = dataset_dir_path
        self.dataset_name = dataset_name
        
        self.dataset_name_split_list = dataset_name.split(".")
        assert len(self.dataset_name_split_list) == 2, "name split length is not 2"
        self.prefix = self.dataset_name_split_list[0]
        self.file_type = self.dataset_name_split_list[1]

    @property
    def dataset_path(self):
        return os.path.join(self.dataset_dir_path, self.dataset_name)

    def get_train_path(self, length):
        return os.path.join(self.dataset_dir_path, f"{self.prefix}_{length}-train.{self.file_type}")

    def get_valid_path(self, length):
        return os.path.join(self.dataset_dir_path, f"{self.prefix}_{length}-valid.{self.file_type}")
        


class DataSpliter:
    def __init__(self, dataset_dir_path, dataset_name, train_length, valid_length):
        self.path_factory = DatasetPathFactory(dataset_dir_path, dataset_name)
        self.train_length = train_length
        self.valid_length = valid_length
        
    def is_splited(self,):
        if os.path.exists(self.path_factory.get_train_path(self.train_length)) \
            and os.path.exists(self.path_factory.get_valid_path(self.valid_length)):
            return True
        return False

    def split(self,):
        if not self.is_splited():
            print("Spliting")
            with open(self.path_factory.dataset_path, "r") as f:
                dataset_list = json.load(f)
                if len(dataset_list) < train_length + valid_length:
                    raise ValueError("dataset length is not enough!")
                train_valid_sample_split_point = int(len(dataset_list) * train_length / (train_length + valid_length))
                
                import pdb
                pdb.set_trace()
                
                train_dataset_list = random.sample(dataset_list[:train_valid_sample_split_point], train_length)
                valid_dataset_list = random.sample(dataset_list[train_valid_sample_split_point:], valid_length)

            with open(self.path_factory.get_train_path(self.train_length), "w") as f:
                json.dump(train_dataset_list, f)
            with open(self.path_factory.get_valid_path(self.valid_length), "w") as f:
                json.dump(valid_dataset_list, f)     
        else:
            print("Existing")


def format_parse(message_map, dataset_name):
    message_map["field"] = FIELD_DICT[dataset_name]
    if "input" in message_map and message_map["input"] != "":
        prompt = PROMPT_DICT["prompt_input"]
    else:
        prompt = PROMPT_DICT["prompt_no_input"]
    return {"prompt": prompt.format_map(message_map), "completion": message_map["output"]}


def dataset_local_load(dataset_dir_path, dataset_name, train_length=5000, valid_length=500):
    path_factory = DatasetPathFactory(dataset_dir_path, dataset_name)
    path_factory.get_train_path(train_length)
    train_list = []
    if path_factory.prefix.startswith("c4"):
        with open(path_factory.get_train_path(train_length), "r") as f:
            for line in f:
                data = json.loads(line)
                train_list.append({"text": data["text"]})

    else:
        with open(path_factory.get_train_path(train_length), "r") as f:
            dataset_list = json.load(f)
            for data in dataset_list:
                if "source" in data and "/CoT/" not in data["source"]:
                    continue
                train_list.append(format_parse(data, path_factory.prefix))

    valid_list = []
    if path_factory.prefix.startswith("c4"):
        with open(path_factory.get_valid_path(valid_length), "r") as f:
            for line in f:
                data = json.loads(line)
                train_list.append({"text": data["text"]})

    else:
        with open(path_factory.get_valid_path(valid_length), "r") as f:
            dataset_list = json.load(f)
            for data in dataset_list:
                if "source" in data and "/CoT/" not in data["source"]:
                    continue
                train_list.append(format_parse(data, path_factory.prefix))


    return train_list, valid_list


if __name__ == "__main__":
    dataset_dir_path = "../dataset"
    dataset_name = "MedInstruct-52k.json"
    train_length = 5000
    valid_length = 500
    data_spliter = DataSpliter(dataset_dir_path, dataset_name, train_length, valid_length)
    data_spliter.split()

    train_list, valid_list = dataset_local_load(dataset_dir_path, dataset_name, train_length=train_length, valid_length=valid_length)
