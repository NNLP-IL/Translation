import pandas as pd
from typing import List
import yaml
import json
import pickle
import os


class FileLoader:
    SUPPORTED_FORMAT: List[str] = ['csv', 'xls', 'xlsx', 'txt', 'yaml', 'json']     
    
    @staticmethod
    def load_txt(txt_file: str):
        with open(txt_file, 'r') as file:
            lines = file.readlines()
        data = [line.strip() for line in lines]
        return data

    @staticmethod
    def load_config(file_path: str):
        # Function to load YAML configuration
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    @staticmethod
    def load_json(json_file: str):
        with open(json_file, 'r') as file:
            data = json.load(file)
        return data

    @staticmethod
    def load_pkl_model(model_path: str):
        with open(model_path, 'rb') as file:
            # Load the model from the file
            model = pickle.load(file)
        return model

    @classmethod
    def get_loader(cls, file_format: str):
        loader_map: dict = {
            "csv": pd.read_csv,
            "xls": pd.read_excel,
            "xlsx": pd.read_excel,
            "txt":  cls.load_txt,
            "yaml": cls.load_config,
            "json": cls.load_json
        }
        if not file_format in cls.SUPPORTED_FORMAT:
            raise ValueError(f"File Format `{file_format}` not found in loader - SUPPORTED_FORMAT")
        return loader_map[file_format]

    @classmethod
    def load_data(cls, uploaded_file: str):
        if os.path.isfile(uploaded_file):
            ext = uploaded_file.rsplit(".", 1)[-1].lower()
            file_loader = cls.get_loader(file_format=ext)
            return file_loader(uploaded_file)
        else:
            raise ValueError(f"{uploaded_file} doesn't exist")