import pandas as pd
import sys
import os


def convert_data_to_list(input_data, desired_field: str = None):
    if isinstance(input_data, list):
        return input_data
    try:
        if isinstance(input_data, int) or isinstance(input_data, float) or isinstance(input_data, bool):
            return [input_data]
        elif isinstance(input_data, pd.DataFrame):
            if len(input_data.columns) == 1:
                return list(input_data.values[:, 0])
            else:
                return input_data[desired_field].tolist()
        elif isinstance(input_data, dict):
            if len(input_data.keys()) == 1:
                return list(input_data.values())[0]
            else:
                return input_data[desired_field].tolist()
        else:
            return list(input_data)
    except:
        raise NameError("'desired_field' has not been chosen, or you chose incorrect one")

def list_to_file(input_list: list, file_path: str):
    with open(file_path, 'w') as file:
        for item in input_list:
            file.write(f"{item}\n")