"""
This is the demo code that uses hydra to access the parameters in under the directory config.

Author: Khuyen Tran
"""
import os
import hydra
import pandas as pd
from omegaconf import DictConfig

from utils import load_obj

@hydra.main(config_path="../config", config_name="main", version_base=None)
def process_data(config: DictConfig):
    """Function to process the data"""
    # tf.data.Dataset.zip(train_dataset, test_dataset).save(config.data.final)
    raw_dataset = pd.DataFrame()
    for obj in os.listdir(config.data.obj):
        v, f = load_obj("{}{}".format(config.data.obj, obj))
        v_dataframe = pd.DataFrame(v, columns = ["X", "Y", "Z"])
        v_dataframe['obj_name'] = obj
        raw_dataset = pd.concat([raw_dataset, v_dataframe])
    raw_dataset.to_csv(config.data.raw)
    print("The dataset shape:{}".format(raw_dataset))
    print(f"Process data using {config.data.raw}")
    print(f"Columns used: {config.process.use_columns}")


if __name__ == "__main__":
    process_data()
