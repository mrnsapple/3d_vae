"""
This is the demo code that uses hydra to access the parameters in under the directory config.

Author: Khuyen Tran
"""
import os
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

def load_obj(fn):
    fin = open(fn, "r")
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []
    faces = []
    for line in lines:
        if line.startswith("v "):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith("f "):
            faces.append(np.int32([item.split("/")[0] for item in line.split()[1:4]]))

    f = np.vstack(faces)
    v = np.vstack(vertices)
    return v, f


def export_obj(out, v, f):
    with open(out, "w") as fout:
        for i in range(v.shape[0]):
            fout.write("v %f %f %f\n" % (v[i, 0], v[i, 1], v[i, 2]))
        for i in range(f.shape[0]):
            fout.write("f %d %d %d\n" % (f[i, 0], f[i, 1], f[i, 2]))


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
