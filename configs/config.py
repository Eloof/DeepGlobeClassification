import os
from pathlib import Path

import pandas as pd

# Get the home directory path
HOME_DIR = Path.home()

# Read the class_dict.csv file into a Pandas DataFrame
class_dict = pd.read_csv(
    os.path.join(HOME_DIR, "DeepGlobeClassificationPr/data/Data/class_dict.csv")
)

# Convert the "name" column of the DataFrame to a dictionary and
# normalize the RGB values to integers and convert them to a list
class_rgb_values = dict(
    (tuple(value), class_ind)
    for class_ind, value in enumerate((class_dict.values[:, 1:] / 255).astype(int))
)
