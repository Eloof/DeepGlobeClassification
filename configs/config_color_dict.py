import os
import pandas as pd
from configs.config_path import HOME_DIR

color_dict = pd.read_csv(os.path.join(HOME_DIR, "DeepGlobeClassificationPr/data/Data/class_dict.csv"))
CLASSES = color_dict['name']
