# DeepGlobeClassification

## Data Description

The training data for the Land Cover Challenge consists of 803 satellite images in RGB format, each with dimensions of 2448x2448 pixels. These images have a pixel resolution of 50cm and were collected by DigitalGlobe's satellite.

The dataset is divided into the following subsets:
- **Training Data**: The training dataset contains a large number of satellite images.
- **Validation Data**: This subset contains 171 images that can be used for model validation.
- **Test Data**: The test dataset includes 172 images, but note that it lacks corresponding masks.

## Labels

Each satellite image in the dataset is paired with a mask image that provides land cover annotations. The masks are represented as RGB images, where different colors correspond to different land cover classes. The labels for the land cover classes are color-coded using the (R, G, B) format, as follows:

1. **Urban land**: RGB (0, 255, 255)
   - Description: Man-made, built-up areas with human artifacts. Please note that roads have been excluded from this category as they can be challenging to label accurately.

2. **Agriculture land**: RGB (255, 255, 0)
   - Description: Includes farms, planned (regular) plantations, cropland, orchards, vineyards, nurseries, ornamental horticultural areas, and confined feeding operations.

3. **Rangeland**: RGB (255, 0, 255)
   - Description: Covers any non-forest, non-farm, green land with grass.

4. **Forest land**: RGB (0, 255, 0)
   - Description: Encompasses any land with a certain tree crown density percentage plus clearcut areas.

5. **Water**: RGB (0, 0, 255)
   - Description: Represents rivers, oceans, lakes, wetlands, ponds, and other water bodies.

6. **Barren land**: RGB (255, 255, 255)
   - Description: Refers to mountainous terrain, rocks, deserts, beaches, and other areas with no significant vegetation.

7. **Unknown**: RGB (0, 0, 0)
   - Description: Encompasses clouds and other regions where the land cover is uncertain or not specified.


For more details and to access the dataset, please visit the [DeepGlobe Land Cover Classification Dataset on Kaggle](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset).

my_project/
│
├── configs/
│ ├── config.yaml
│ └── hyperparameters.yaml
│
├── data/
│ ├── train/
│ ├── validation/
│ └── test/
│
├── dataloader/
│ ├── data_loader.py
│ ├── preprocessing.py
│ └── transforms.py
│
├── mlartifacts/
│ ├── model_weights/
│ ├── logs/
│ └── figures/
│
├── mlruns/
│ ├── run1/
│ ├── run2/
│ └── ...
│
└── src/
├── model.py
├── train.py
├── infer.py
└── utils.py
