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

Project Organization
------------

    DeepGlobeClassificationPr/
      ├── configs/
      │   ├── config.py              <- Config to genereate homedir and classRGBvalues
      │   └── config_train_test.py   <- Config with train and test augmentation
      ├── data/
      │   └── data_download.py       <- Script to download data from Kaggle(need access token)
      ├── dataLoader/
      │   ├── data_loader.py         <- Script to preprocess and load data
      │   └── data_module.py         <- Script to generate train/val datatoader
      ├── mlartifacts/
      ├── mlruns/
      └── src/
      │   └── CustomUnetNN/
      │      ├── fine_tune_model.py     <- Script to fineTune model from best checkpoint
      │      ├── train.py               <- Script to train
      │      ├── unet_nn.py             <- Class with NNUnet
      │      └── utils.py               <- Class with logging and configure with pytorch_lightning
      │   └── Production/
      │      ├── predict.py             <- Inference script
      


--------
# Metrics

 **IoU** (Intersection over Union)
- **Измерение качества сегментации**: IoU позволяет измерять, насколько точно модель сегментирует объекты на изображении. Чем выше значение IoU, тем лучше сегментация.

- **Интерпретация**: Значение IoU выражается в процентах, что делает его легко интерпретируемым.

**F1-мера** 
- **Учет ошибок**: F1-мера учитывает как ложные позитивы (ложные обнаружения), так и ложные негативы (пропущенные объекты). Это важно для задач сегментации, где мы стремимся минимизировать оба типа ошибок.
- **Баланс несбалансированных классов**: Многоклассовая сегментация может столкнуться с проблемой несбалансированных классов, где одни классы встречаются чаще, чем другие.
- **Простая интерпретация**: F1-мера представляет собой одно число, которое объединяет точность и полноту, что делает ее интерпретируемой.



