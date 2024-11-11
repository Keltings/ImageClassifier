# AI Programming with Python Project

## Flower Classifier: Command Line Application

This project provides a command-line application for classifying flower species using a deep learning model built with PyTorch. The application consists of two main Python scripts: `train.py` for training the model and `predict.py` for making predictions using a trained model. The model can be customized and trained on a dataset of flower images, and the predictions can be displayed along with the top classes and their probabilities.

### Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Making Predictions](#making-predictions)
- [Dependencies](#dependencies)
- [License](#license)

### Installation

To use the Flower Classifier, follow these steps:

```bash
pip install -r requirements.txt
```
### Train Model
```bash
python train.py data_directory
```

#### Example usage
```bash
python train.py flowers_data/ --arch "vgg13" --learning_rate 0.005 --hidden_units 256 --epochs 10 --gpu --save_dir ./checkpoints
```

### Make pedictions
```bash
python predict.py image_path checkpoint.pth
```
#### Example usage
```bash
python predict.py flowers_data/test/image_04567.jpg checkpoints/checkpoint.pth --top_k 3 --category_names cat_to_name.json --gpu
```



