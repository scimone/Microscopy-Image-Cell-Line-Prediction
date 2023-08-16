# Cell Line Prediction from Microscopy Images
This notebook focuses on predicting the cell line of given microscopy images using a deep learning approach. Each microscopy image is composed of three separate `.png` images (channels) representing the nucleus, microtubules, and endoplasmic reticulum for each cell. The original files have not been included in this repository for copyright reasons.

## Approach
The approach for this task involves treating the combined channels as RGB images, augmenting them, and utilizing them as training inputs for the `efficientnet-b1` model, a pre-trained Convolutional Neural Network (CNN).

## 1. Preprocessing
Custom `ImageDataset` Class: A custom `ImageDataset` class, inherited from `torch.utils.data.Dataset`, is developed to manage data loading and preprocessing. This class is later utilized by the DataLoader.

In this class:
- The three channels per image are loaded and stacked to create an RGB image. These images are stored in the `self.images` list.
- Labels are obtained from a label `.csv` file, where each image ID is associated with a cell line. Labels are tokenized as integers (`0 to 8`) and stored in the `self.labels` list.
- Stratified train-test split (`70%` training and `30%` validation) is applied to the training images to create training and validation data.
- The `__getitem__` method retrieves a training image along with its label, applies transformations (augmentation) to the image, converts it to a torch tensor, resizes it to match the original training data size of the pretrained network, and normalizes it using mean and standard deviation values calculated from the training data. Validation images are only resized and normalized.

## 2. Model
The model employed in this project is the pre-trained `efficientnet-b1`. EfficientNet is a CNN architecture and scaling method that uniformly scales depth, width, and resolution. The last layer of the model is adjusted to have an output dimension of 9, corresponding to the number of possible cell lines (classes).

## 3. Training
- Training Setup: Training involves using the Adam optimizer with a weight decay of `1e-6` and the cross-entropy loss as the loss function.
- Training Parameters: The model is trained for `30` epochs with a learning rate of `7e-5` and a batch size of `64`. The entire training process takes approximately `1.5` hours.
- Validation Performance: The final validation loss achieved is `0.147`, and the locally computed balanced accuracy score of the validation set is `0.94`.

## 4. Submission Predictions
- Submission Process: Submission images are processed through the custom ImageDataset class, resized, and normalized. The model predictions are obtained by iterating through the submission dataloader.
- Label Decoding: The predicted labels are decoded from tokens to strings and saved in a `.csv` file for submission.
