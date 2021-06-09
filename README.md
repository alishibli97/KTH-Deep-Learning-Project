# DD2424-Deep-Learning-Project

## Abstract
The aim of this project is to apply state of the art image segmentation analysis deep learning techniques to agricultural image segmentation tasks. Specifically, we apply the deep convolution architecture UNet for agricultural application. Typically the UNet model have been trained on small datasets, consisting of less than 100 input pictures. In this work, we present our results and conclusions when the model is applied on large datasets with over 10000 images. We received  mixed results from our experiments. First, the model is very efficient on small datasets even with very low quality input images. Second, it is possible to train on large datasets, albeit the model is computationally heavy to train and requires carefully tuning of settings. We also find that it is possible to combine use the model not only for image segmentation but also for classification of objects in the pictures. Using the model for image classification is harder than segmentation and requires more carefully selection of parameters. Training the model on large datasets proved to be very computationally heavy, requiring multiple high end GPUs and many hours. However, it only took seconds to run the trained model on unseen images.

The model is implemented from scratch following the original paper. The dataset we train on is from a competition for agricultural image segmentation, a task for detecting damaged areas in crops. Both resources can be found in the references.


## Requirements
Python 3.8 or later with all [requirements.txt](https://github.com/alishibli97/KTH-Deep-Learning-Project/blob/main/requirements.txt) dependencies installed. To install run:

`pip install -r requirements.txt`


## Training
`train.py` runs the training pipeline and save the model every 3 epochs by default. You can run it as:
```
python train.py 
      --data_dir directory_to_dataset (required)
      --classes number_of_classes (required)
      --learning_rate learning_rate (default = 1e-3)
      --epochs number_of_epochs (default = 10)
      --batch batch_size (default = 8)
      --iter number_of_batches_before_saving_the_model (default = 3)
```


## Pretranined weights links

You can find pretrained weights and history for accuracy/loss using the following links. We save the model every 3 epochs for 18 complete epochs, for 12 hours of training on 10 1080 Ti GPUs. Thus you can complete the links by insert any value from {0,3,6,9,12,15,18}:

https://deeplearningweights.s3.amazonaws.com/trained_model_{}.pth

https://deeplearningweights.s3.amazonaws.com/history_{}.txt


## References
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

[AGRICULTURE-VISION: CHALLENGES & OPPORTUNITIES FOR COMPUTER VISION IN AGRICULTURE](https://www.agriculture-vision.com/agriculture-vision-2021)
