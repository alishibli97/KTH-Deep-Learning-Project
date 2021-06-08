# DD2424-Deep-Learning-Project

## Requirements
Python 3.8 or later with all [requirements.txt](https://github.com/alishibli97/KTH-Deep-Learning-Project/blob/main/requirements.txt).

`pip install -r requirements.txt`


The aim of this project is to apply state of the art image segmentation analysis deep learning techniques to agricultural image segmentation tasks. Specifically, we apply the deep convolution architecture UNet for agricultural application. Typically the UNet model have been trained on small datasets, consisting of less than 100 input pictures. In this paper, we present our results and conclusions when the model is applied on large datasets with over 10000 images. We received  mixed results from our experiments. First, the model is very efficient on small datasets even with very low quality input images. Second, it is possible to train on large datasets, albeit the model is computationally heavy to train and requires carefully tuning of settings. We also find that it is possible to combine use the model not only for image segmentation but also for classification of objects in the pictures. Using the model for image classification is harder than segmentation and requires more carefully selection of parameters. Training the model on large datasets proved to be very computationally heavy, requiring multiple high end GPUs and many hours. However, it only took seconds to run the trained model on unseen images.
