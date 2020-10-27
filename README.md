# MalariaDetection_Vgg19

Dataset Link:   https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria

Detecting Malaria with Deep Learning:

Abstract
Malaria is a fatal disease, continues to be a major burden on global health. Malaria is caused by plasmodium parasites that were injected by a certain female mosquito. About half million deaths were caused by malaria every year. The way in which people are diagnosed for malaria is microscopic examination, by observing blood smears under a microscope. However, these techniques are accurate but they are very time consuming, in contrast, using computer vision and deep learning techniques, the proposed system is automated. In this work, we developed an automated and robust diagnosis system to detect malaria parasites. We used Region based Fully Convolutional Neural Network (RFCN) object detection model for detection and classification of malaria parasites. For this work we used labeled dataset with bounding boxes of approximately 1328 images and demonstrated that our work outperforms baseline methods.

Overview
Malaria is a mosquito-borne infectious disease that affects humans and other animals. Malaria causes symptoms that typically
include fever, tiredness, vomiting, and headaches. In severe cases it can cause yellow skin, seizures, coma, or death.Malaria is typically diagnosed by the microscopic examination of blood using blood films, or with antigen-based rapid diagnostic tests. So, I've used segmented cells from the thin blood smear slide images to classify whether a person has malaria or not.


Project Details
I have used Keras (an open source neural network library) to build this classifier. The model (Model.ipynb) contains a self- made ConvNet (Convolutional Neural Network) that has around 4 convlutional layers, 4 Max Pool layers and 2 dense layers having 'relu' activation unit. The last layer has the 'sigmoid' activation unit to classify cell images having Malaria or not. It has around 336017 trainable parameters.

I have got around 96% accuracy in the Test set.


Dataset
The dataset contains two folders

Infected
Uninfected
And a total of 27,558 images.

Datatset Source : Kaggle

In order to use this model for this dataset, you have to organize the data in a particular format.


Inspiration
Save humans by detecting Image Cells that contain Malaria or not
