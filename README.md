# MalariaDetection_Vgg19

Detecting Malaria with Deep Learning:

Abstract
Malaria is a fatal disease, continues to be a major burden on global health. Malaria is caused by plasmodium parasites that were injected by a certain female mosquito. About half million deaths were caused by malaria every year. The way in which people are diagnosed for malaria is microscopic examination, by observing blood smears under a microscope. However, these techniques are accurate but they are very time consuming, in contrast, using computer vision and deep learning techniques, the proposed system is automated. In this work, we developed an automated and robust diagnosis system to detect malaria parasites. We used Region based Fully Convolutional Neural Network (RFCN) object detection model for detection and classification of malaria parasites. For this work we used labeled dataset with bounding boxes of approximately 1328 images and demonstrated that our work outperforms baseline methods.


Deep Learning Model Training Phase
In the model training phase, we will build several deep learning models and train them on our training data and compare their performance on the validation data. We will then save these models and use them later on again in the model evaluation phase.


Model 1: CNN from Scratch
Our first malaria detection model will be building and training a basic convolutional neural network (CNN) from scratch. First let’s define our model architecture.

Based on the architecture in the preceding code, our CNN model has three convolution and pooling layers followed by two dense layers and dropout for regularization. Let’s train our model now!

We get a validation accuracy of 95.6% which is pretty good, though our model looks to be overfitting slightly looking at our training accuracy which is 99.9%. We can get a clear perspective on this by plotting the training and validation accuracy and loss curves.


Ideas for deep transfer learning
For the purpose of this article, the idea is, can we leverage a pre-trained deep learning model (which was trained on a large dataset — like ImageNet) to solve the problem of malaria detection by applying and transferring its knowledge in the context of our problem?
We will apply the two most popular strategies for deep transfer learning.


Pre-trained Model as a Feature Extractor
Pre-trained Model with Fine-tuning
We will be using the pre-trained VGG-19 deep learning model, developed by the Visual Geometry Group (VGG) at the University of Oxford, for our experiments. A pre-trained model like the VGG-19 is an already pre-trained model on a huge dataset (ImageNet) with a lot of diverse image categories. Considering this fact, the model should have learned a robust hierarchy of features, which are spatial, rotation, and translation invariant with regard to features learned by CNN models. Hence, the model, having learned a good representation of features for over a million images, can act as a good feature extractor for new images suitable for computer vision problems just like malaria detection! Let’s briefly discuss the VGG-19 model architecture before unleashing the power of transfer learning on our problem.

Understanding the VGG-19 model
The VGG-19 model is a 19-layer (convolution and fully connected) deep learning network built on the ImageNet database, which is built for the purpose of image recognition and classification. This model was built by Karen Simonyan and Andrew Zisserman and is mentioned in their paper titled ‘Very Deep Convolutional Networks for Large-Scale Image Recognition’. I recommend all interested readers to go and read up on the excellent literature in this paper. The architecture of the VGG-19 model is depicted in the following figure.


VGG-19 Model Architecture
You can clearly see that we have a total of 16 convolution layers using 3 x 3convolution filters along with max pooling layers for downsampling and a total of two fully connected hidden layers of 4096 units in each layer followed by a dense layer of 1000 units, where each unit represents one of the image categories in the ImageNet database. We do not need the last three layers since we will be using our own fully connected dense layers to predict malaria. We are more concerned with the first five blocks, so that we can leverage the VGG model as an effective feature extractor.

For one of the models, we will use it as a simple feature extractor by freezing all the five convolution blocks to make sure their weights don’t get updated after each epoch. For the last model, we will apply fine-tuning to the VGG model, where we will unfreeze the last two blocks (Block 4 and Block 5) so that their weights get updated in each iteration (per batch of data) as we train our own model.


Model 2: Pre-trained Model as a Feature Extractor
For building this model, we will leverage TensorFlow to load up the VGG-19 model, and freeze the convolution blocks so that we can use it as an image feature extractor. We will plugin our own dense layers at the end for performing the classification task.
