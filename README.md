# Image-classification-model.
A trained CNN model on over 12000 images from 20 different classes.
A model that has been trained on a huge amount of data, and the task was to classify 20 different classes of images with an accuracy of over 50%, Through the journey of building the models, I got to discover and learn new techniques that can help me in this task, such as:
Data preprocessing: How can I handle such big data with over 12000 images?
Tunning Model's hyperparameters: How can I tune the hyperparameters and discover what affects the model's performance and accuracy?
Handling images' data: Dealing with Images, enables me to use the data augmentation techniques to improve the model's performance.

# Overview and Explanation of the Image Classification Project:
In this project, I mainly used the assignments files I did during the semester. Also, I looked at some utility files to help me. The goal was to make an image classification model that works well, So here is  a brief step-by-step overview of the project.

Data Preprocessing

Data Splitting: I split the dataset into a training set and a test set with a test size of 20%, because we have a sufficient dataset to train and evaluate. This lets me check how good the model is before submitting the results. The training set was used to train the model, and the test set was used to see how well the model did on the test set.

Data Augmentation: 
To help the model generalize better and avoid overfitting, I used data augmentation techniques. This included random rotations, flips, shifts, and zooms. Data augmentation makes the training dataset bigger by creating different versions of the original images.

Normalization: Image pixel values were changed to a range of 0 to 1 by dividing by 255. This helps the model train faster.

Resizing: All images were resized to the same size, so they all had the same input dimensions for the CNN model. This is important because neural networks need fixed-size inputs.

CNN Model Architecture
The CNN model I used was designed to balance complexity and efficiency. Here's a summary of the architecture:

Input Layer: Takes the preprocessed images as input.

Convolutional Layers: Several convolutional layers were used to extract features from the images. Each convolutional layer was followed by a Rectified Linear Unit (ReLU) activation function to add non-linearity.

Pooling Layers: Max-pooling layers were added after some convolutional layers to reduce the size of the feature maps, which also reduces the number of parameters and computational load.

Fully Connected Layers: After flattening the output from the convolutional and pooling layers, fully connected (dense) layers were added. These layers help learn complex patterns and do the final classification.

Output Layer.

Training and Evaluation Process
The training and evaluation process had several steps, changing hyperparameters step by step to get good accuracy:

Hyperparameter Tuning: 
I tried different hyperparameters, like the learning rate, changing optimizers, batch size, number of epochs, and the architecture of the CNN itself (number of layers, number of filters in each layer, etc.). Gradually, I fine-tuned these parameters to improve the model's performance.

Loss Function and Optimizer: 
The categorical cross-entropy loss function was used because it's good for multi-class classification problems. The AdamW optimizer was used because it's efficient and has an adaptive learning rate.

Training: 
The model was trained on the training set, with data augmentation applied in real-time during training. This helped the model learn good features and generalize well to new data.

Evaluation: 
After training, the model was evaluated on the test set to see how well it did. The evaluation metrics included accuracy, precision, recall, and the confusion matrix. This showed how well the model was doing and if there were any specific classes the model was having trouble with.

Iteration: 
Based on the evaluation results, I made more adjustments to the model and hyperparameters. This process continued until the model reached an accuracy of about 60%, which was enough for the project goals.
