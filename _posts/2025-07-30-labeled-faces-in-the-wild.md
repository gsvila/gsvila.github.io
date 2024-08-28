---
layout: post
title:  "Face identification: Labeled Faces in the Wild"
date:   2024-07-25 18:26:57 -0600
categories: jekyll update
---

## **The problem**

Here, we will train and test some simple neural network models for **face identification**: given the picture of the face of an unknown person, identify the name of the person by comparing to the labeled pictures of persons in the database. This is a supervised learning classification problem. 

## **The dataset**

The [*Labeled Faces in the Wild*](http://vis-www.cs.umass.edu/lfw/)  (LFW) is a database of face photographs designed for studying the problem of unconstrained face recognition. The dataset contains more than 13,000 JPEG pictures of faces of famous people collected from the web. Each face has been labeled with the name of the person pictured.

We will access the LFW dataset through [`scikit-learn`](https://scikit-learn.org/0.16/datasets/labeled_faces.html). To load the dataset we invoke the [`fetch_lfw_people()`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html) method, appropriate for face recognition tasks.

The complete version of the dataset obtained through `fetch_lfw_people()` with default parameters (including a downsizing of the images to half of their original size) contains 13,233 images classified into 5,749 categories. Each image is 62x47 pixels in size and the value of each pixel is a floating point number in the range [0,1]. 

We will use a reduced sub-set with the images in the 7 categoeries for which more than 70 images are available.


### Import packages


```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time
from PIL import Image
from os import listdir, path
```


```python
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
```


```python
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Resizing, UpSampling2D, BatchNormalization, GlobalAveragePooling2D

from tensorflow.keras.metrics import MSE
from tensorflow.keras.optimizers import Adadelta, SGD, Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

### Load and inspect the dataset

Let's start by loading the dataset. 

As mentioned above, we'll consider only those categories that are represented with at least 70 pictures. We can impose this constraint through the `min_faces_per_person` parameter. We'll also use `resize=0.5` to downsize the images to half the number of pixels in each dimension. This returns 1,288 images of 62x47 pixels.

By default, the images are converted to grayscale so there's only one channel per image. 


```python
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.5)
```

Summarize some properties of our dataset. 

- The attribute `images` returns a `numpy` array of size (1288, 62, 47) = (#images, height, width), where each row corresponds to an image of (62, 47) pixels.
- The size of the array can be retrieved with `lfw_people.images.shape`.
- The `target` attribute returns an array of size (1288,) that holds the labels. Since we are working only with 7 categories, these are numbers from 0 to 6.
- The `target_names` attribute returns an array of size (7,) with the names of the persons in each category.
- The number of categories can be retrived from the size of `target_names`.


```python
# Images
X = lfw_people.images

# Size of the array
n_samples, h, w = lfw_people.images.shape

# Labels
y = lfw_people.target

# Names
target_names = lfw_people.target_names

# Number of categories
n_classes = target_names.shape[0]

# Print summary
print("Some properties of the dataset:")
print("  * number of images (n_samples): %d" % n_samples)
print("  * height (h), with (w) of each image in pixels: (%d, %d)" % (h, w))
print("  * number of categories (n_classes): %d" % n_classes)
print("  * name of the person in each category (target_names): %s" % target_names)
```

    Some properties of the dataset:
      * number of images (n_samples): 1288
      * height (h), with (w) of each image in pixels: (62, 47)
      * number of categories (n_classes): 7
      * name of the person in each category (target_names): ['Ariel Sharon' 'Colin Powell' 'Donald Rumsfeld' 'George W Bush'
     'Gerhard Schroeder' 'Hugo Chavez' 'Tony Blair']
    

We can also check that the data type of the images is `float` and that the values in the arrays are within the expected range [0,1]: 


```python
image_no = 0
print(f'This is the array for image {image_no}: \n')
print(X[image_no])
print('\nThe array data type is:', X[image_no].dtype)
```

    This is the array for image 0: 
    
    [[0.9973857  0.9973857  0.99607843 ... 0.26928106 0.23267974 0.20261438]
     [0.9973857  0.99607843 0.99477124 ... 0.275817   0.24052288 0.20915033]
     [0.9882353  0.97647065 0.96732026 ... 0.26928106 0.24052288 0.21830066]
     ...
     [0.3372549  0.2784314  0.20522876 ... 0.4117647  0.39869282 0.37908497]
     [0.30980393 0.2522876  0.19738562 ... 0.39607847 0.39607844 0.37254906]
     [0.28496733 0.24705882 0.19869281 ... 0.38431373 0.3869281  0.3803922 ]]
    
    The array data type is: float32
    

Everything looks ok.

Some convenience functions to display the images.

- `plot_gallery` displays the first `n` images in the dataset, arranged in `n_row` rows and `n_col` columns. The parameter `titles` is the title given to each image, whereas `h` and `w` are the image height and width in pixels, respectively.
- `title` returns a string with the image label and the person's name. 


```python
def plot_gallery(images, number, titles, h, w, n_row=3, n_col=4):
    """Auxiliary function to display a gallery of images"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(number):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i], cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

def title(y):
    return "Category: [%d] " % (y)
```

Now let's plot the first 12 images of the dataset with the functions defined above.


```python
n_plot = 12 # number of images to display
titles = [title(y[i]) for i in range(n_plot)] # list of titles
plot_gallery(X, n_plot, titles, h, w) 
```


<img src="/assets/images/labeled_faces_in_the_wild/output_18_0.png">        


How many images are there per category? Let's make a barplot.


```python
fig, ax = plt.subplots()
sns.countplot(y=y, ax=ax)
tick_labels = ['[' + str(i) + '] ' for i in range(n_classes)]
ax.set_yticklabels(tick_labels)
ax.set_xlabel("#images")
ax.set_ylabel("category")
ax.set_xlim([0, np.max(np.bincount(y))+50])
for bar in ax.patches:
    ax.annotate('{:.0f}'.format(bar.get_width()), (bar.get_width()+5, bar.get_y()+0.5))
plt.show()
```


    
<img src="/assets/images/labeled_faces_in_the_wild/output_20_0.png">           


As expected, all categories have 70 images or more. However, they are not balanced: the most numerous category (3) has 530 images whereas the least numerous one (5) has only 71. 

## **Face recognition with neural networks**

We'll apply three approaches to the tackle this problem:

- A simple **convolutional neural network** (CNN)
- A CNN with **transfer learning**: `ResNet50` trained with the `ImageNet` dataset.
- A CNN with **transfer learning** and **data augmentation**: `ResNet50` trained with the `ImageNet` dataset.

### 1) Simple convolutional neural network (CNN)

### Data pre-processing

Let's begin by splitting the dataset into training and test sets with the `train_test_split()` method.


```python
random_state  = 17  # controls the shuffling of the data before the split
                    # I fix it for reproducibility across executions
test_set_size = 0.2 # save 20% of the data for the test set

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size, random_state=random_state)

# Check the number of images in each set
print("Number of images in the training set:", X_train.shape[0])
print("Number of images in the test set:", X_test.shape[0])
```

    Number of images in the training set: 1030
    Number of images in the test set: 258
    

Let's investigate how each set looks like in terms of number of images per category with some barplots.


```python
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

# Training set
sns.countplot(y=y_train, ax=axs[0])
tick_labels = ['[' + str(i) + '] ' for i in range(n_classes)]
axs[0].set_yticklabels(tick_labels)
axs[0].set_xlabel("#images")
axs[0].set_ylabel("category")
axs[0].set_xlim([0, np.max(np.bincount(y_train))+100])
axs[0].set_title('Training')
for bar in axs[0].patches:
    axs[0].annotate('{:.0f}'.format(bar.get_width()), (bar.get_width()+5, bar.get_y()+0.5))

# Test set
sns.countplot(y=y_test, ax=axs[1])
tick_labels = ['[' + str(i) + '] '  for i in range(n_classes)]
axs[1].set_yticklabels(tick_labels)
axs[1].set_xlabel("#images")
axs[1].set_ylabel("category")
axs[1].set_xlim([0, np.max(np.bincount(y_test))+30])
axs[1].set_title('Test')
for bar in axs[1].patches:
    axs[1].annotate('{:.0f}'.format(bar.get_width()), (bar.get_width()+5, bar.get_y()+0.5))

plt.subplots_adjust(wspace=0.3)
plt.show()
```


    
<img src="/assets/images/labeled_faces_in_the_wild/output_29_0.png">       
    


The unbalance across categories remains in the training and test sets; category 3 is the most numerous. 

Now we have to take care of a technical detail: a dimension must be added to the arrays `X_train` and `X_text` to let Keras know that the images are grayscale (1 channel). See the discussion in this [Stack Overflow post](https://stackoverflow.com/questions/47665391/keras-valueerror-input-0-is-incompatible-with-layer-conv2d-1-expected-ndim-4).


```python
X_train = np.expand_dims(X_train , axis=-1)
X_test  = np.expand_dims(X_test , axis=-1)

# Check the new dimensions
print(X_train.shape)
print(X_test.shape)
```

    (1030, 62, 47, 1)
    (258, 62, 47, 1)
    

Finally, convert the arrays with the labels to binary class matrices:


```python
y_train = to_categorical(y_train, n_classes)
y_test  = to_categorical(y_test, n_classes) 
```

### Parameters of the neural network


```python
lr         = 1.0    # learning rate
epochs     = 30     # number of epochs
batch_size = 10     # batch size
np.random.seed(14)  # seed for the random number generator, fixed  for reproducibility
```

### CNN definition

We'll a apply a simple convolutional neural network architecture. 

This consists of two *convolutional* layers: the first with 32 (3x3) kernels and the second with 64 (3x3) kernels. A *max pooling* layer of (2x2) and a *dropout* layer of 25% follow.  After the convolutional layers, the output is flattened and fed to *dense network* with a layer of 100 neurons and a *ReLU* activation function, followed by a dropout layer of 25% and finally an output layer with *softmax* activation.

The *AdaDelta* optimizer is applied to minimize the *categorical crossentropy*, a loss function appropriate for classification problems with more than two categories.

To assess the network's performance we'll look at two metrics: the `mean square error (mse)`  and the `accuracy`.


```python
# Convolutional Neural Network Model
#---------------------------------------------------------------------#
input_layer  = Input(shape=X_train.shape[1:])
conv_1       = Conv2D(32, (3, 3), activation='relu') (input_layer)
conv_2       = Conv2D(64, (3, 3), activation='relu') (conv_1)
pool_1       = MaxPooling2D(pool_size=(2, 2)) (conv_2)
dropout_1    = Dropout(0.25) (pool_1)
flatten_1    = Flatten() (dropout_1)
dense_1      = Dense(100, activation='relu') (flatten_1)
dropout_2    = Dropout(0.25) (dense_1)
output_layer = Dense(n_classes, activation='softmax') (dropout_2)
#---------------------------------------------------------------------#
model_conv   = Model(input_layer, output_layer)
```


```python
# Inizialization of the optimizer, compilation of the model and inspection of its summary
Adadelta_optimizer = Adadelta(learning_rate=lr, rho=0.95)
model_conv.compile(optimizer=Adadelta_optimizer, loss='categorical_crossentropy', metrics=['acc', 'mse'])
model_conv.summary()
```

    Model: "model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_1 (InputLayer)        [(None, 62, 47, 1)]       0         
                                                                     
     conv2d (Conv2D)             (None, 60, 45, 32)        320       
                                                                     
     conv2d_1 (Conv2D)           (None, 58, 43, 64)        18496     
                                                                     
     max_pooling2d (MaxPooling2D  (None, 29, 21, 64)       0         
     )                                                               
                                                                     
     dropout (Dropout)           (None, 29, 21, 64)        0         
                                                                     
     flatten (Flatten)           (None, 38976)             0         
                                                                     
     dense (Dense)               (None, 100)               3897700   
                                                                     
     dropout_1 (Dropout)         (None, 100)               0         
                                                                     
     dense_1 (Dense)             (None, 7)                 707       
                                                                     
    =================================================================
    Total params: 3,917,223
    Trainable params: 3,917,223
    Non-trainable params: 0
    _________________________________________________________________
    

The model has a total of 3,917,223 parameters, all of them trainable.

Now, let's train the model keeping track of the time it took to complete (in my old laptop). 


```python
start_time   = time.time()
history_conv = model_conv.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), shuffle=True, verbose=1)
end_time     = time.time()
print('\nTotal training time of the simple CNN model : {:.5f} s'.format(end_time-start_time))
```

    Epoch 1/30
    103/103 [==============================] - 10s 83ms/step - loss: 1.7865 - acc: 0.3806 - mse: 0.1136 - val_loss: 1.6454 - val_acc: 0.4186 - val_mse: 0.1073
    Epoch 2/30
    103/103 [==============================] - 8s 82ms/step - loss: 1.6207 - acc: 0.4097 - mse: 0.1047 - val_loss: 1.3996 - val_acc: 0.4419 - val_mse: 0.0917
    Epoch 3/30
    103/103 [==============================] - 8s 81ms/step - loss: 1.2949 - acc: 0.5175 - mse: 0.0860 - val_loss: 0.9828 - val_acc: 0.6318 - val_mse: 0.0642
    Epoch 4/30
    103/103 [==============================] - 8s 81ms/step - loss: 0.9834 - acc: 0.6417 - mse: 0.0662 - val_loss: 0.8665 - val_acc: 0.7287 - val_mse: 0.0560
    Epoch 5/30
    103/103 [==============================] - 8s 82ms/step - loss: 0.7470 - acc: 0.7379 - mse: 0.0508 - val_loss: 0.7512 - val_acc: 0.7442 - val_mse: 0.0488
    Epoch 6/30
    103/103 [==============================] - 8s 82ms/step - loss: 0.5308 - acc: 0.8252 - mse: 0.0370 - val_loss: 0.7174 - val_acc: 0.7713 - val_mse: 0.0439
    Epoch 7/30
    103/103 [==============================] - 8s 82ms/step - loss: 0.3868 - acc: 0.8621 - mse: 0.0274 - val_loss: 0.6086 - val_acc: 0.8256 - val_mse: 0.0369
    Epoch 8/30
    103/103 [==============================] - 8s 81ms/step - loss: 0.2976 - acc: 0.8981 - mse: 0.0207 - val_loss: 0.5324 - val_acc: 0.8372 - val_mse: 0.0322
    Epoch 9/30
    103/103 [==============================] - 8s 81ms/step - loss: 0.2568 - acc: 0.9194 - mse: 0.0179 - val_loss: 0.5042 - val_acc: 0.8566 - val_mse: 0.0295
    Epoch 10/30
    103/103 [==============================] - 9s 88ms/step - loss: 0.1993 - acc: 0.9282 - mse: 0.0145 - val_loss: 0.6249 - val_acc: 0.8643 - val_mse: 0.0313
    Epoch 11/30
    103/103 [==============================] - 9s 85ms/step - loss: 0.1542 - acc: 0.9466 - mse: 0.0113 - val_loss: 0.5010 - val_acc: 0.8760 - val_mse: 0.0264
    Epoch 12/30
    103/103 [==============================] - 9s 85ms/step - loss: 0.1498 - acc: 0.9456 - mse: 0.0114 - val_loss: 0.5134 - val_acc: 0.8953 - val_mse: 0.0250
    Epoch 13/30
    103/103 [==============================] - 8s 82ms/step - loss: 0.1252 - acc: 0.9583 - mse: 0.0088 - val_loss: 0.5840 - val_acc: 0.8721 - val_mse: 0.0273
    Epoch 14/30
    103/103 [==============================] - 8s 81ms/step - loss: 0.1034 - acc: 0.9670 - mse: 0.0070 - val_loss: 0.5206 - val_acc: 0.8915 - val_mse: 0.0241
    Epoch 15/30
    103/103 [==============================] - 8s 82ms/step - loss: 0.0833 - acc: 0.9738 - mse: 0.0057 - val_loss: 0.7331 - val_acc: 0.8953 - val_mse: 0.0269
    Epoch 16/30
    103/103 [==============================] - 8s 82ms/step - loss: 0.0883 - acc: 0.9709 - mse: 0.0065 - val_loss: 0.6011 - val_acc: 0.8798 - val_mse: 0.0268
    Epoch 17/30
    103/103 [==============================] - 8s 82ms/step - loss: 0.0687 - acc: 0.9718 - mse: 0.0052 - val_loss: 0.5774 - val_acc: 0.8953 - val_mse: 0.0246
    Epoch 18/30
    103/103 [==============================] - 8s 82ms/step - loss: 0.0624 - acc: 0.9767 - mse: 0.0047 - val_loss: 0.5692 - val_acc: 0.8915 - val_mse: 0.0226
    Epoch 19/30
    103/103 [==============================] - 8s 82ms/step - loss: 0.0763 - acc: 0.9777 - mse: 0.0051 - val_loss: 0.5952 - val_acc: 0.8953 - val_mse: 0.0240
    Epoch 20/30
    103/103 [==============================] - 8s 82ms/step - loss: 0.0553 - acc: 0.9796 - mse: 0.0040 - val_loss: 0.7606 - val_acc: 0.8721 - val_mse: 0.0279
    Epoch 21/30
    103/103 [==============================] - 9s 83ms/step - loss: 0.0736 - acc: 0.9767 - mse: 0.0055 - val_loss: 0.7147 - val_acc: 0.9109 - val_mse: 0.0233
    Epoch 22/30
    103/103 [==============================] - 9s 83ms/step - loss: 0.0396 - acc: 0.9854 - mse: 0.0029 - val_loss: 0.6385 - val_acc: 0.8837 - val_mse: 0.0256
    Epoch 23/30
    103/103 [==============================] - 9s 83ms/step - loss: 0.0360 - acc: 0.9874 - mse: 0.0026 - val_loss: 0.6607 - val_acc: 0.8992 - val_mse: 0.0225
    Epoch 24/30
    103/103 [==============================] - 9s 85ms/step - loss: 0.0504 - acc: 0.9835 - mse: 0.0034 - val_loss: 0.6197 - val_acc: 0.9147 - val_mse: 0.0210
    Epoch 25/30
    103/103 [==============================] - 9s 85ms/step - loss: 0.0400 - acc: 0.9825 - mse: 0.0031 - val_loss: 0.7151 - val_acc: 0.8953 - val_mse: 0.0254
    Epoch 26/30
    103/103 [==============================] - 9s 84ms/step - loss: 0.0275 - acc: 0.9922 - mse: 0.0019 - val_loss: 0.7475 - val_acc: 0.8953 - val_mse: 0.0253
    Epoch 27/30
    103/103 [==============================] - 8s 82ms/step - loss: 0.0363 - acc: 0.9845 - mse: 0.0026 - val_loss: 0.7274 - val_acc: 0.9109 - val_mse: 0.0221
    Epoch 28/30
    103/103 [==============================] - 8s 82ms/step - loss: 0.0300 - acc: 0.9903 - mse: 0.0021 - val_loss: 0.6912 - val_acc: 0.9070 - val_mse: 0.0226
    Epoch 29/30
    103/103 [==============================] - 8s 81ms/step - loss: 0.0508 - acc: 0.9816 - mse: 0.0036 - val_loss: 0.7435 - val_acc: 0.8992 - val_mse: 0.0232
    Epoch 30/30
    103/103 [==============================] - 9s 84ms/step - loss: 0.0180 - acc: 0.9961 - mse: 9.9345e-04 - val_loss: 0.7571 - val_acc: 0.8953 - val_mse: 0.0210
    
    Total training time of the simple CNN model : 256.66685 s
    

### Model evaluation:  *accuracy* and *mean squared error  (MSE)*

Let's plot the evolution of these metrics with the epochs.


```python
f = plt.figure(figsize=(8,5))
f.subplots_adjust(wspace=0.4)

plt.subplot(1,2,1)
plt.plot(range(1, epochs+1), history_conv.history['mse'], '.-', linewidth=2, label='training')
plt.plot(range(1, epochs+1), history_conv.history['val_mse'], '.-', linewidth=2, label='test')
plt.xlabel('epoch')
plt.ylabel('MSE')
#plt.xticks(range(0, epochs+1, 5))
plt.grid()
plt.legend(loc='upper right')
plt.title('MSE')

plt.subplot(1,2,2)
plt.plot(range(1, epochs+1), history_conv.history['acc'], '.-', linewidth=2, label='training')
plt.plot(range(1, epochs+1), history_conv.history['val_acc'], '.-', linewidth=2, label='test')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.legend(loc='lower right')
plt.title('accuracy')

plt.suptitle('Metrics vs. epoch - Simple CNN ')
plt.show()
```


    
<img src="/assets/images/labeled_faces_in_the_wild/output_45_0.png">       
    


After 25 epochs, both `mse` and `accuracy` become approximately asyntotic. The `mse` reaches values of ~0.001 and ~0.02 for the training and test sets, respectively; the corresponding values of the `accuracy` are ~0.99 and ~0.89. No overfitting is observed. 

### Confusion matrix

Let's first calculate the predicted labels for the images in the test set and then plot the confusion matrix.

The `predict()` method returns an array whose elements are interpreted as the probabilities that a given images belongs to each category. The category with the largest probability defines the label predicted by the model for that image.


```python
# Model predictions for the test set
y_pred  = model_conv.predict(X_test, verbose=1) 

# Transform the predictions in the format above to a label in the range 0-6 
y_true  = np.argmax(y_test, axis=1) # real labels
y_model = np.argmax(y_pred, axis=1) # predicted labels 
```

    9/9 [==============================] - 0s 22ms/step
    


```python
# Calculation of the confusion matrix
CM = confusion_matrix(y_true, y_model) 
```

Plot of the confusion matrix with annotations:


```python
plt.figure(figsize=(5,5))
str_labels = ['[' + str(i) + '] '  for i in range(n_classes)] 
sns.heatmap(CM, annot=True,  fmt='d', xticklabels=str_labels, yticklabels=str_labels, cmap='jet', cbar=False)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Simple CNN')
plt.show()
```


    
<img src="/assets/images/labeled_faces_in_the_wild/output_51_0.png">       
    


For most categories, the images are largely assigned the correct label. However, the performance varies across categories.

As a side note, the ratio between the trace of the confusion matrix and the sum of all its elements returns the model `accuracy`:


```python
print(f'accuracy of the simple CNN model on the test set after {epochs} epochs = {np.trace(CM)/np.sum(CM):.4f}')
```

    accuracy of the simple CNN model on the test set after 30 epochs = 0.8953
    

To analize the results per category in more detail, let's calculate the `precision`, `recall` and `F1-score` for each category: 

* `precision = TP/(TP+FP)`
* `recall    = TP/(TP+FN)`
* `F1_score  = 2 * precision * recall/(precision+recall)` (harmonic mean of `precision` and `recall`)

where:

* `TP` = true positives
* `FP` = false positives
* `FN` = false negatives


```python
precision = precision_score(y_true, y_model, average=None)
recall    = recall_score(y_true, y_model, average=None)
F1_score  = f1_score(y_true, y_model, average=None) 
```


```python
# Store the results in a DataFrame and print it
cols = ['model', 'category', 'no_images', 'precision', 'recall', 'F1-score']
precision_recall = pd.DataFrame(columns=cols)

# Completo las columnas
precision_recall['model']        = ['simple CNN']*n_classes
precision_recall['category']     = range(n_classes)
precision_recall['no_images']    = [n for n in np.sum(CM, axis=1)]
precision_recall['precision']    = precision
precision_recall['recall']       = recall
precision_recall['F1-score']     = F1_score

# Sort by number of images
precision_recall.sort_values(by='no_images', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>category</th>
      <th>no_images</th>
      <th>precision</th>
      <th>recall</th>
      <th>F1-score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>simple CNN</td>
      <td>3</td>
      <td>108</td>
      <td>0.903509</td>
      <td>0.953704</td>
      <td>0.927928</td>
    </tr>
    <tr>
      <th>1</th>
      <td>simple CNN</td>
      <td>1</td>
      <td>48</td>
      <td>0.921569</td>
      <td>0.979167</td>
      <td>0.949495</td>
    </tr>
    <tr>
      <th>6</th>
      <td>simple CNN</td>
      <td>6</td>
      <td>30</td>
      <td>0.810811</td>
      <td>1.000000</td>
      <td>0.895522</td>
    </tr>
    <tr>
      <th>2</th>
      <td>simple CNN</td>
      <td>2</td>
      <td>25</td>
      <td>0.944444</td>
      <td>0.680000</td>
      <td>0.790698</td>
    </tr>
    <tr>
      <th>4</th>
      <td>simple CNN</td>
      <td>4</td>
      <td>21</td>
      <td>0.888889</td>
      <td>0.761905</td>
      <td>0.820513</td>
    </tr>
    <tr>
      <th>5</th>
      <td>simple CNN</td>
      <td>5</td>
      <td>14</td>
      <td>0.916667</td>
      <td>0.785714</td>
      <td>0.846154</td>
    </tr>
    <tr>
      <th>0</th>
      <td>simple CNN</td>
      <td>0</td>
      <td>12</td>
      <td>0.875000</td>
      <td>0.583333</td>
      <td>0.700000</td>
    </tr>
  </tbody>
</table>
</div>



In general, for this model, the performance as measured by these metrics decreases with decreasing number of images. 

### 2) CNN model with transfer learning 

We'll work with the same subset of images, but now in RGB instead of grayscale.

Let's load the data. Now we use `fetch_lfw_people()` with `color=True` to load the three channels.


```python
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.5, color=True)

# Images
X_tl = lfw_people.images

# Array size
n_samples, h, w, d = lfw_people.images.shape

# Labels and categories
y_tl = lfw_people.target

# Names
target_names = lfw_people.target_names

# Number of categories
n_classes = target_names.shape[0]
```

Display the first 12 images in the dataset:


```python
n_plot = 12 # number of images to display
titles = [title(y_tl[i]) for i in range(n_plot)] # titles
plot_gallery(X_tl, n_plot, titles, h, w) 
```


    
<img src="/assets/images/labeled_faces_in_the_wild/output_64_0.png">       
    


Let's now split the data into the training and test sets as before:


```python
random_state  = 17  # controls the shuffling of the data before the split
                    # I fix it for reproducibility across executions
test_set_size = 0.2 # save 20% of the data for the test set

# Splitting
X_train_tl, X_test_tl, y_train_tl, y_test_tl = train_test_split(X_tl, y_tl, test_size=test_set_size, 
                                                                random_state=random_state)

# Convert the arrays with the labels to binary class matrices
y_train_tl = to_categorical(y_train_tl, n_classes)
y_test_tl  = to_categorical(y_test_tl, n_classes) 
```

### Parameters of the neural network


```python
lr_tl         = 1.0    # learning rate
epochs_tl     = 15     # number of epochs
batch_size_tl = 10     # batch size
np.random.seed(14)     # seed for the random number generator, fixed  for reproducibility
```

### Model definition

This a model with `transfer learning`. For the convolutional part of the network, we'll apply the architecture of the `ResNet50` network - [one of the pre-trained deep learning models available through Keras Applications](https://keras.io/api/applications/#usage-examples-for-image-classification-models). The network was trained with the images in the [`ImageNet` database](https://www.image-net.org/index.php). 

Let's define an instance of `ResNet50`. 

We fix `weights='imagenet'` to use the weights of the newtowrk trained with `ImageNet`. Setting `include_top=False` does not include the totally connected layer of the network in our model.  The expected sixe of the images is (224,224,3).

The weights of the `BatchNormalization` layers are particular to each problem, so we define these layers as trainable within our instance of `ResNet50`.


```python
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

for layer in resnet_model.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = True
    else:
        layer.trainable = False
```

The images require some pre-processing to take them to the format expected by `ResNet50`.

According to the TensorFlow documentation, the images are first converted from RGB to BGR and each channel is centered around zero with respect to the images in `ImageNet`, without any scaling.
Check the documentation [here](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/preprocess_input).



```python
X_train_tl = preprocess_input(X_train_tl)
X_test_tl  = preprocess_input(X_test_tl)
```

Now, we define the complete network adding the necessary layers before and after the `ResNet` block.

After the input layer, I apply a `Resampling` layer to take the input images to the dimensions expected by ResNet.
After the ResNet block, I apply a `Flatten` layer to prepare the output of `ResNet` as input for a totally connected layer of 256 neurons.
A `Dropout` and a  `BatchNormalization` layer follow, and finally the output layer with `softmax` activation.


```python
# CNN model with transfer learning using ResNet50 trained with ImageNet.
#------------------------------------------------------------------------------------------------------------------#
model_tl = Sequential()
model_tl.add(Input(shape=X_train_tl.shape[1:]))
model_tl.add(Resizing(224,224))
model_tl.add(resnet_model)
model_tl.add(Flatten())
model_tl.add(Dense(256, activation='relu'))
model_tl.add(Dropout(.25))
model_tl.add(BatchNormalization())
model_tl.add(Dense(n_classes, activation='softmax'))
```

The *Adam* optimizer is applied to minimize the *categorical crossentropy*.
To assess the network's performance we'll look at the same two metrics: the `mean square error` and the `accuracy`.


```python
# Inizialization of the optimizer, compilation of the model and inspection of its summary
model_tl.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mse'])
model_tl.build()
model_tl.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     resizing (Resizing)         (None, 224, 224, 3)       0         
                                                                     
     resnet50 (Functional)       (None, 7, 7, 2048)        23587712  
                                                                     
     flatten_1 (Flatten)         (None, 100352)            0         
                                                                     
     dense_2 (Dense)             (None, 256)               25690368  
                                                                     
     dropout_2 (Dropout)         (None, 256)               0         
                                                                     
     batch_normalization (BatchN  (None, 256)              1024      
     ormalization)                                                   
                                                                     
     dense_3 (Dense)             (None, 7)                 1799      
                                                                     
    =================================================================
    Total params: 49,280,903
    Trainable params: 25,745,799
    Non-trainable params: 23,535,104
    _________________________________________________________________
    

The model has a total of 49,280,903 parameters, of which 25,745,799 are trainable.

Now, let's train the model keeping track of the time it took to complete (again in my old laptop). 


```python
start_time = time.time()
history_tl = model_tl.fit(X_train_tl, y_train_tl, epochs=epochs_tl, batch_size=batch_size_tl, 
                          validation_data=(X_test_tl, y_test_tl), shuffle=True, verbose=1)
end_time   = time.time()
print('\nTotal training time of the CNN wit transfer learning ResNet50: {:.5f} seconds'.format(end_time-start_time))
```

    Epoch 1/15
    103/103 [==============================] - 254s 2s/step - loss: 1.4228 - acc: 0.5379 - mse: 0.0874 - val_loss: 2.8092 - val_acc: 0.1860 - val_mse: 0.1565
    Epoch 2/15
    103/103 [==============================] - 259s 3s/step - loss: 0.4451 - acc: 0.8689 - mse: 0.0294 - val_loss: 2.3669 - val_acc: 0.1860 - val_mse: 0.1490
    Epoch 3/15
    103/103 [==============================] - 233s 2s/step - loss: 0.2106 - acc: 0.9505 - mse: 0.0124 - val_loss: 2.3542 - val_acc: 0.1163 - val_mse: 0.1547
    Epoch 4/15
    103/103 [==============================] - 232s 2s/step - loss: 0.1355 - acc: 0.9689 - mse: 0.0080 - val_loss: 2.2601 - val_acc: 0.1202 - val_mse: 0.1342
    Epoch 5/15
    103/103 [==============================] - 228s 2s/step - loss: 0.1316 - acc: 0.9689 - mse: 0.0075 - val_loss: 2.1539 - val_acc: 0.1434 - val_mse: 0.1380
    Epoch 6/15
    103/103 [==============================] - 224s 2s/step - loss: 0.0771 - acc: 0.9893 - mse: 0.0038 - val_loss: 1.5194 - val_acc: 0.4457 - val_mse: 0.1019
    Epoch 7/15
    103/103 [==============================] - 238s 2s/step - loss: 0.0484 - acc: 0.9961 - mse: 0.0023 - val_loss: 1.0577 - val_acc: 0.6202 - val_mse: 0.0710
    Epoch 8/15
    103/103 [==============================] - 221s 2s/step - loss: 0.0467 - acc: 0.9942 - mse: 0.0024 - val_loss: 0.7665 - val_acc: 0.7248 - val_mse: 0.0505
    Epoch 9/15
    103/103 [==============================] - 226s 2s/step - loss: 0.0525 - acc: 0.9883 - mse: 0.0030 - val_loss: 0.5470 - val_acc: 0.8178 - val_mse: 0.0392
    Epoch 10/15
    103/103 [==============================] - 220s 2s/step - loss: 0.0735 - acc: 0.9767 - mse: 0.0048 - val_loss: 0.3785 - val_acc: 0.8566 - val_mse: 0.0282
    Epoch 11/15
    103/103 [==============================] - 222s 2s/step - loss: 0.0732 - acc: 0.9825 - mse: 0.0044 - val_loss: 0.6008 - val_acc: 0.8140 - val_mse: 0.0388
    Epoch 12/15
    103/103 [==============================] - 223s 2s/step - loss: 0.0716 - acc: 0.9825 - mse: 0.0043 - val_loss: 0.3931 - val_acc: 0.8682 - val_mse: 0.0288
    Epoch 13/15
    103/103 [==============================] - 242s 2s/step - loss: 0.0519 - acc: 0.9893 - mse: 0.0030 - val_loss: 0.4946 - val_acc: 0.8372 - val_mse: 0.0352
    Epoch 14/15
    103/103 [==============================] - 246s 2s/step - loss: 0.0618 - acc: 0.9845 - mse: 0.0040 - val_loss: 0.4202 - val_acc: 0.8527 - val_mse: 0.0293
    Epoch 15/15
    103/103 [==============================] - 237s 2s/step - loss: 0.0388 - acc: 0.9932 - mse: 0.0022 - val_loss: 0.7170 - val_acc: 0.7481 - val_mse: 0.0515
    
    Total training time of the CNN wit transfer learning ResNet50: 3506.01013 seconds
    

### Model evaluation:  *accuracy* and *mean squared error  (MSE)*

Let's plot the evolution of these metrics with the epoch:


```python
f = plt.figure(figsize=(10,5))
f.subplots_adjust(wspace=0.4)

plt.subplot(1,2,1)
plt.plot(range(1, epochs_tl+1), history_tl.history['mse'], '.-', linewidth=2, label='training')
plt.plot(range(1, epochs_tl+1), history_tl.history['val_mse'], '.-', linewidth=2, label='test')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.xticks(range(0, epochs_tl+1, 5))
plt.grid()
plt.legend(loc='upper right', fontsize=9)
plt.title('MSE')

plt.subplot(1,2,2)
plt.plot(range(1, epochs_tl+1), history_tl.history['acc'], '.-', linewidth=2, label='training')
plt.plot(range(1, epochs_tl+1), history_tl.history['val_acc'], '.-', linewidth=2, label='test')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.xticks(range(0, epochs_tl+1, 5))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.grid()
plt.legend(loc='lower right', fontsize=9)
plt.title('accuracy')

plt.suptitle('Metrics vs. epoch - CNN with transfer learning ResNet50')
plt.show()
```


    
<img src="/assets/images/labeled_faces_in_the_wild/output_84_0.png">       
    


After 15 epochs, the `mse` reaches values of ~0.002 and ~0.04 for the training and test sets, respectively; the corresponding values of the `accuracy` are ~0.99 and ~0.8. No overfitting is observed.

Let's calculate the model predictions and inspect the confusion matrix:


```python
# Model predictions
y_pred_tl  = model_tl.predict(X_test_tl, verbose=1)
y_true_tl  = np.argmax(y_test_tl, axis=1) 
y_model_tl = np.argmax(y_pred_tl, axis=1)
```

    9/9 [==============================] - 14s 1s/step
    


```python
# Confusion matrix
CM_tl = confusion_matrix(y_true_tl, y_model_tl)
```


```python
# Plot the confusion matrix
plt.figure(figsize=(5,5))
str_labels = ['[' + str(i) + '] '  for i in range(n_classes)] 
sns.heatmap(CM_tl, annot=True,  fmt='d', xticklabels=str_labels, yticklabels=str_labels, cmap='jet', cbar=False)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix - CNN with transfer learning ResNet50')
plt.show()
```


    
<img src="/assets/images/labeled_faces_in_the_wild/output_89_0.png">       
    


The results are similar to those of the simple CNN model.

### 3) Model with *transfer learning* and *data augmentation*

*Data augmentation* is a technique to expand the size of the dataset with new images obtained from the original ones applying different types of transformations. The `ImageDataGenerator` class in Keras is a convenient way to augment the dataset *on-the-fly*, while the model is training: a batch of images enters the `ImageDataGenerator`, each image is randomly transformed in some way, and the transformed batch (only) is returned. 

I found [this](https://www.analyticsvidhya.com/blog/2020/08/image-augmentation-on-the-fly-using-keras-imagedatagenerator/) and [this](https://pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/) post on the Keras `ImageDataGenerator` very instructive. 

We'll use the same subset of RGB images of the previous model.


```python
# Fetch the dataset
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.5, color=True)

# Images
X_tl_aug = lfw_people.images

# Array size
n_samples, h, w, d = lfw_people.images.shape

# Labels and categories
y_tl_aug = lfw_people.target

# Names
target_names = lfw_people.target_names

# Number of categories
n_classes = target_names.shape[0]
```

Splitting of the dataset:


```python
random_state  = 17  # controls the shuffling of the data before the split
                    # I fix it for reproducibility across executions
test_set_size = 0.2 # save 20% of the data for the test set

# Spliting
X_train_tl_aug, X_test_tl_aug, y_train_tl_aug, y_test_tl_aug = train_test_split(X_tl_aug, y_tl_aug, 
                                                                                test_size=test_set_size, 
                                                                                random_state=random_state)

# Convert the arrays with the labels to binary class matrices
y_train_tl_aug = to_categorical(y_train_tl_aug, n_classes)
y_test_tl_aug  = to_categorical(y_test_tl_aug, n_classes) 
```

Pre-processing of the images to take them to the format expected by `ResNet50`.


```python
X_train_tl_aug = preprocess_input(X_train_tl_aug)
X_test_tl_aug  = preprocess_input(X_test_tl_aug)
```

### Parameters of the neural network


```python
lr_tl_aug         = 1.0    # learning rate
epochs_tl_aug     = 15     # number of epochs
batch_size_tl_aug = 10     # batch size
np.random.seed(14)         # seed for the random number generator, fixed  for reproducibility
```

### Data augmentation

Here we configure the `ImageDataGenerator`. The type of transformations applied are random shifts (`width_shift_range`, `height_shift_range`) and flips (`horizontal_flip`, `vertical_flip`).


```python
steps_per_epoch = int(round(X_train_tl_aug.shape[0]/batch_size_tl_aug))

datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)           
            width_shift_range=0.1, # randomly shift images horizontally (fraction of total width)            
            height_shift_range=0.1, # randomly shift images vertically (fraction of total height)
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts           
            fill_mode='nearest', # set mode for filling points outside the input boundaries
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            rescale=None, # set rescaling factor (applied before any other transformation)
            preprocessing_function=None, # set function that will be applied on each input            
            data_format=None, # image data format, either "channels_first" or "channels_last"            
            validation_split=0.0) # fraction of images reserved for validation (strictly between 0 and 1)

datagen.fit(X_train_tl_aug)
```

### Model definition

We'll use the same network architecture and optimizer as in the previous case.


```python
# CNN with transfer learning with ResNet50 trained with ImageNet
#------------------------------------------------------------------------------------------------------------------#
model_tl_aug = Sequential()
model_tl_aug.add(Input(shape=X_train_tl_aug.shape[1:]))
model_tl_aug.add(Resizing(224,224))
model_tl_aug.add(resnet_model)
model_tl_aug.add(Flatten())
model_tl_aug.add(Dense(256, activation='relu'))
model_tl_aug.add(Dropout(.25))
model_tl_aug.add(BatchNormalization())
model_tl_aug.add(Dense(n_classes, activation='softmax'))
```


```python
# Optimizer, compilation and summary
model_tl_aug.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mse'])
model_tl_aug.build()
model_tl_aug.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     resizing_1 (Resizing)       (None, 224, 224, 3)       0         
                                                                     
     resnet50 (Functional)       (None, 7, 7, 2048)        23587712  
                                                                     
     flatten_2 (Flatten)         (None, 100352)            0         
                                                                     
     dense_4 (Dense)             (None, 256)               25690368  
                                                                     
     dropout_3 (Dropout)         (None, 256)               0         
                                                                     
     batch_normalization_1 (Batc  (None, 256)              1024      
     hNormalization)                                                 
                                                                     
     dense_5 (Dense)             (None, 7)                 1799      
                                                                     
    =================================================================
    Total params: 49,280,903
    Trainable params: 25,745,799
    Non-trainable params: 23,535,104
    _________________________________________________________________
    

As before, the model has a total of 49,280,903 parameters, of which 25,745,799 are trainable.


```python
# Model training
start_time      = time.time()
train_generator = datagen.flow(X_train_tl_aug, y_train_tl_aug, batch_size=batch_size_tl_aug)

history_tl_aug = model_tl_aug.fit(train_generator,
                           steps_per_epoch=steps_per_epoch,
                           epochs=epochs_tl_aug,
                           validation_data=(X_test_tl_aug, y_test_tl_aug),
                           workers=5,
                           shuffle=True,
                           verbose=1)

end_time = time.time()

print('\nTotal training time of the CNN model with transfer learning ResNet50 and data augmentation: {:.5f} seconds'.format(end_time-start_time))
```

    Epoch 1/15
    103/103 [==============================] - 259s 2s/step - loss: 1.4483 - acc: 0.5087 - mse: 0.0906 - val_loss: 1.1782 - val_acc: 0.6008 - val_mse: 0.0790
    Epoch 2/15
    103/103 [==============================] - 256s 2s/step - loss: 0.7687 - acc: 0.7592 - mse: 0.0503 - val_loss: 0.7340 - val_acc: 0.7403 - val_mse: 0.0524
    Epoch 3/15
    103/103 [==============================] - 243s 2s/step - loss: 0.5727 - acc: 0.8107 - mse: 0.0394 - val_loss: 0.8029 - val_acc: 0.7403 - val_mse: 0.0521
    Epoch 4/15
    103/103 [==============================] - 239s 2s/step - loss: 0.4811 - acc: 0.8544 - mse: 0.0330 - val_loss: 0.6646 - val_acc: 0.8372 - val_mse: 0.0375
    Epoch 5/15
    103/103 [==============================] - 226s 2s/step - loss: 0.4539 - acc: 0.8466 - mse: 0.0310 - val_loss: 0.4673 - val_acc: 0.8566 - val_mse: 0.0307
    Epoch 6/15
    103/103 [==============================] - 247s 2s/step - loss: 0.4179 - acc: 0.8689 - mse: 0.0291 - val_loss: 0.6164 - val_acc: 0.7946 - val_mse: 0.0405
    Epoch 7/15
    103/103 [==============================] - 234s 2s/step - loss: 0.3534 - acc: 0.8854 - mse: 0.0243 - val_loss: 0.3733 - val_acc: 0.8876 - val_mse: 0.0254
    Epoch 8/15
    103/103 [==============================] - 241s 2s/step - loss: 0.3317 - acc: 0.8922 - mse: 0.0227 - val_loss: 0.4130 - val_acc: 0.8643 - val_mse: 0.0286
    Epoch 9/15
    103/103 [==============================] - 245s 2s/step - loss: 0.2704 - acc: 0.9233 - mse: 0.0178 - val_loss: 0.2736 - val_acc: 0.9147 - val_mse: 0.0194
    Epoch 10/15
    103/103 [==============================] - 225s 2s/step - loss: 0.3107 - acc: 0.9049 - mse: 0.0211 - val_loss: 0.3392 - val_acc: 0.8915 - val_mse: 0.0232
    Epoch 11/15
    103/103 [==============================] - 226s 2s/step - loss: 0.3035 - acc: 0.8981 - mse: 0.0212 - val_loss: 0.2435 - val_acc: 0.9186 - val_mse: 0.0177
    Epoch 12/15
    103/103 [==============================] - 233s 2s/step - loss: 0.2659 - acc: 0.9126 - mse: 0.0187 - val_loss: 0.2434 - val_acc: 0.9302 - val_mse: 0.0170
    Epoch 13/15
    103/103 [==============================] - 227s 2s/step - loss: 0.2245 - acc: 0.9262 - mse: 0.0163 - val_loss: 0.1697 - val_acc: 0.9457 - val_mse: 0.0121
    Epoch 14/15
    103/103 [==============================] - 229s 2s/step - loss: 0.1922 - acc: 0.9495 - mse: 0.0128 - val_loss: 0.2787 - val_acc: 0.9186 - val_mse: 0.0193
    Epoch 15/15
    103/103 [==============================] - 242s 2s/step - loss: 0.1970 - acc: 0.9408 - mse: 0.0133 - val_loss: 0.2156 - val_acc: 0.9419 - val_mse: 0.0139
    
    Total training time of the CNN model with transfer learning ResNet50 and data augmentation: 3570.76614 seconds
    

The training time is similar to that of the previous model without data augmentation.

### Model evaluation:  *accuracy* and *mean squared error  (MSE)*

Let's plot once more the evolution of the chosen metrics with the epoch.


```python
f = plt.figure(figsize=(10,5))
f.subplots_adjust(wspace=0.4)

plt.subplot(1,2,1)
plt.plot(range(1, epochs_tl_aug+1), history_tl_aug.history['mse'], '.-', linewidth=2, label='training')
plt.plot(range(1, epochs_tl_aug+1), history_tl_aug.history['val_mse'], '.-', linewidth=2, label='test')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.xticks(range(0, epochs_tl_aug+1, 5))
plt.grid()
plt.legend(loc='upper right', fontsize=9)
plt.title('MSE')

plt.subplot(1,2,2)
plt.plot(range(1, epochs_tl_aug+1), history_tl_aug.history['acc'], '.-', linewidth=2, label='training')
plt.plot(range(1, epochs_tl_aug+1), history_tl_aug.history['val_acc'], '.-', linewidth=2, label='test')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.xticks(range(0, epochs_tl_aug+1, 5))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.grid()
plt.legend(loc='lower right', fontsize=9)
plt.title('accuracy')

plt.suptitle('Metrics vs. epoch - CNN with transfer learning ResNet50 and data augmentation')
plt.show()
```


    
<img src="/assets/images/labeled_faces_in_the_wild/output_113_0.png">       
    


For this model, after 15 epochs `mse` and `accuracy` reach values of ~0.013 and ~0.94, respectively, both for the training and test sets.

This model performs better than the two others in terms of `mse` and `accuracy` on the training set. However, this could indicate that it suffers from **overfitting**. 

Let's calculate the model predictions for the images in the test set, and the confusion matrix


```python
# Model predictions
y_pred_tl_aug  = model_tl.predict(X_test_tl_aug, verbose=1)
y_true_tl_aug  = np.argmax(y_test_tl_aug, axis=1) 
y_model_tl_aug = np.argmax(y_pred_tl_aug, axis=1)
```

    9/9 [==============================] - 14s 2s/step
    


```python
# Confusion matrix
CM_tl_aug = confusion_matrix(y_true_tl_aug, y_model_tl_aug)
```


```python
# Plot the confusion matrix
plt.figure(figsize=(5,5))
str_labels = ['[' + str(i) + '] '  for i in range(n_classes)] 
sns.heatmap(CM_tl_aug, annot=True,  fmt='d', xticklabels=str_labels, yticklabels=str_labels, cmap='jet', cbar=False)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix - CNN with transfer learning ResNet50 and data augmentation')
plt.show()
```


    
<img src="/assets/images/labeled_faces_in_the_wild/output_118_0.png">       
    


##  Testing of the trained networks on new images

For each person in the 7 categories, I downloaded the picture from their Wikipedia entry (access date 07/18/2023) at the following links:.

* [0] [Ariel Sharon](https://en.wikipedia.org/wiki/Ariel_Sharon)
* [1] [Collin Powell](https://en.wikipedia.org/wiki/Colin_Powell)
* [2] [Donal Rumsfeld](https://en.wikipedia.org/wiki/Donald_Rumsfeld)
* [3] [George W Bush](https://en.wikipedia.org/wiki/George_W._Bush)
* [4] [Gerhard Schroder](https://en.wikipedia.org/wiki/Gerhard_Schr%C3%B6der)
* [5] [Hugo Chávez](https://en.wikipedia.org/wiki/Hugo_Ch%C3%A1vez)
* [6] [Tony Blair](https://en.wikipedia.org/wiki/Tony_Blair)

The pictures were trimmed to center the faces, trying to emulate the style of the images in the *Labeled Faces in the Wild* dataset. Then, they were resized to the same size in pixels as in LFW, height x width = 62 x 47.

Finally, I applied the three CNN models to classify the new images.

### CNN with *transfer learning*


```python
# Size of the images
x_wiki_height     = 62  # height in pixels
x_wiki_width      = 47  # width in pixels
x_wiki_n_channels = 3   # no. of channels (RGB images)

# List with the path and name of the images
file_path  = './wikipedia_images' 
file_names = [f for f in listdir(file_path) if path.isfile(path.join(file_path, f))] # nombres

# Arrays to store the images and their true labels
x_wiki_n_images = len(file_names)
x_wiki = np.empty((x_wiki_n_images, x_wiki_height, x_wiki_width, x_wiki_n_channels), dtype=np.int32) # images
y_wiki = np.empty((x_wiki_n_images), dtype=np.int64) # labels

# load the images
for i, name in enumerate(file_names):
    x = image.load_img(path.join(file_path, name), target_size=(x_wiki_height, x_wiki_width))
    x = image.img_to_array(x)
    x_wiki[i] = x 
    y_wiki[i] = np.where(lfw_people.target_names == name.split('.')[0].replace('_', ' '))[0][0]
    
# Check the dimensions of the arrays
print(x_wiki.shape)
print(y_wiki.shape)
```

    (7, 62, 47, 3)
    (7,)
    


```python
# Pre-processig to take the images to the format expected by ResNet50
x_wiki_tl = x_wiki.astype('float32') / 255.0 
x_wiki_tl = preprocess_input(x_wiki_tl)
```


```python
# Predictions
y_wiki_tl   = model_tl.predict(x_wiki_tl, verbose=1)
y_wiki_pred = np.argmax(y_wiki_tl, axis=1)
```

    1/1 [==============================] - 0s 399ms/step
    


```python
# Plot the images indicating the true and predicted labels
titles = ['True: [' + str(y_wiki[n]) + '] ' + '\n' + 'Predicted: [' + str(y_wiki_pred[n]) + ']' for n in range(len(file_names)) ]
plot_gallery(x_wiki, x_wiki_n_images, titles, x_wiki_height, x_wiki_width) 
plt.tight_layout()
```


    
<img src="/assets/images/labeled_faces_in_the_wild/output_125_0.png">       
    


The CNN with transfer learning model predicted the correct label for 5 out of the 7 images; this is an `accuracy ~ 0.7`.

### CNN with *transfer learning* and *data augmentation*


```python
# Predictions
y_wiki_tl_aug   = model_tl_aug.predict(x_wiki_tl, verbose=1)
y_wiki_pred_aug = np.argmax(y_wiki_tl_aug, axis=1)
```

    1/1 [==============================] - 2s 2s/step
    


```python
# Plot the images indicating the true and predicted labels
titles = ['True: [' + str(y_wiki[n]) + '] ' + '\n' + 'Predcted: [' + str(y_wiki_pred_aug[n]) + ']' for n in range(len(file_names)) ]
plot_gallery(x_wiki, x_wiki_n_images, titles, x_wiki_height, x_wiki_width) 
plt.tight_layout()
```


    
<img src="/assets/images/labeled_faces_in_the_wild/output_129_0.png">       
    


The CNN with *transfer learning* and *data augmentation* also classified 5 out of the 7 images correctly.

### Simple CNN 


```python
# Size of the images
x_wiki_height = 62  # height in pixels
x_wiki_width  = 47  # width in pixels

# List with the path and name of the images
file_path  = './wikipedia_images' 
file_names = [f for f in listdir(file_path) if path.isfile(path.join(file_path, f))] 

# Arrays to store the images and their true labels
x_wiki_n_images = len(file_names)
x_wiki = np.empty((x_wiki_n_images, x_wiki_height, x_wiki_width), dtype=np.int32) # images
y_wiki = np.empty((x_wiki_n_images), dtype=np.int64) # labels

# Load the images
for i, name in enumerate(file_names):
    x = Image.open(path.join(file_path, name)).convert('L') # load and convert to grayscale
    x = np.array(x)
    x_wiki[i] = x 
    y_wiki[i] = np.where(lfw_people.target_names == name.split('.')[0].replace('_', ' '))[0][0]
    
# Check the dimension of the arrays
print(x_wiki.shape)
print(y_wiki.shape)
```

    (7, 62, 47)
    (7,)
    


```python
# Add the extra dimension (see above)
x_wiki = np.expand_dims(x_wiki , axis=-1)
x_wiki.shape
```




    (7, 62, 47, 1)




```python
# Convert the values of the pixels from [0,255] to [0,1]
x_wiki_conv = x_wiki.astype('float32') / 255.0 
```


```python
# Preditions
y_wiki_conv      = model_conv.predict(x_wiki_conv, verbose=1)
y_wiki_conv_pred = np.argmax(y_wiki_conv, axis=1)
```

    1/1 [==============================] - 0s 47ms/step
    


```python
# Plot the images indicating the true and predicted labels
titles = ['True: [' + str(y_wiki[n]) + '] ' + '\n' + 'Predicted: [' + str(y_wiki_conv_pred[n]) + ']' for n in range(len(file_names)) ]
plot_gallery(x_wiki, x_wiki_n_images, titles, x_wiki_height, x_wiki_width) 
plt.tight_layout()
```


    
<img src="/assets/images/labeled_faces_in_the_wild/output_136_0.png">       
    


The simple CNN model classified 3 out of the 7 images correctly.
