import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

path_to_project = r'C:\Users\Charles\Documents\Cours\3A\DeepLearning\Projet2'

##

artistDF = pd.read_csv(os.path.join(path_to_project,'artists.csv'))
artistDF = artistDF[['name','paintings']]
artistDF = artistDF.sort_values(by=['paintings'], ascending=False)
artistDF = artistDF[artistDF['paintings']>100].reset_index()
artistDF['name'] = artistDF['name'].str.replace(' ','_')
totalPaintings = artistDF['paintings'].sum()
artistDF['weight']=artistDF['paintings']/totalPaintings
print(artistDF)

print(totalPaintings)


##
imageFolderPath = os.path.join(path_to_project,r'images\images')

for artistName in artistDF['name']:
    try:
        imageName = artistName+'_1.jpg'
        imagePath = os.path.join(imageFolderPath,artistName,imageName)
        image = plt.imread(imagePath)

    except:
        print(artistName+' not found')


##
inputShape = (224, 224, 3)
nbClass = len(artistDF['name'])
##


##

resnet = tf.keras.applications.ResNet50(include_top=False, input_shape=inputShape)

resnetOutput = resnet.output
flatResnetOutput = tf.keras.layers.Flatten()(resnetOutput)
layer1 = tf.keras.layers.Dense(256, activation='relu')(flatResnetOutput)
layer2 = tf.keras.layers.Dense(256, activation='relu')(layer1)
output = tf.keras.layers.Dense(nbClass, activation='softmax')(layer2)

model_2 = tf.keras.Model(inputs=resnet.input, outputs=output)

for layer in model_2.layers[:81]:
    layer.trainable=False

for i in enumerate(model_2.layers):
    print(i[0],i[1].name, i[1].trainable)

model_2.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.0001),
              metrics=['accuracy'])
##

resnet18 = tf.keras.applications.ResNet18(include_top=False, input_shape=inputShape)

resnetOutput = resnet.output
flatResnetOutput = tf.keras.layers.Flatten()(resnetOutput)
layer1 = tf.keras.layers.Dense(256, activation='relu')(flatResnetOutput)
layer2 = tf.keras.layers.Dense(256, activation='relu')(layer1)
output = tf.keras.layers.Dense(nbClass, activation='softmax')(layer2)

model_2 = tf.keras.Model(inputs=resnet.input, outputs=output)

for layer in model_2.layers[:81]:
    layer.trainable=False

for i in enumerate(model_2.layers):
    print(i[0],i[1].name, i[1].trainable)

model_2.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.0001),
              metrics=['accuracy'])

##



epochs = 15
batchSize = 16




aug = tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split=0.2,
    horizontal_flip=True,
    brightness_range=[0.8,1],
    rotation_range=5
    )

trainGenerator = aug.flow_from_directory(
    directory=imageFolderPath,
    class_mode='categorical',
    target_size=(inputShape[0],inputShape[1]),
    batch_size=batchSize,
    subset="training",
    shuffle=True,
    classes=artistDF['name'].tolist() )

validGenerator = aug.flow_from_directory(
    directory=imageFolderPath,
    class_mode='categorical',
    target_size=(inputShape[0],inputShape[1]),
    batch_size=batchSize,
    subset="validation",
    shuffle=True,
    classes=artistDF['name'].tolist() )

stepPerEpoch = trainGenerator.n//batchSize
validStepPerEpoch = validGenerator.n//batchSize

classWeights = {}
weights = artistDF['weight'].to_list()
for i in range(len(weights)):
    classWeights[i]=weights[i]
print(classWeights)

cpPath = os.path.join(path_to_project,'checkpoints')
cpCallback = tf.keras.callbacks.ModelCheckpoint(cpPath, save_weights_only=True)

hist_2 = model_2.fit(
    x=trainGenerator,
    steps_per_epoch=stepPerEpoch,
    validation_data=validGenerator,
    validation_steps=validStepPerEpoch,
    epochs=epochs,
    shuffle=True,
    class_weight=classWeights,
    callbacks = [cpCallback]
    )

