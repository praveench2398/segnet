from keras.models import Model, Sequential
from keras.layers import Activation, Dense, BatchNormalization, Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Input, Reshape
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.optimizers import Adam, SGD
import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import re
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
filterwarnings('ignore')
plt.rcParams["axes.grid"] = False
np.random.seed(1)

class Seg():
    
    def __init__(self,Input_Image_filename,Input_mask_filename,epochs_num,val_output_path):
        
        self.Input_Image_filename=Input_Image_filename
        
        self.Input_mask_filename=Input_mask_filename
        
        self.epochs_num=epochs_num
        
        self.val_output_path=val_output_path
    
     
    def segnet(self):
        # This is the architecture of Segnet
        img_input = Input(shape= (192,256,3))
        x = Conv2D(64, (3, 3), padding='same', name='conv1',strides= (1,1))(img_input)
        x = BatchNormalization(name='bn1')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same', name='conv2')(x)
        x = BatchNormalization(name='bn2')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D()(x)
    
        x = Conv2D(128, (3, 3), padding='same', name='conv3')(x)
        x = BatchNormalization(name='bn3')(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same', name='conv4')(x)
        x = BatchNormalization(name='bn4')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D()(x)

        x = Conv2D(256, (3, 3), padding='same', name='conv5')(x)
        x = BatchNormalization(name='bn5')(x)
        x = Activation('relu')(x)
        x = Conv2D(256, (3, 3), padding='same', name='conv6')(x)
        x = BatchNormalization(name='bn6')(x)
        x = Activation('relu')(x)
        x = Conv2D(256, (3, 3), padding='same', name='conv7')(x)
        x = BatchNormalization(name='bn7')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D()(x)

        x = Conv2D(512, (3, 3), padding='same', name='conv8')(x)
        x = BatchNormalization(name='bn8')(x)
        x = Activation('relu')(x)
        x = Conv2D(512, (3, 3), padding='same', name='conv9')(x)
        x = BatchNormalization(name='bn9')(x)
        x = Activation('relu')(x)
        x = Conv2D(512, (3, 3), padding='same', name='conv10')(x)
        x = BatchNormalization(name='bn10')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D()(x)
      
        x = Conv2D(512, (3, 3), padding='same', name='conv11')(x)
        x = BatchNormalization(name='bn11')(x)
        x = Activation('relu')(x)
        x = Conv2D(512, (3, 3), padding='same', name='conv12')(x)
        x = BatchNormalization(name='bn12')(x)
        x = Activation('relu')(x)
        x = Conv2D(512, (3, 3), padding='same', name='conv13')(x)
        x = BatchNormalization(name='bn13')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D()(x)

        x = Dense(1024, activation = 'relu', name='fc1')(x)
        x = Dense(1024, activation = 'relu', name='fc2')(x)
        # Decoding Layer 
        x = UpSampling2D()(x)
        x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv1')(x)
        x = BatchNormalization(name='bn14')(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv2')(x)
        x = BatchNormalization(name='bn15')(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv3')(x)
        x = BatchNormalization(name='bn16')(x)
        x = Activation('relu')(x)
    
        x = UpSampling2D()(x)
        x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv4')(x)
        x = BatchNormalization(name='bn17')(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv5')(x)
        x = BatchNormalization(name='bn18')(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv6')(x)
        x = BatchNormalization(name='bn19')(x)
        x = Activation('relu')(x)

        x = UpSampling2D()(x)
        x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv7')(x)
        x = BatchNormalization(name='bn20')(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv8')(x)
        x = BatchNormalization(name='bn21')(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(128, (3, 3), padding='same', name='deconv9')(x)
        x = BatchNormalization(name='bn22')(x)
        x = Activation('relu')(x)

        x = UpSampling2D()(x)
        x = Conv2DTranspose(128, (3, 3), padding='same', name='deconv10')(x)
        x = BatchNormalization(name='bn23')(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(64, (3, 3), padding='same', name='deconv11')(x)
        x = BatchNormalization(name='bn24')(x)
        x = Activation('relu')(x)
    
        x = UpSampling2D()(x)
        x = Conv2DTranspose(64, (3, 3), padding='same', name='deconv12')(x)
        x = BatchNormalization(name='bn25')(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(1, (3, 3), padding='same', name='deconv13')(x)
        x = BatchNormalization(name='bn26')(x)
        x = Activation('sigmoid')(x)
        pred = Reshape((192,256))(x)

        model = Model(inputs=img_input, outputs=pred)
        return model
    
    def train(self):
        
        # Here we are reading images from image directory using glob operator
        filelist_trainx = sorted(glob.glob(self.Input_Image_filename+'*.png'))
        
        # Here we are reading masks from mask directory using glob operator
        filelist_trainy = sorted(glob.glob(self.Input_mask_filename+'*.png'))
        
        # Here we are taking two empty list so that we can append them into list after resizing and converting to array
        img_arr1=list()
        label_arr1=list()
        
        # Here we are reading each image and resize them and convert image to array and append to list
        for fname in filelist_trainx:
            img=Image.open(fname)
            size=(256,192)
            img=img.resize(size)
            img_arr=np.array(img)
            img_arr1.append(img_arr)
            
        # Here we are reading each mask and resize them and convert image to array and append to list
        for fname in filelist_trainy:
            img=Image.open(fname)
            size=(256,192)
            img=img.resize(size)
            label_arr=np.array(img)
            label_arr1.append(label_arr)
    
        # Here we are converting appended lists above to numpy array
        X_train=np.array(img_arr1)
        Y_train=np.array(label_arr1)
        
        # Here we are splitting data for training and validation
        x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size = 0.1)
        a=(len(x_val))

        model=self.segnet()
        model.compile(optimizer=Adam(lr=0.0001), loss= ["binary_crossentropy"])
        #model.summary()
        hist = model.fit(x_train,y_train,epochs= self.epochs_num,
                         batch_size= 1, validation_data=(x_val,y_val), verbose=1)
        model.save('segnet.h5')
        print('model is saved')
        
        for image in x_val:
            img_pred = model.predict(image.reshape(1,192,256,3))
            c=img_pred.reshape(192, 256)
            #plt.figure(figsize=(16,16))
            #plt.imshow(img_pred.reshape(192, 256),plt.cm.binary_r)
            for i in range(0,a):
                plt.imsave(self.val_output_path+'%d.jpg'%(i),c)

if __name__ == "__main__":
    
    Input_Image_filename=input('Enter the path of Input image for training :')
    
    Input_mask_filename=input('Enter the path of mask input for training :')
    
    epochs_num=int(input('Enter how many epochs do you want to run :'))
     
    val_output_path=input('Enter the path to save validation images :')
    
    a=Seg(Input_Image_filename,Input_mask_filename,epochs_num,val_output_path)
    
    b=a.train()
    