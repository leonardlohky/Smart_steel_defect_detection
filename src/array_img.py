import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import pandas as pd
import numpy as np
import os 
import cv2
import shutil
from PIL import Image
import matplotlib.pyplot as plt
#%matplotlib inline
#%env SM_FRAMEWORK=tf.keras
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import Model, load_model
import tensorflow as tf
import segmentation_models as sm
from segmentation_models import Unet
from segmentation_models import get_preprocessing    
    
class img_array_gen(keras.utils.Sequence):
    
    input_folder = '../Input/'

    def __init__(self, df, batch_size = 1, image_path = input_folder, preprocess=None, info={}):
        super().__init__()
        self.df = df
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.info = info
        self.data_path = image_path
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))

    def __getitem__(self, index): 
        '''
        The DataGenerator takes ImageIds of batch size 1 and returns Image array to the model.
        With the help of ImageIds the DataGenerator locates the Image file in the path, the image is read and resized from
        256x1600 to 256x800.
        '''
        X = np.empty((self.batch_size,256,800,3),dtype=np.float32)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        for i,f in enumerate(self.df['ImageId'].iloc[indexes]):
            self.info[index*self.batch_size+i]=f
            X[i,] = Image.open(self.data_path + f).resize((800,256))      
        if self.preprocess!=None: X = self.preprocess(X)

        return X