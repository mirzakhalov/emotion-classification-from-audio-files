"""
This file can be used to try a live prediction. 
"""
import tensorflow as tf
from tensorflow import keras
import librosa
import numpy as np
import os
import csv

from config import EXAMPLES_PATH
from config import MODEL_DIR_PATH
from config import DATA_PATH


class LivePredictions:
    """
    Main class of the application.
    """

    def __init__(self, folder):
        """
        Init method is used to initialize the main parameters.
        """
        self.files = os.listdir(folder)
        self.path = MODEL_DIR_PATH + 'Emotion_Voice_Detection_Model.h5'
        self.loaded_model = keras.models.load_model(self.path)

    def make_predictions(self):
        """
        Method to process the files and create your features.
        """
        fields = ['Filename', 'Prediction Class', 'Confidence'] 
        csvfile = open("output.csv", 'w')

        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields) 
    


        for file in self.files:
            data, sampling_rate = librosa.load("data/" + file)
            mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
            x = np.expand_dims(mfccs, axis=2)
            x = np.expand_dims(x, axis=0)
            predictions = self.loaded_model.predict_classes(x)
            confidence = max(self.loaded_model.predict(x)[0])
            label = self.convert_class_to_emotion(predictions)

            # writing the data rows 
            csvwriter.writerow([file, label, confidence])

            

    @staticmethod
    def convert_class_to_emotion(pred):
        """
        Method to convert the predictions (int) into human readable strings.
        """
        
        label_conversion = {'0': 'neutral',
                            '1': 'calm',
                            '2': 'happy',
                            '3': 'sad',
                            '4': 'angry',
                            '5': 'fearful',
                            '6': 'disgust',
                            '7': 'surprised'}

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value
        return label


if __name__ == '__main__':
    live_prediction = LivePredictions(folder=DATA_PATH)
    live_prediction.loaded_model.summary()
    live_prediction.make_predictions()
    
