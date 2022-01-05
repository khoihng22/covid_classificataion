import sounddevice as sd
import soundfile as sf
from tkinter import *
from tkinter import messagebox
import numpy as np
import librosa
import pylab
from tensorflow.keras.models import load_model
import os
import librosa.display
from tensorflow.keras.models import load_model
import cv2
import tensorflow_hub as hub
import tensorflow.keras.backend as K

def get_recall(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    # precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    # f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return recall

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def Voice_rec():
    fs = 48000
      
    # seconds
    duration = 5
    myrecording = sd.rec(int(duration * fs), 
                         samplerate=fs, channels=1)
    sd.wait()
      
    # Save as FLAC file at correct sampling rate
    return sf.write('my_Audio_file.wav', myrecording, fs)
  
model = load_model('model',custom_objects={'KerasLayer':hub.KerasLayer, 'get_recall': get_recall,'get_f1': get_f1})
master = Tk()

master.title("COUGH-BASED COVID-19 DETECTION")
master.geometry("500x300")

fever_muscle_pain = IntVar()
respitory_condition = IntVar()

fever_muscle_pain_label = Label(master, text="Do you have fever or muscle pain: ")
fever_muscle_pain_label.grid(row=0, column=0)

fmp_checkbox = Checkbutton(master,variable=fever_muscle_pain, onvalue=1, offvalue=0)
fmp_checkbox.grid(row=0, column=1)

respitory_condition_label = Label(master, text="Do you have respiratory abnormal: ")
respitory_condition_label.grid(row=1, column=0)

rc_checkbox = Checkbutton(master,variable=respitory_condition, onvalue=1, offvalue=0)
rc_checkbox.grid(row=1, column=1)

def feature_extractor():
  
  name     = "F:\\Khoinh\\DoAn\\app\\my_Audio_file.wav"
  audio,sr = librosa.load(name)
  hop_length = np.floor(0.010*sr).astype(int) #10ms
  win_length = np.floor(0.020*sr).astype(int) #20ms 
  if(audio.shape > (0,)):
    #For MFCCS 
    mfccs    = librosa.feature.mfcc(y=audio,sr=sr, n_mfcc=13,hop_length=hop_length,win_length=win_length)
    mfccsscaled = np.mean(mfccs.T,axis=0)
    
    #Mel Spectogram
    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    s_db     = librosa.power_to_db(mfccs, ref=np.max)
    librosa.display.specshow(s_db)

    savepath = "mfcc.png"
    pylab.savefig(savepath, bbox_inches=None, pad_inches=0)
    pylab.close()
    img   = cv2.imread("mfcc.png")
    img   = cv2.resize(img, (224,224),interpolation = cv2.INTER_CUBIC)
    img   = np.array([img])
    features = [[fever_muscle_pain.get(), respitory_condition.get()]]
    features = np.array(features)
    mfccsscaled = np.array([mfccsscaled])
    data = []
    data.append([mfccsscaled, img, features])
    predicted = model.predict(data)
    if predicted >= 0.5:
      messagebox.showinfo('result', 'You are POSITIVE to COVID-19')
    else:
      messagebox.showinfo('result', 'You are NEGATIVE to COVID-19')
  return mfccsscaled,savepath

count = StringVar()
  
Label(master, text=" Voice Recoder (5s): "
     ).grid(row=2, sticky=W)

b = Button(master, text="Start", command=Voice_rec)
b.grid(row=2, column=3)

b = Button(master, text="Submit", command=feature_extractor)
b.grid(row=4, column=3, padx=5, pady=20)
  
mainloop()