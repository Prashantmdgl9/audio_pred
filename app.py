import librosa
import streamlit as st
import os
from pydub import AudioSegment
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from tensorflow.keras.models import Sequential, model_from_json
import pickle



def main():

    page = st.sidebar.selectbox("App Selections", ["Homepage", "About", "Identify"])
    if page == "Identify":
        st.title(" Speech Emotion Recognition")
        identify()
    #elif page == "About":
    #    about()
    #elif page == "Homepage":
    #    homepage()



def save_file(sound_file):
    # save your sound file in the right folder by following the path
    with open(os.path.join(sound_file.name),'wb') as f:
         f.write(sound_file.getbuffer())
    a = sound_file.name
    return a

datapath = '/Users/prashantmudgal/Documents/Audio_emotions/data/ravdess/speech/'
datapath2 = '/Users/prashantmudgal/Documents/Audio_emotions/'


def save_file(sound_file):
    # save your sound file in the right folder by following the path
    with open(os.path.join(sound_file.name),'wb') as f:
         f.write(sound_file.getbuffer())
    a = sound_file.name
    return a

def identify():

    st.subheader("Choose an audio file")
    uploaded_file = st.file_uploader('Select')
    if uploaded_file is not None:
        file_details = {'filename':uploaded_file.name, 'filetype':uploaded_file.type, 'filesize':uploaded_file.size}
        #st.write(uploaded_file.type)
        if uploaded_file is not None:
            audio_bytes = uploaded_file.read()
            st.audio(audio_bytes, format='audio/wav')
            #st.write("success")
            x = save_file(uploaded_file)
            sound = AudioSegment.from_file(x)
            #st.write("success2")
            z = sound.export('wav_file'+'.wav', format="wav")
            #wav_file = datapath2+'wav_file'+'.wav'
            y, sr = librosa.load(z)
            plot_spectrogram(y, sr)



        
def plot_spectrogram(y, sr):
    st.header('Spectrogram of the audio is')
    return mel_gram(y, sr)


def mel_gram(signal, sampling_rate, slider_length = 512):
    y_axis="log"
    fig = plt.figure(figsize=(10, 6))
    librosa.display.specshow(librosa.core.amplitude_to_db(librosa.feature.melspectrogram( y=signal,sr=sampling_rate)), hop_length=slider_length, x_axis="time",
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    st.pyplot(fig)
    #name = 'spects/test/spect.png'
    #fig.savefig(datapath[:-5]+name)
    #saveMel(signal)
    classify()

#path_1 = "spects/test/0Euras/"
#path_2 = "spects/test"



model_path = 'model_working/'


def zcr(data,frame_length,hop_length):
    zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(zcr)
def rmse(data,frame_length=2048,hop_length=512):
    rmse=librosa.feature.rms(data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(rmse)
def mfcc(data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
    mfcc=librosa.feature.mfcc(data,sr=sr)
    return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)

def extract_features(data,sr=22050,frame_length=2048,hop_length=512):
    result=np.array([])
    
    result=np.hstack((result,
                      zcr(data,frame_length,hop_length),
                      rmse(data,frame_length,hop_length),
                      mfcc(data,sr,frame_length,hop_length)
                     ))
    return result

with open(model_path+'scaler2.pickle', 'rb') as f:
    scaler2 = pickle.load(f)
        
with open(model_path+'encoder2.pickle', 'rb') as f:
    encoder2 = pickle.load(f)


def get_predict_feat(path):
    d, s_rate= librosa.load(path, duration=2.5, offset=0.6)
    res=extract_features(d)
    result=np.array(res)
    result=np.reshape(result,newshape=(1,2376))
    i_result = scaler2.transform(result)
    final_result=np.expand_dims(i_result, axis=2)
    
    return final_result





def classify():
    json_file = open(model_path+'CNN_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_path+"best_model1_weights.h5")
    loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    
    emotions1={1:'Neutral', 2:'Calm', 3:'Happy', 4:'Sad', 5:'Angry', 6:'Fear', 7:'Disgust',8:'Surprise'}
    res=get_predict_feat("wav_file.wav")
    predictions=loaded_model.predict(res)
    y_pred = encoder2.inverse_transform(predictions)
    st.header("The speaker in the audio is:")
    st.title(y_pred[0][0])
    #prediction("wav_file.wav")


if __name__ == "__main__":
    main()

