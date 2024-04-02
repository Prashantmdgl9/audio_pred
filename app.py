import librosa
import streamlit as st
import os
from pydub import AudioSegment
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from tensorflow.keras.models import Sequential, model_from_json
import pickle
from PIL import Image



def main():

    page = st.sidebar.selectbox("App Selections", ["Homepage", "Speech Emotion Recognition", "Emotional States"])
    if page == "Emotional States":
        st.title("Emotional States of the Drivers")
        identify()
    elif page == "Speech Emotion Recognition":
        about()
    elif page == "Homepage":
        homepage()



def about():
    set_png_as_page_bg('emotes.png')
    st.title("You don't want a distracted driver on the road, do you?")
    st.divider()
    st.subheader("What is Speech Emotion Recognition?")
    st.markdown("Speech emotion recognition (SER) is the process of detecting and analyzing emotional states from speech signals. It employs machine learning techniques to classify emotions such as happiness, sadness, anger, or neutral states. SER finds applications in human-computer interaction, customer service, healthcare, education, security, entertainment, and psychological research.")
    st.subheader("What are we doing differently?")
    st.markdown("There are multiple solutions in the market that are trained on various data and offering insights about the emotional states but...")
    st.divider()
    st.divider()
    st.divider()
    st.markdown(":blue[We _trained_] our models not only of contextual data but also on benchmarked ones such as Crema, Ravdess, Tess, and Savee to get robust models") 
    st.markdown("We aren't only generating the seven standard states -  neutral, sad, happy, surprised, afraid, angry, and disgusted but also predicting the :green[_valence_] - positive vs negative emotion and :blue[_arousal_] - energetic vs passive expressions")
    st.divider()
    
    


import base64

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return



def homepage():
    html_temp = """
    <html>
    <head>
    <style>
    body {
      background-color: #fe2631;
    }
    </style>
    </head>
    <body>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html = True)

    image = Image.open('home63.png')
    st.image(image, use_column_width = True)




def save_file(sound_file):
    # save your sound file in the right folder by following the path
    with open(os.path.join(sound_file.name),'wb') as f:
         f.write(sound_file.getbuffer())
    a = sound_file.name
    return a

#datapath = '/Users/prashantmudgal/Documents/Audio_emotions/data/ravdess/speech/'
#datapath2 = '/Users/prashantmudgal/Documents/Audio_emotions/'


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




def mfcc(data,sr,frame_length,hop_length,flatten:bool=True):
    mfcc=librosa.feature.mfcc(y = data,sr=sr)
    return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)


def rmse(data,frame_length=2048,hop_length=512):
    rmse=librosa.feature.rms(y = data, frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(rmse)


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
    #st.markdown(predictions)
    y_pred = encoder2.inverse_transform(predictions)
    st.header("The speaker in the audio is:")
    st.title(y_pred[0][0])
    #prediction("wav_file.wav")





if __name__ == "__main__":
    main()

