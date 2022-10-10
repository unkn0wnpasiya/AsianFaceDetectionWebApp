import streamlit as slit
import time
import os.path
from PIL import Image
from datetime import datetime as dt
from transformers import ViTFeatureExtractor, ViTForImageClassification
from hugsvision.inference.VisionClassifierInference import VisionClassifierInference



@slit.cache
def loadModel():
    model_path = '/mnt/c/Users/pdaks/Downloads/model/content/AsianFacesOut/ASIANFACES/3_2022-10-04-17-20-28/model'
    classifier = VisionClassifierInference(
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k'),
        model = ViTForImageClassification.from_pretrained(model_path),
    )
    return classifier

def theModel():
    slit.markdown('***')
    slit.header('Real Time Prediction')
    uploaded_files = slit.file_uploader("Upload the photo here: ", accept_multiple_files=False)
    try:
        with slit.container():
            if uploaded_files is not None:
                image = Image.open(uploaded_files)
                slit.markdown('***')
                with slit.expander("To see the uploaded photo:"):
                    slit.image(image)
                if slit.button('Predict'):
                    slit.markdown('***')
                    with slit.spinner('Wait for it...'):
                        start = dt.now()
                        classifier = loadModel()
                        label = classifier.predict(img_path=uploaded_files)
                        running_secs = (dt.now() - start).microseconds
                    slit.success(f'Done!, predicted as a: {label}')
                    slit.write(running_secs, 'μs')
    except:
        slit.warning('Something went wrong :exclamation:')

def firstPage():
    slit.title('Asian Faces Detection')
    slit.header('About Us: :+1:')
    slit.text('Ziyi Luu -> 10495781 ')
    slit.text('Mohamed Bishr -> 10507845 ')
    slit.text('Boh Yee Choong -> 10484578 ')
    slit.text('Pasindu Jayasekara -> 10521966 ')
    slit.markdown('***')
    

def main():
    # infoTeam()
    firstPage()
    theModel()
    slit.markdown('***')


if __name__=='__main__':
    main()