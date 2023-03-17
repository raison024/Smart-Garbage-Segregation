import time
import streamlit as st
import numpy as np
from PIL import Image
import urllib.request
from utils import *

labels = gen_labels()

html_temp = '''
  <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: -50px">
    <div style = "display: flex; flex-direction: row; align-items: center; justify-content: center;">
     <center><h1 style="color: #000; font-size: 50px;"><span style="color: #0e7d73">Smart </span>Garbage</h1></center>
    <img src="https://cdn-icons-png.flaticon.com/128/1345/1345823.png" style="width: 0px;">
    </div>
    <div style="margin-top: -20px">
    <img src="https://i.postimg.cc/W3Lx45QB/Waste-management-pana.png" style="width: 400px;">
    </div>  
    </div>
    '''

st.markdown(html_temp, unsafe_allow_html=True)
html_temp = '''
    <div>
    <center><h3 style="color: #008080; margin-top: -20px">Check the type here </h3></center>
    </div>
    '''
st.set_option('deprecation.showfileUploaderEncoding', False)
st.markdown(html_temp, unsafe_allow_html=True)
opt = st.selectbox("How do you want to upload the image for classification?\n", ('Please Select', 'Upload image via link', 'Upload image from device'))
if opt == 'Upload image from device':
    file = st.file_uploader('Select', type = ['jpg', 'png', 'jpeg'])
    st.set_option('deprecation.showfileUploaderEncoding', False)
    if file is not None:
        image = Image.open(file)

elif opt == 'Upload image via link':

  try:
    img = st.text_input('Enter the Image Address')
    image = Image.open(urllib.request.urlopen(img))
    
  except:
    if st.button('Submit'):
      show = st.error("Please Enter a valid Image Address!")
      time.sleep(4)
      show.empty()

try:
  if image is not None:
    st.image(image, width = 300, caption = 'Uploaded Image')
    if st.button('Predict'):
        img = preprocess(image)

        model = model_arc()
        model.load_weights("./weights/modelnew.h5")

        prediction = model.predict(img[np.newaxis, ...])
        st.info('Hey! The uploaded image has been classified as " {} product " '.format(labels[np.argmax(prediction[0], axis=-1)]))
except Exception as e:
  st.info(e)
  pass
