import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.preprocessing import image
from PIL import Image


tf.flags.DEFINE_string('model_path', "model-weights/xception.h5", '')
tf.flags.DEFINE_integer('img_size', 224, "square images acquired")
FLAGS = tf.flags.FLAGS


Index2Class = { 0: 'cat', 1: 'dog'}

def load_img(input_image, shape):
    img = Image.open(input_image).convert('RGB')
    img = img.resize((shape, shape))
    img = image.img_to_array(img)
    return np.reshape(img, [1, shape, shape, 3])/255

@st.cache(allow_output_mutation=True)
def load_own_model(path):
    return load_model(path)

if __name__ == "__main__":

    result = st.empty()
    uploaded_img = st.file_uploader(label='upload your image:')
    if uploaded_img:
        st.image(uploaded_img, caption="the input image", width=350)
        result.info("please wait for your results")
        model = load_own_model(FLAGS.model_path)
        pred_img = load_img(uploaded_img, FLAGS.img_size)
        pred = Index2Class[np.argmax(model.predict(pred_img))]
        result.success("it is a(an) %s inside " % pred)
