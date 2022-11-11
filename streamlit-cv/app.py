import streamlit as st

import io
import os
import yaml

from PIL import Image

from predict import load_model, get_prediction

from confirm_button_hack import cache_on_button_press

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")


root_password = 'password'


def main():
    st.title("Mask Classification Model")

    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    

    model = load_model()
    model.eval()

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg","png"])

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))

        st.image(image, caption='Uploaded Image')
        st.write("Classifying...")
        mask_pred, gender_pred, age_pred = get_prediction(model, image_bytes)
        mask = config['mask'][mask_pred.item()]
        gender = config['gender'][gender_pred.item()]
        age = config['age'][age_pred.item()]
        st.write(f'Mask: {mask}, Gender: {gender}, Age: {age}')


@cache_on_button_press('Authenticate')
def authenticate(password) ->bool:
    print(type(password))
    return password == root_password


password = st.text_input('password', type="password")

if authenticate(password):
    st.success('You are authenticated!')
    main()
else:
    st.error('The password is invalid.')