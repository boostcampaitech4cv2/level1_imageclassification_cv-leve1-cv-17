import torch
import streamlit as st
from model import EfficientnetB2_MD2
from utils import transform_image
import yaml
from typing import Tuple

@st.cache
def load_model() -> EfficientnetB2_MD2:
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = EfficientnetB2_MD2(num_classes=8).to(device)
    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    
    return model


def get_prediction(model:EfficientnetB2_MD2, image_bytes: bytes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = transform_image(image_bytes=image_bytes).to(device)
    out = model.forward(tensor)
    (mask_out, gender_out, age_out) = torch.split(out, [3, 2, 3], dim=1)
    pred_mask = torch.argmax(mask_out, dim=-1)
    pred_gender = torch.argmax(gender_out, dim=-1)
    pred_age = torch.argmax(age_out, dim=-1)

    return pred_mask, pred_gender, pred_age