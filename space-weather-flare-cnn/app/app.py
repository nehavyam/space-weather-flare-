# demo website

import os
import sys

import numpy as np
import streamlit as st
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from generate_data import make_one_image
from image_processing import process
from model import FlareCNN

st.title("Solar Flare Prediction")
st.write("Upload or generate a magnetogram image. The CNN says if a strong flare is likely in 24 hours.")

model = FlareCNN()
w = os.path.join(ROOT, "models", "flare_cnn.pt")
if os.path.exists(w):
    model.load_state_dict(torch.load(w, map_location="cpu"))
model.eval()

kind = st.selectbox("sample type", ["complex", "simple"])
seed = st.number_input("seed", 0, 999, 1)

if st.button("Generate image"):
    rng = np.random.default_rng(int(seed))
    pic = make_one_image(rng, complex_region=(kind == "complex"))
    st.session_state["pic"] = pic

up = st.file_uploader("or upload png/jpg")
if up is not None:
    from PIL import Image
    im = Image.open(up).convert("L").resize((32, 32))
    arr = np.array(im, dtype=np.float32) / 255.0
    st.session_state["pic"] = arr * 2 - 1

if "pic" in st.session_state:
    pic = st.session_state["pic"]
    st.image((pic - pic.min()) / (pic.max() - pic.min() + 1e-6), caption="magnetogram", width=250)

    x = torch.tensor(process(pic)).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()

    st.write("flare probability:", round(prob * 100, 1), "%")
    if prob >= 0.5:
        st.write("prediction: flare likely")
    else:
        st.write("prediction: no flare")
else:
    st.write("click Generate image to try")
