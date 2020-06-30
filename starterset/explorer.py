import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mriqa_dataset import MRIQADataset, simulate_artefacts
import torch

dataset = MRIQADataset(
    '.',    # path to save data to
    modalities=(['T1', 'T2']),
    download=False,
)

z = st.slider("z-axis", 0, 150, 0)

@st.cache()
def load_data():
    subject = dataset.subjects[0]
    sample = dataset._get_sample_dict_from_subject(subject)
    x = sample['T1'].data
    x = x[0]
    x -= torch.min(x)
    x /= torch.max(x)

    return x

@st.cache()
def augment(patient, settings):
    af = simulate_artefacts(patient.reshape(1, 256, 256, 150), artefacts=settings) 
    return af

@st.cache()
def make_plots(sample, z):
    af_eins = augment(sample, (1,0,0,0))
    af_zwei = augment(sample, (0,1,0,0))
    af_drei = augment(sample, (0,0,1,0))
    af_vier = augment(sample, (0,0,0,1))

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20,5))
    
    fig.suptitle('Challenge Data')
    ax1.axis('off')
    ax1.imshow(sample[:,:,z], cmap="gray")

    ax2.axis('off')
    ax2.imshow(af_eins[0,:,:,z], cmap="gray")
    
    ax3.axis('off')
    ax3.imshow(af_zwei[0,:,:,z], cmap="gray")

    ax4.axis('off')
    ax4.imshow(af_drei[0,:,:,z], cmap="gray")

    ax5.axis('off')
    ax5.imshow(af_vier[0,:,:,z], cmap="gray")

patient = load_data()
make_plots(patient, z)
st.pyplot()