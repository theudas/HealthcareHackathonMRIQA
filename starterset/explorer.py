import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mriqa_dataset import MRIQADataset, simulate_artefacts

dataset = MRIQADataset(
    '.',    # path to save data to
    modalities=(['T1', 'T2']),
    download=False,
)

z = st.slider("z-axis", 0, 150, 0)

subject = dataset.subjects[0]
sample = dataset._get_sample_dict_from_subject(subject)
x = sample['T1'].data

plt.imshow(x[0][z])
st.pyplot()

af = simulate_artefacts(x, artefacts=(1,0,0,0))

plt.imshow(af[0][z])
st.pyplot()