# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:47:18 2024

@author: thoma
"""

import streamlit as st
import fredapi
import numpy  as np
import matplotlib.pyplot as plt
import datetime

st.title('Current Regime of Macroeconomy')

# %% 

def state_fp_mahab(z,
                   z_star,
                   s2,
                   h = 1,
                   normalize=False):

    """
    # Generiert eine "getrimmte" Wahrscheinlichkeitsverteilung p basierend auf Scenario-Probability Distribution
   
    # Covariance selber 

    Parameters
    ----------
        z : array, shape (t_,k_)
        z_star : array, shape (k_, )
        alpha : scalar

    Returns
    -------
        p : array, shape (t_,k_) if k_>1 or (t_,) for k_==1
        z_lb : array, shape (k_,)
        z_ub : array, shape (k_,)
    """

    ### Pre-Tests
    try:
        t_,k_ = z.shape

    except:
        t_ = len(z)
        k_ = 1
        
    if k_ > 1:
        assert k_ == len(np.array(h)),      "Abweichung Datendimension und h"
        assert k_ == len(np.array(z_star)), "Abweichung Datendimension und z_star"

    elif k_ == 1:       
        assert isinstance(h,float) | isinstance(h,int),             "Abweichung Datendimension und h"
        assert isinstance(z_star,float) | isinstance(z_star,int),   "Abweichung Datendimension und z_star"    

    if normalize == True:
        z_mu = z.mean(axis=0)
        z_std = z.std(axis=0)
        
        z = (z-z_mu) / z_std
        z_star = (z_star - z_mu) / z_std
        s2 = 1 * h                          # h hier dann als Skalierungsvariable
        

    p_mah = np.zeros(t_)

    for t in range(0,t_):
        # p_mah[t] = np.exp(-(z[t,:] - z_star).reshape(-1,1).T@np.linalg.pinv(s2)@(z[t,:] - z_star).reshape(-1,1))
        p_mah[t] = np.exp(-(z[t] - z_star)*(1/s2)*(z[t] - z_star))
    p_mah_weighted = p_mah / np.sum(p_mah)    

    return p_mah_weighted





# %% Download via FredAPI

# Define the layout

# Sidebar layout
st.sidebar.title('Input Parameters')

# Input fields on the sidebar


### Input 1: Fred API
input1_fredapi = st.sidebar.text_input('Fred API', value='551dd59ab0588de45222e068c5da4951')

### Input 2: Time Series
   
# User input for FRED series ID
input2_series_id = st.sidebar.text_input('Enter FRED Series ID:',"USALOLITONOSTSAM")

### Input 3: Start Date
# User input for FRED series ID
input3_time_id = st.sidebar.text_input('Enter Start Date:',"2000-01-01")
time_datetime = datetime.datetime.strptime(input3_time_id,"%Y-%m-%d")

# Get Fred Download
# fred_api = "551dd59ab0588de45222e068c5da4951"
fred = fredapi.Fred(input1_fredapi)

# Retrieve data from FRED based on user input
try:
    # Fetch data from FRED
    data = fred.get_series(input2_series_id,time_datetime)     
    metadata = fred.get_series_info(input2_series_id)
    # Display the retrieved data
    # st.write('Retrieved Data:')
    # st.write(data)
except Exception as e:
    st.error(f'Error retrieving data: {e}')


# %% Transformation of Time Series
# input3_time_id = st.sidebar.text_input('Enter Start Date:',"2000-01-01")
# -> LIST! [z-score]

# %% Flexible Probabilities
# Sliders on the sidebar
# Sets Critical value z-ster
slider1 = st.sidebar.slider('Set Critical Value', min_value=-50, max_value=-1, value=-1)
st.sidebar.write(slider1)

slider2 = st.sidebar.slider('Set Bandwith', min_value=0, max_value=5, value=1)
st.sidebar.write(slider2)


### Berechnung Flexible Probabilities
fp_ = state_fp_mahab(data.values,
                   data.values[slider1],
                   s2 = np.std(data.values),
                   h = slider2,
                   normalize=True)

### Coloring
p_colors = [0 if fp_[t] >= max(fp_) else 20 if fp_[t] <= 10**-20 
            else np.interp(fp_[t], np.array([10**-20, max(fp_)]), np.array([20, 0])) for t in range(fp_.shape[0])]

### Main PLot
fig,ax = plt.subplots(figsize = (14,7))
fig.suptitle("Regime-Analyse: US Regime",fontsize=20)
ax.set_title("USA: Leading-Lagging")
ax.plot(np.array(data.index), data.values,c="gray",alpha=0.2)
# ax.plot(np.array(dates_after), z_after,c="green",alpha=0.8,label = "Aktueller Pfad")
ax.scatter(np.array(data.index), data.values, s=30, c=p_colors, marker='.', cmap='gray',label="Datenpunkte im Regime")
# Cutoff Point
ax.axhline(data.values[slider1],label = "Zielwert")
ax.axvline(np.array(data.index)[slider1],c="b",label= "Cutoff-Point")
# ax["A"].set_yticklabels([])
ax.legend(loc="best")
plt.show()
st.pyplot(fig)
