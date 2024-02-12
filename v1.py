# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:47:18 2024

@author: thoma
"""

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

# %%  Packages
import streamlit as st
import fredapi
import numpy  as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
# import xlsxwriter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff


# %% Define Sidebar Element 1: Regime Variable Download

### Header 
st.sidebar.header('Input Parameters')

### Input 1: Fred API
input1_fredapi = st.sidebar.text_input('Fred API', value='551dd59ab0588de45222e068c5da4951')

### Section - Regime Variable
st.sidebar.subheader('Regime Variable')


## Input 2: Time Series
# Dictionary of Seleted Fred Time Series
options_ts_dict = {"Leading Indicators OECD: Leading Indicators": "USALOLITONOSTSAM", 
                   "Coincident Economic Activity Index for the United States":"USPHCI"}
# Change to List
options_ts = list(options_ts_dict.keys())
# Display select box
selected_option_ts = st.sidebar.selectbox("Select an option:", options_ts)
# If no option is chosen, display text input
custom_option_ts = st.sidebar.text_input("Enter your FRED Series ID:")   
# Final selected option (either from select box or custom input)
if custom_option_ts == "":
    input2_series_id = options_ts_dict[selected_option_ts]
else:
    input2_series_id = custom_option_ts
# Display the final selected option
st.sidebar.write("Selected option:", input2_series_id)

## Input 3: Determine DL Start Date
# User input for FRED series ID
input3_time_id = st.sidebar.text_input('Download Start Date:',"1980-01-01")
# Change to Datetime
time_datetime = datetime.datetime.strptime(input3_time_id,"%Y-%m-%d")

## Input 4: Transformation
# Später noch Z-Score
# Options
options_trans = ["Levels","Perc. Chg. 12M", "Perc. Chg. 6M", "Perc. Chg. 3M"]
# Display select box
selected_option_trans = st.sidebar.selectbox("Data Transformation:", options_trans)


# %% Download: Regime Variable

# Get Fred Download
# fred_api = "551dd59ab0588de45222e068c5da4951"
fred = fredapi.Fred(input1_fredapi)

# Retrieve data from FRED based on user input

try:
    # Fetch data from FRED
    data = fred.get_series(input2_series_id,time_datetime)     
    metadata = fred.get_series_info(input2_series_id)
    st.session_state.data = data
    # Transform Data
    if selected_option_trans == "Levels":
        data = data
    elif selected_option_trans == "Perc. Chg. 12M":
        data = data.pct_change(12).dropna()
    elif selected_option_trans == "Perc. Chg. 6M":
        data = data.pct_change(6).dropna()
    elif selected_option_trans == "Perc. Chg. 3M":
        data = data.pct_change(3).dropna()
    else:
        raise TypeError("Unknown Transformation") 
    
except Exception as e:
    st.error(f'Error retrieving data: {e}')

# %% Flexible Probability: Inputs Calculation

### Inputs
st.sidebar.subheader('Regime Calculation Input')

## Set Regime Date
# Select Box based on downloaded data
option_regime_data = st.sidebar.selectbox('Choose Date for Macro-Regime', 
                              data.index[::-1])
# Transform to index position
regime_date_idxpos = data.reset_index(drop="True")[data.index == option_regime_data].idxmax()

# Sets Bandwith of Kernel
slider2 = st.sidebar.slider('Set Bandwith Parameter', min_value=0.0, max_value=5.0, value=1.0, step=0.1)

### Berechnung Flexible Probabilities
fp_ = state_fp_mahab(data.values,
                   data.values[regime_date_idxpos],
                   s2 = np.std(data.values),
                   h = slider2,
                   normalize=True)

# %% Main Window

### Main Header
st.title('Regime Analysis of the US Economy')

### Explanation of Approach
st.subheader('Empirical Approach')
st.write("The following approach determines state-dependent probabilities based on a selected economic time series and a predefined data.\n that aim at improving the Signal-to-Noise Ratio.")
st.write("Instead of using equal weights across time, this approach increase weights that are close to a determined point in time. Thereby, it overweights past data points which regimes which are close to the current one.")
st.write("The Analysis marks data points with are similar to the data point of the selected Regime Period.")
st.write("The darker the point, the higher the resemblence.") 
st.write("A point with a higher resemblence obtains a higher weight according to the Gaussian Kernel (Mahalanobis Distance):")

st.latex(r'''
         p_{t}|z^{*} = p*exp({(-((x-\mu_{x})^{'} * (h^{2})^{(-1)} * (x - \mu_{x})))})       
         ''')
st.latex(r'''
         with: \mu_{x}=E\{X|z^{*}\}) ~ and ~ h \in \mathbb{R}
         ''')
st.write("h is the Bandwith Parameter and determines the amount of data points included in the analysis.")




### Create Plotly Graph - Regime Plot
# Set Header
st.subheader('Regime Analysis')

# Create color array similar to Matplotlib code
p_colors = (fp_ - min(fp_)) / (max(fp_) - min(fp_))

# Number of Scenarios
ens = np.exp(-(fp_@np.log(fp_))) / len(data)
ens_string = f'Data Information included: {np.round(ens,2)*100}%)'

# Create scatter trace
trace = go.Scatter(
    x=np.array(data.index),
    y=data.values,
    mode='markers',
    marker=dict(
        size=5,
        color = p_colors,
        colorscale='Blues',  # Use grayscale colormap
        cmin=min(p_colors),  # Set the minimum value of the color scale
        cmax=max(p_colors),  # Set the maximum value of the color scale
        opacity=0.8
                ),
    name='Weight of Data Point based on Regime'
    )                   

# Create layout
layout = go.Layout(
    title= f'Regime-Analyse based on: {metadata["title"]}',
    xaxis=dict(title='Date',
               range=[np.min(data.index), np.max(data.index)]),  # Add appropriate title for x-axis
    yaxis=dict(title='Value',
               range=[np.min(data), np.max(data)]),  # Add appropriate title for y-axis
    legend=dict(x=0, y=-0.3),  # Adjust legend position
    annotations=[
        dict(
            x=0,
            y=np.max(data),
            xref="x",
            yref="y",
            text=ens_string,
            showarrow=False,
            font=dict(
                size=12
            )
        )
    ]
)

# Create figure
fig_plotly = go.Figure(data=[trace], 
                       layout = layout)
### Line Chart
line_trace = go.Scatter(
    x=np.array(data.index),
    y=data.values,
    mode='lines',  # Change mode to lines for a line chart
    line=dict(color='grey', width=1),  # Customize line color and width as needed
    name=metadata["title"]  # Provide a name for the line chart trace
)

# Create a vertical line - Baed on Regime Date
fig_plotly.add_shape(
    type="line",
    x0 = pd.Series(data.index).iloc[regime_date_idxpos], y0 = np.min(data),
    x1 = pd.Series(data.index).iloc[regime_date_idxpos], y1 = np.max(data),
    line=dict(color="RoyalBlue", width=1)
)
 
# Create a horizontal line - Baed on Regime Date
fig_plotly.add_shape(
    type="line",
    x0=np.min(data.index), y0=data.values[regime_date_idxpos],
    x1=np.max(data.index), y1=data.values[regime_date_idxpos],
    line=dict(color="RoyalBlue", width=1)
)


# Add the line trace to the existing figure
fig_plotly.add_trace(line_trace)

st.plotly_chart(fig_plotly)


# 
st.subheader("Probability Plot")

prob_bar = go

# Create scatter trace with mode 'lines+markers'
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=fp_, mode='lines+markers',name = "Regime Weight"))
fig.add_trace(go.Scatter(x=data.index, y=np.repeat(1/len(fp_),len(fp_)), mode='lines',name = "Equal Weight"))


fig.update_layout(yaxis=dict(tickformat='.2%'))


# Show the plot
st.plotly_chart(fig)




# Create histogram 

st.subheader("Distribution of Regime Variable")
### Create Plotly Graph - Regime Plot
b = np.round(fp_*10000,0)
b = np.array(b,dtype=int)
c = np.repeat(data.values,b)


# Group data together
hist_data = [c, data]
group_labels = ['Distribution - Regime', 'Distribution - Equal Weighted']

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, 
                         group_labels,
                         curve_type = "normal",
                         bin_size=.3,
                         show_hist=False)

st.plotly_chart(fig)


### Mean-Covariance Graph for S&P
# https://fred.stlouisfed.org/series/SP500

data_sp500 = fred.get_series("SP500",time_datetime)     
metadata = fred.get_series_info("SP500")

# Resample to get index of same type
data_ = data.resample("ME").last()
data_normalized = (data_ - data_.mean()) / data_.std()
data_sp500_ = data_sp500.resample("ME").last()
data_sp500_shift = data_sp500_.shift(-3) / data_sp500_ -1

# Combine Data
data_com = pd.DataFrame(data_normalized).join(pd.DataFrame(data_sp500_shift,columns=["SP500"]))

# Generate Data points
plt.show()
plt.scatter(data_com.iloc[:,0], data_com.iloc[:,1])







# %% Download Function
st.subheader('Download Regime Weights')

# Function to handle download button click event
def download_excel():
    # Set the file name
    file_name = 'flex_prob.xlsx'
    # Download the DataFrame as an Excel file
    fp_data = pd.Series(fp_,index=data.index)
    fp_data.name = "Flexible_Probs" 
    fp_data.to_excel(file_name, index=True)
    # Display a success message
    st.success(f"File '{file_name}' has been downloaded successfully!")

# Display the download button
if st.button('Download Excel'):
    download_excel()
