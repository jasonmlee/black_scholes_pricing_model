import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go

st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""
<style>
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px; /* Adjust the padding to control height */
    width: auto; /* Auto width for responsiveness, or set a fixed width if necessary */
    margin: 0 auto; /* Center the container */
}
                       
.metric-call {
    background-color: #90ee90; /* Light green background */
    color: black; /* Black font color */
    margin-right: 10px; /* Spacing between CALL and PUT */
    border-radius: 10px; /* Rounded corners */
}

.metric-put {
    background-color: #ffcccb; /* Light red background */
    color: black; /* Black font color */
    border-radius: 10px; /* Rounded corners */
}  

.metric-value {
    font-size: 1.5rem; /* Adjust font size */
    font-weight: bold;
    margin: 0; /* Remove default margins */
}

.metric-label {
    font-size: 1rem; /* Adjust font size */
    margin-bottom: 4px; /* Spacing between label and value */
}
            
</style>     
""", unsafe_allow_html=True)

class BlackScholes:
    def __init__(self, time_to_maturity:float, strike: float, current_price: float, volatility: float, interest_rate: float):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def calculate_price(self):
        time_to_maturity = self.time_to_maturity
        strike = self.strike
        current_price = self.current_price
        volatility = self.volatility
        interest_rate = self.interest_rate

        d1 = (np.log(current_price / strike) + 
             (interest_rate + 0.5 * volatility ** 2) * time_to_maturity
             ) / (
                 volatility * np.sqrt(time_to_maturity)
             )
        d2 = d1 - volatility * np.sqrt(time_to_maturity)

        call_price = current_price * norm.cdf(d1) - (
            strike * np.exp(-(interest_rate * time_to_maturity)) * norm.cdf(d2)
        )

        put_price = (
            strike * np.exp(-(interest_rate * time_to_maturity)) * norm.cdf(-d2)
        ) - current_price * norm.cdf(-d1)

        self.call_price = call_price
        self.put_price = put_price

        # GREEKS
        # Delta
        self.call_delta = norm.cdf(d1)
        self.put_delta = 1 - norm.cdf(d1)

        # Gamma
        self.call_gamma = norm.pdf(d1) / (
            strike * volatility * np.sqrt(time_to_maturity)
        )
        self.put_gamma = self.call_gamma

        return call_price, put_price

with st.sidebar:
    st.title('Black-Scholes Model')
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/jason-lee-cfa/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Jason Lee`</a>', unsafe_allow_html=True)

    current_price = st.number_input("Current Asset Price", value=100, min_value=0)
    strike = st.number_input("Strike Price", value=100, min_value=0)
    time_to_maturity = st.number_input("Time to Maturity (Years)", value=1)
    volatility = st.number_input("Volatility", value=0.20)
    interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05)

    st.divider()

    st.write("### Heatmap Parameters")

    spot_min = st.number_input('Min Spot Price', min_value=0.01, value=current_price*0.8, step=0.01)
    spot_max = st.number_input('Max Spot Price', min_value=0.01, value=current_price*1.2, step=0.01)
    vol_min = st.slider('Min Volatility', min_value=0.01, max_value=1.0, value=volatility*0.5, step=0.1)
    vol_max = st.slider('Max Volatility', min_value=0.01, max_value=1.0, value=volatility*1.5, step=0.1)
    

st.title('Black-Scholes Model')

bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
call_price, put_price = bs_model.calculate_price() 

# Display Call and Put Values in colored tables
col1, col2 = st.columns([1,1], gap="small")

with col1:
    # Using the custom class for CALL value
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Value</div>
                <div class="metric-value">${call_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    # Using the custom class for PUT value
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Value</div>
                <div class="metric-value">${put_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
# Range of underlying asset prices (S) and time to expiration (T)
spot_range = np.linspace(spot_min, spot_max, 10)     # Underlying prices
vol_range = np.linspace(vol_min, vol_max, 10)     # Time to expiration

import numpy as np
import plotly.graph_objects as go

def make_plot(bs_model, spot_range, vol_range, strike, plot_type='call'):
    # Initialize arrays for call and put prices
    call_prices = np.zeros((len(spot_range), len(vol_range)))
    put_prices = np.zeros((len(spot_range), len(vol_range)))
    
    # Calculate prices for each combination of spot price and volatility
    for i, price in enumerate(spot_range):
        for j, vol in enumerate(vol_range):
            bs_temp = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=strike,
                current_price=price,
                volatility=vol,
                interest_rate=bs_model.interest_rate
            )
            bs_temp.calculate_price()
            call_prices[i, j] = bs_temp.call_price
            put_prices[i, j] = bs_temp.put_price

    # Select which price matrix to plot based on plot_type
    prices = call_prices if plot_type == 'call' else put_prices
    title = f"{plot_type.capitalize()} Option Prices Heatmap"

    # Create a text matrix with formatted option prices for display in each cell
    text_matrix = [[f"{price:.2f}" for price in row] for row in prices]

    # Create heatmap with individual call prices in each cell
    fig = go.Figure(data=go.Heatmap(
        z=prices,
        x=spot_range,
        y=vol_range,
        text=text_matrix,
        hoverongaps=False,
        colorbar=dict(title="Option Price"),
        texttemplate="%{text}",  # Use text content for each cell
        textfont={"size":10},     # Adjust font size if necessary
        reversescale=True
    ))

    # Update layout with labels and title
    fig.update_layout(
        title=title,
        xaxis_title="Spot Price",
        yaxis_title="Volatility",
    )

    return fig


plot_fig_call = make_plot(bs_model, spot_range, vol_range, strike, plot_type="call")
plot_fig_put = make_plot(bs_model, spot_range, vol_range, strike, plot_type="put")

st.divider()

st.write("#### Explore how volatility and spot price affects the price of an option for a given strike price with the below heatmap:")

plot1, plot2 = st.columns([1,1], gap="small")

with plot1:
    st.plotly_chart(plot_fig_call)

with plot2:
    st.plotly_chart(plot_fig_put)
