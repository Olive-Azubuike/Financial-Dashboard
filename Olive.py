import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import matplotlib.pyplot as plt

# Function to get S&P 500 stock tickers
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    return df['Symbol'].tolist()

# Function to get stock data
def get_stock_data(ticker, interval="1d"):
    stock = yf.Ticker(ticker)
    data = stock.history(period="max", interval=interval)
    return data, stock

# Function to calculate indicators
def calculate_indicators(data, indicator):
    if indicator == "MA":
        data['MA'] = data['Close'].rolling(window=50).mean()
    elif indicator == "RSI":
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
    elif indicator == "MACD":
        data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data

# Function to conduct Monte Carlo simulation
def monte_carlo_simulation(data, n_simulations, time_horizon):
    returns = data['Close'].pct_change().dropna()
    mean_return = returns.mean()
    std_return = returns.std()
    current_price = data['Close'].iloc[-1]

    simulation_results = []
    for _ in range(n_simulations):
        price_path = [current_price]
        for _ in range(time_horizon):
            price_path.append(price_path[-1] * (1 + np.random.normal(mean_return, std_return)))
        simulation_results.append(price_path)

    return simulation_results

# Function to calculate Value at Risk (VaR)
def calculate_var(simulation_results, confidence_level=0.95):
    final_prices = [result[-1] for result in simulation_results]
    var = np.percentile(final_prices, 100 * (1 - confidence_level))
    return var

# Function to update data
def update_data(selected_ticker):
    data, stock = get_stock_data(selected_ticker)
    return data, stock

# Streamlit app
# Set page configuration
st.set_page_config(page_title="Financial Dashboard", layout="wide")

# Header with a finance-related image and title
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 10px;">
        <img src="https://cdn-icons-png.flaticon.com/512/2620/2620857.png" width="50">
        <h1 style="display: inline; font-family: Arial, sans-serif; color: navy;">Financial Dashboard</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar
tickers = get_sp500_tickers()
selected_ticker = st.sidebar.selectbox("Select Stock", tickers)
update_button = st.sidebar.button("Update Data")

# Initialize data and stock
if 'data' not in st.session_state:
    st.session_state.data, st.session_state.stock = update_data(selected_ticker)

# Update data when the button is clicked
if update_button:
    st.session_state.data, st.session_state.stock = update_data(selected_ticker)

data = st.session_state.data
stock = st.session_state.stock

# I have five Tabs
tabs = st.tabs(["Summary", "Chart", "Financials", "Monte Carlo Simulation", "My Analysis"])

# Tab 1: Summary
with tabs[0]:
    st.markdown("### Company Profile")
    info = stock.info
    st.write(f"**Name:** {info.get('longName', 'N/A')}")
    st.write(f"**Sector:** {info.get('sector', 'N/A')}")
    st.write(f"**Industry:** {info.get('industry', 'N/A')}")
    st.write(f"**Description:** {info.get('longBusinessSummary', 'N/A')}")
    st.write(f"**Headquarters:** {info.get('city', 'N/A')}, {info.get('state', 'N/A')}, {info.get('country', 'N/A')}")
    st.write(f"**Address:** {info.get('address1', 'N/A')}")
    st.write(f"**Phone:** {info.get('phone', 'N/A')}")

    st.markdown("### Top Institutional Holders")
    institutional_holders = stock.institutional_holders
    if not institutional_holders.empty:
        st.write(institutional_holders)
    else:
        st.write("No top institutional holders information available.")

    # Stock Price Chart Title
    st.markdown("### Stock Price Chart")

    # Stock Price Chart 
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Stock price area plot
    area_plot = go.Scatter(
        x=data.index,
        y=data['Close'],
        fill='tozeroy',
        fillcolor='rgba(133, 133, 241, 0.2)',
        showlegend=False
    )
    fig.add_trace(area_plot, secondary_y=True)

    # Stock volume bar plot
    bar_plot = go.Bar(
        x=data.index,
        y=data['Volume'],
        marker_color=np.where(data['Close'].pct_change() < 0, 'red', 'green'),
        showlegend=False
    )
    fig.add_trace(bar_plot, secondary_y=False)

    #range selector buttons
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=3, label="3Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(step="all", label="MAX")
            ])
        )
    )

    # Customize the layout
    fig.update_layout(template='plotly_white')
    fig.update_yaxes(title_text="Volume", secondary_y=False)
    fig.update_yaxes(title_text="Close Price", secondary_y=True)

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Chart
with tabs[1]:
    st.markdown(selected_ticker + " Stock Price")
    
    # Calculate SMA
    data['SMA'] = data['Close'].rolling(window=50).mean()

    # Select plot type
    plot_type = st.radio("Select Plot Type", ["Line", "Candlestick"])

    # Select time interval
    time_interval = st.selectbox("Select Time Interval", ["Day", "Month", "Year"])

    # Create the plot figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if plot_type == "Line":
        # Line plot for stock price
        line_plot = go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Close Price'
        )
        fig.add_trace(line_plot, secondary_y=True)

        # Line plot for SMA
        sma_plot = go.Scatter(
            x=data.index,
            y=data['SMA'],
            mode='lines',
            name='SMA (50 days)',
            line=dict(color='orange')
        )
        fig.add_trace(sma_plot, secondary_y=True)

    elif plot_type == "Candlestick":
        # Candlestick plot for stock price
        candlestick_plot = go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Candlestick'
        )
        fig.add_trace(candlestick_plot, secondary_y=True)

    # Stock volume bar plot
    bar_plot = go.Bar(
        x=data.index,
        y=data['Volume'],
        marker_color=np.where(data['Close'].pct_change() < 0, 'red', 'green'),
        name='Volume'
    )
    fig.add_trace(bar_plot, secondary_y=False)

    #range selector buttons
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=3, label="3Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(step="all", label="MAX")
            ])
        )
    )

    # Customize the layout
    fig.update_layout(template='plotly_white')
    fig.update_yaxes(title_text="Volume", secondary_y=False)
    fig.update_yaxes(title_text="Close Price", secondary_y=True)

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: Financials
with tabs[2]:
    st.header(selected_ticker + " Financial Info")

    # Financial Statement Selector
    financial_statements = ["Income Statement", "Balance Sheet", "Cash Flow"]
    selected_financial_statement = st.selectbox("Select Financial Statement", financial_statements)

    # Period Selector
    periods = ["Annual", "Quarterly"]
    selected_period = st.selectbox("Select Period", periods)

    # Fetch financial data
    if selected_period == "Annual":
        if selected_financial_statement == "Income Statement":
            financial_data = stock.financials
        elif selected_financial_statement == "Balance Sheet":
            financial_data = stock.balance_sheet
        elif selected_financial_statement == "Cash Flow":
            financial_data = stock.cashflow
    else:
        if selected_financial_statement == "Income Statement":
            financial_data = stock.quarterly_financials
        elif selected_financial_statement == "Balance Sheet":
            financial_data = stock.quarterly_balance_sheet
        elif selected_financial_statement == "Cash Flow":
            financial_data = stock.quarterly_cashflow

    # Display financial data
    st.write(f"{selected_financial_statement} ({selected_period})")
    st.write(financial_data)

# Tab 4: Monte Carlo Simulation
with tabs[3]:
    st.header("Monte Carlo Simulation")

    # Number of simulations selector
    n_simulations_options = [200, 500, 1000]
    n_simulations = st.selectbox("Select Number of Simulations", n_simulations_options)

    # Time horizon selector
    time_horizon_options = [30, 60, 90]
    time_horizon = st.selectbox("Select Time Horizon (days)", time_horizon_options)

    # Conduct Monte Carlo simulation
    simulation_results = monte_carlo_simulation(data, n_simulations, time_horizon)

    # Convert simulation results to a DataFrame
    simulation_df = pd.DataFrame(simulation_results).T

    # Plot the simulation results
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(simulation_df)
    ax.axhline(y=data['Close'].iloc[-1], color='red')
    ax.set_title(f'Monte Carlo simulation for {selected_ticker} stock price in next {time_horizon} days')
    ax.set_xlabel('Day')
    ax.set_ylabel('Price')
    ax.legend([f'Current stock price is: {np.round(data["Close"].iloc[-1], 2)}'])
    ax.get_legend().legend_handles[0].set_color('red')
    st.pyplot(fig)

    # Calculate ending prices
    ending_price = simulation_df.iloc[-1]

    # Plot the histogram of ending prices
    fig, ax = plt.subplots()
    ax.hist(ending_price, bins=50)
    ax.axvline(np.percentile(ending_price, 5), color='red', linestyle='--', linewidth=1)
    st.pyplot(fig)

    # Calculate Value at Risk (VaR)
    future_price_95ci = np.percentile(ending_price, 5)
    VaR = data['Close'].iloc[-1] - future_price_95ci
    st.write(f'VaR at 95% confidence interval is: {np.round(VaR, 2)} USD')

# Tab 5: My Analysis (MACD Calculation and Analysis)
with tabs[4]:
    st.header("Moving Average Convergence Divergence (MACD) Analysis")

    # Calculate MACD
    data = calculate_indicators(data, "MACD")

    # Display MACD chart
    st.subheader(f"MACD for {selected_ticker}")
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot stock price
    ax[0].plot(data.index, data['Close'], label='Close Price')
    ax[0].set_title(f"Close Price for {selected_ticker}")
    ax[0].set_ylabel("Price")
    ax[0].legend()
    
    # Plot MACD components
    ax[1].plot(data.index, data['MACD'], label='MACD Line', color='blue')
    ax[1].plot(data.index, data['MACD_Signal'], label='Signal Line', color='red')
    ax[1].bar(data.index, data['MACD'] - data['MACD_Signal'], label='MACD Histogram', color='green')
    ax[1].set_title(f"MACD for {selected_ticker}")
    ax[1].set_xlabel("Date")
    ax[1].set_ylabel("MACD")
    ax[1].legend()
    
    st.pyplot(fig)
    
    # Display MACD values
    st.subheader("MACD Values")
    macd_data = pd.DataFrame({
        'MACD Line': data['MACD'],
        'Signal Line': data['MACD_Signal'],
        'MACD Histogram': data['MACD'] - data['MACD_Signal']
    })
    st.write(macd_data.tail(10))  # Display last 10 MACD values


