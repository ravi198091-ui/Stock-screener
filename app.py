import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from nselib import capital_market
from datetime import datetime, timedelta
import concurrent.futures
import io
import requests

# --- UI Setup ---
st.set_page_config(page_title="Nifty 500 Screener", layout="wide")
st.title("📈 Quantamental Nifty 500 Screener")
st.markdown("Screens for: PE < 20 (or Industry Avg), Vol > 2x 20-Day SMA, RSI > 50, 5-Day Return >= 1%, Delivery > 45%")

# ==========================================
# FUNCTIONS
# ==========================================
@st.cache_data(ttl=3600)
def fetch_nifty500_symbols():
    url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            symbols = df.Symbol.astype(str).str.strip().tolist()
            return symbols, df
        return list(), pd.DataFrame()
    except Exception:
        return list(), pd.DataFrame()

def fetch_pe_and_industry(symbol):
    yf_symbol = f"{symbol}.NS"
    try:
        ticker = yf.Ticker(yf_symbol)
        info = ticker.info
        pe = info.get('trailingPE', np.nan)
        industry = info.get('industry', 'Unknown')
        return {'Symbol': symbol, 'PE': pe, 'Industry': industry}
    except Exception:
        return {'Symbol': symbol, 'PE': np.nan, 'Industry': 'Unknown'}

@st.cache_data(ttl=3600)
def get_last_5_trading_days_bhavcopy():
    bhavcopies = list()
    days_checked = 0
    current_date = datetime.now()
    
    while len(bhavcopies) < 5 and days_checked < 15:
        date_str = current_date.strftime("%d-%m-%Y")
        if current_date.weekday() < 5: 
            try:
                df = capital_market.bhav_copy_with_delivery(date_str)
                df.columns = df.columns.str.strip().str.upper()
                bhavcopies.append(df)
            except Exception:
                pass
        current_date -= timedelta(days=1)
        days_checked += 1
        
    if not bhavcopies:
        return pd.DataFrame()
    return pd.concat(bhavcopies, ignore_index=True)

def calculate_rsi(prices, period=14):
    delta = prices.diff(1)
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ==========================================
# MAIN EXECUTION
# ==========================================
if st.button("🚀 Run Screener Now"):
    with st.spinner("Fetching Nifty 500 Symbols..."):
        nifty500_symbols, mapping_df = fetch_nifty500_symbols()
        
    if not nifty500_symbols:
        st.error("Failed to load symbols from NSE.")
        st.stop()

    with st.spinner("Step 1: Fetching Fundamental Data (P/E & Industry)... this takes about 30 seconds."):
        fundamental_data = list()
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            results = executor.map(fetch_pe_and_industry, nifty500_symbols)
            for res in results:
                fundamental_data.append(res)
                
        fund_df = pd.DataFrame(fundamental_data)
        industry_pe = fund_df.groupby('Industry')['PE'].mean().reset_index()
        industry_pe.rename(columns={'PE': 'Industry_Avg_PE'}, inplace=True)
        fund_df = fund_df.merge(industry_pe, on='Industry', how='left')
        
        condition1 = fund_df['PE'] < 20
        condition2 = fund_df['PE'] < fund_df['Industry_Avg_PE']
        fund_df['PE_Pass'] = np.where(fund_df['PE'].notna() & (condition1 | condition2), True, False)
        
        passed_df = fund_df.loc[fund_df['PE_Pass']]
        passed_fundamental_symbols = passed_df.tolist()

    with st.spinner(f"Step 2: Checking Technicals for {len(passed_fundamental_symbols)} stocks..."):
        yf_symbols = list()
        for sym in passed_fundamental_symbols:
            yf_symbols.append(f"{sym}.NS")
            
        hist_data = yf.download(yf_symbols, period="1mo", group_by="ticker", progress=False)
        
        technical_results = list()
        for symbol in passed_fundamental_symbols:
            yf_sym = f"{symbol}.NS"
            try:
                if len(passed_fundamental_symbols) > 1:
                    df = hist_data[yf_sym].dropna()
                else:
                    df = hist_data.dropna()
                    
                if len(df) < 20: 
                    continue
                    
                close_prices = df['Close']
                volumes = df['Volume']
                
                current_close = float(close_prices.iloc[-1])
                close_5d_ago = float(close_prices.iloc[-6])
                return_5d = ((current_close - close_5d_ago) / close_5d_ago) * 100
                
                last_volume = float(volumes.iloc[-1])
                avg_vol_20d = float(volumes.rolling(window=20).mean().iloc[-2])
                current_rsi = float(calculate_rsi(close_prices).iloc[-1])
                
                if last_volume > (2 * avg_vol_20d) and current_rsi > 50 and return_5d >= 1.0:
                    technical_results.append({
                        'Symbol': symbol, 'Last_Close': round(current_close, 2),
                        'Return_5D_%': round(return_5d, 2), 'RSI_14': round(current_rsi, 2),
                        'Last_Volume': last_volume, 'Avg_Vol_20D': round(avg_vol_20d, 0)
                    })
            except Exception:
                continue
                
        tech_df = pd.DataFrame(technical_results)

    if tech_df.empty:
        st.warning("No stocks passed the technical criteria today.")
        st.stop()

    with st.spinner("Step 3: Analyzing Institutional Delivery Data..."):
        delivery_raw = get_last_5_trading_days_bhavcopy()
        
        target_col = None
        possible_cols = ('DELIV_PER', 'DELIV_QTY', 'DELIVERY')
        for col in delivery_raw.columns:
            for p_col in possible_cols:
                if p_col in col:
                    target_col = col
                    break
            if target_col:
                break
        
        if target_col:
            delivery_raw[target_col] = pd.to_numeric(delivery_raw[target_col], errors='coerce')
            avg_delivery = delivery_raw.groupby('SYMBOL')[target_col].mean().reset_index()
            avg_delivery.rename(columns={'SYMBOL': 'Symbol', target_col: 'Avg_Delivery_%'}, inplace=True)
            
            final_df = pd.merge(tech_df, avg_delivery, on='Symbol', how='left')
            final_df = pd.merge(final_df, fund_df, on='Symbol', how='left')
            
            delivery_mask = final_df > 45.0
            final_screened = final_df.loc[delivery_mask].copy().round(2)
            
            st.success(f"Screening Complete! Found {len(final_screened)} stocks.")
            st.dataframe(final_screened.sort_values(by='Avg_Delivery_%', ascending=False), use_container_width=True)
        else:
            st.error("Could not find delivery data from NSE.")
