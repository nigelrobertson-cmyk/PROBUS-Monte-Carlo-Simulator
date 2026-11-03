import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import numpy_financial as npf
import matplotlib.ticker as ticker

st.set_page_config(page_title='PROBUS Monte Carlo Simulator', layout='wide')
st.title('PROBUS Monte Carlo Portfolio Simulator')
st.write('Bull, Neutral, Bear and Crisis regime Monte Carlo simulator')

col1, col2 = st.columns(2)
with col1:
    opening_portfolio = st.number_input('Opening Portfolio ($)', value=100000.0, step=1000.0)
    years = st.number_input('Years to Simulate', value=30, step=1)
    monthly_cashflow = st.number_input('Monthly Contribution (+) or Withdrawal (-)', value=500.0, step=100.0)
with col2:
    annual_return = st.number_input('Expected Annual Return (%)', value=7.0, step=0.5)/100
    annual_volatility = st.number_input('Expected Annual Volatility (%)', value=15.0, step=1.0)/100
    num_sims = st.number_input('Number of Simulations', value=2000, step=100)

if st.button('Run Simulation'):
    months = int(years) * 12
    df = 5

    TRANSITIONS = {
        'bull': {'bull':0.96,'neutral':0.04,'bear':0.00,'crisis':0.00},
        'neutral': {'bull':0.05,'neutral':0.90,'bear':0.04,'crisis':0.01},
        'bear': {'bull':0.02,'neutral':0.07,'bear':0.90,'crisis':0.01},
        'crisis': {'bull':0.00,'neutral':0.20,'bear':0.30,'crisis':0.50}
    }

    REGIME_RANGES = {
        'neutral': {'return':(0.00,0.00),'vol':(0.00,0.00)},
        'bull': {'return':(0.02,0.06),'vol':(-0.07,-0.03)},
        'bear': {'return':(-0.08,-0.03),'vol':(0.05,0.15)},
        'crisis': {'return':(-0.30,-0.15),'vol':(0.15,0.35)}
    }

    def next_regime(current):
        return np.random.choice(list(TRANSITIONS[current].keys()), p=list(TRANSITIONS[current].values()))

    def draw_regime_parameters(regime):
        r_low, r_high = REGIME_RANGES[regime]['return']
        v_low, v_high = REGIME_RANGES[regime]['vol']
        return np.random.uniform(r_low,r_high), np.random.uniform(v_low,v_high)

    CALIBRATION_MONTHS = 100000
    current_regime='neutral'
    regime_shifts=[]
    return_shift,_ = draw_regime_parameters(current_regime)
    for _ in range(CALIBRATION_MONTHS):
        new_regime = next_regime(current_regime)
        if new_regime != current_regime:
            current_regime = new_regime
            return_shift,_ = draw_regime_parameters(current_regime)
        regime_shifts.append(return_shift)
    calibration_offset = -np.mean(regime_shifts)

    all_paths=np.zeros((int(num_sims), months+1))
    terminal_values=np.zeros(int(num_sims))
    bankrupt_count=0

    for sim in range(int(num_sims)):
        portfolio=opening_portfolio
        ruined=False
        current_regime='neutral'
        return_shift, vol_shift = draw_regime_parameters(current_regime)
        path=[portfolio]

        for month in range(months):
            if ruined:
                path.append(0)
                continue
            new_regime = next_regime(current_regime)
            if new_regime != current_regime:
                current_regime = new_regime
                return_shift, vol_shift = draw_regime_parameters(current_regime)

            effective_return = annual_return + calibration_offset + return_shift
            effective_volatility = max(0.02, annual_volatility + vol_shift)
            monthly_mean = np.log(max(0.001,1+effective_return))/12
            monthly_vol = effective_volatility/np.sqrt(12)
            t_std=np.sqrt(df/(df-2))
            scale=monthly_vol/t_std

            portfolio += monthly_cashflow
            if portfolio <=0:
                portfolio=0
                ruined=True
                path.append(0)
                continue

            log_return = monthly_mean + np.random.standard_t(df) * scale
            portfolio *= np.exp(log_return)
            path.append(portfolio)

        if ruined:
            bankrupt_count +=1
        all_paths[sim]=path
        terminal_values[sim]=portfolio

    median_path=np.median(all_paths,axis=0)

    naive_path=[opening_portfolio]
    portfolio=opening_portfolio
    monthly_growth=((1+annual_return)**(1/12))-1
    for _ in range(months):
        portfolio += monthly_cashflow
        if portfolio>0:
            portfolio *= (1+monthly_growth)
        else:
            portfolio=0
        naive_path.append(portfolio)

    naive_terminal=naive_path[-1]
    median_terminal=np.median(terminal_values)
    mean_terminal=np.mean(terminal_values)
    p5=np.percentile(terminal_values,5)
    p95=np.percentile(terminal_values,95)
    probability_ruin=bankrupt_count/int(num_sims)

    c1,c2,c3 = st.columns(3)
    c1.metric('Median Terminal Value', f'${median_terminal:,.0f}')
    c2.metric('Mean Terminal Value', f'${mean_terminal:,.0f}')
    c3.metric('Smooth Forecast', f'${naive_terminal:,.0f}')

    st.write(f'5th Percentile: ${p5:,.0f}')
    st.write(f'95th Percentile: ${p95:,.0f}')
    st.write(f'Probability of Ruin: {probability_ruin:.2%}')

    fig, ax = plt.subplots(figsize=(12,7))
    num_display=min(20,int(num_sims))
    sample_indices=np.random.choice(int(num_sims), size=num_display, replace=False)
    for idx in sample_indices:
        ax.plot(all_paths[idx], color='lightgray', linewidth=1)
    ax.plot(median_path, color='blue', linewidth=3, label='Median Monte Carlo Outcome')
    ax.plot(naive_path, color='green', linewidth=3, label='Smooth Growth Forecast')
    ax.axhline(0,color='red',linestyle='--',linewidth=2,label='Ruin Level')
    ax.set_title('Market Regime Monte Carlo Simulation')
    ax.set_xlabel('Months')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,p: f'${x/1_000_000:.1f}M' if abs(x)>=1000000 else f'${x:,.0f}'))
    st.pyplot(fig)
