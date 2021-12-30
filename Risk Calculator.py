import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from datetime import timedelta
from random import randrange
from matplotlib.pyplot import figure
from scipy.stats import multivariate_normal
from scipy.stats import norm

#Initial Variables - INSERT YOUR PORTFOLIO VARIABLES HERE
hours = 24 #Number of hours ahead you want to simulate
iterations = 1000 #Number of simulations you want to run
Currencies = ["BTC-USD", "ETH-USD", "AVAX-USD", "ADA-USD"] #Currencies in your portfolio
size = [20000, -10000, -10000, 5000] #Size of your positions in USD terms
Initial_USDC = 8000 #USDC in your dYdX account

#Import Price Data from picke files
def run_historical_simulation(hours, iterations, Currencies, size, Initial_USDC):
    result = pd.read_pickle("./Price_Data/1HOUR/" + str(Currencies[0]) + ".pkl")
    result = result[['startedAt', 'close']]
    result.loc[:, 'close'] = result['close'].astype(float)
    result.loc[:, str('returns_'+str(Currencies[0]))] = result["close"].pct_change()
    result = result.drop(['close'], axis=1)
    
    portfolio_paths = pd.DataFrame()
    liquidation_scenarios = pd.DataFrame()
    
    
    for currency in Currencies[1:]:
        df = pd.read_pickle("./Price_Data/1HOUR/" + str(currency) + ".pkl")
        df = df[['startedAt', 'close']]
        df.loc[:, 'close'] = df['close'].astype(float)
        df.loc[:, str('returns_'+str(currency))] = df["close"].pct_change()
        result = pd.merge(result, df[['startedAt',str('returns_'+str(currency))]], on=["startedAt"], how='inner',
                          suffixes=(currency, currency))
    result = result.iloc[1: , :]
    
    maintenance_margin_fraction = [0.03 if x == "BTC-USD" or x == "ETH-USD" else 0.05 for x in Currencies]
    size_abs = list(map(abs, size))
    
    Total_Maintenance_Margin_Requirement = []
    for num1, num2 in zip(size_abs, maintenance_margin_fraction):
        Total_Maintenance_Margin_Requirement.append(num1 * num2)
    Total_Maintenance_Margin_Requirement = sum(Total_Maintenance_Margin_Requirement)
    print("The total maintenace margin of this porfolio is: $", Total_Maintenance_Margin_Requirement)
    
    result_nostring = result.drop(columns="startedAt")
    
    for x in range(0,iterations):
        price_paths = np.full((hours, len(result_nostring.columns)), float(1))
        for t in range(1, hours):
            price_paths[t] = np.array(price_paths[t-1]*(1 + result_nostring.iloc[randrange(len(result_nostring))]), dtype=float)    
        #Calculate Maintenance Margin
        maintenance_margin = price_paths * size_abs * maintenance_margin_fraction
        maintenance_margin = np.sum(maintenance_margin, axis=1)

        #Calculate Total Account Value
        Total_Account_Value = Initial_USDC + np.sum(price_paths * size, axis=1)

        portfolio_paths = pd.concat([portfolio_paths, pd.DataFrame(Total_Account_Value)], axis=1)
        liquidation_scenarios = pd.concat([liquidation_scenarios, pd.DataFrame(Total_Account_Value > maintenance_margin)], axis=1)
        
    df = liquidation_scenarios.apply(pd.Series.value_counts).T
    try:
        print("The portfolio would have been liquidated in ", df[False].count(), " scenarios out of ", iterations)
    except KeyError:
        print("Your portfolio was not liquidated in any scenarios ")

    print("The average portfolio value is: ", portfolio_paths.iloc[hours-1].mean())
    print("The median portfolio value is: ", portfolio_paths.iloc[hours-1].median())
    print("The maximum portfolio value is: ", portfolio_paths.iloc[hours-1].max())
    print("The minimum portfolio value is: ", max(int(portfolio_paths.iloc[hours-1].min()), 0))
    
    VaR = np.percentile(portfolio_paths.iloc[23], 5, axis=0)
    ES = portfolio_paths.iloc[23][portfolio_paths.iloc[23] <= np.percentile(portfolio_paths.iloc[23], 5, axis=0)].mean()

    print("\nPortfolio VaR: ", VaR, "\nA VaR of ", VaR, "  suggests that we are \
    95% certain that our portfolio will be greater than ", VaR, 
         "\n in the next 24 hours")

    print("\nExpected Shortfall: ", ES, "\nOn the condition that the 24h loss is greater than the 5th percentile" 
          " of the loss distribution, it is expected that \n the portfolio will be ", ES)

    sns.displot(portfolio_paths.iloc[23])
    plt.axvline(x=portfolio_paths.iloc[23].median())
    plt.xlabel('Portfolio Value')
    
    figure(figsize=(8, 6), dpi=80)
    plt.plot(portfolio_paths)
    plt.show()
    
def run_monte_simulation(hours, iterations, Currencies, size, Initial_USDC):
    result = pd.read_pickle("./Price_Data/1HOUR/" + str(Currencies[0]) + ".pkl")
    result = result[['startedAt', 'close']]
    result.loc[:, 'close'] = result['close'].astype(float)
    result.loc[:, str('returns_'+str(Currencies[0]))] = result["close"].pct_change()
    result = result.drop(['close'], axis=1)
    
    portfolio_paths = pd.DataFrame()
    liquidation_scenarios = pd.DataFrame()
    
    
    for currency in Currencies[1:]:
        df = pd.read_pickle("./Price_Data/1HOUR/" + str(currency) + ".pkl")
        df = df[['startedAt', 'close']]
        df.loc[:, 'close'] = df['close'].astype(float)
        df.loc[:, str('returns_'+str(currency))] = df["close"].pct_change()
        result = pd.merge(result, df[['startedAt',str('returns_'+str(currency))]], on=["startedAt"], how='inner',
                          suffixes=(currency, currency))
    result = result.iloc[1: , :]
    result_nostring = result.drop(columns="startedAt")
    correlations = result.corr(method='kendall')
    
    for y in range(0,iterations):
        random_vals = multivariate_normal(cov=correlations).rvs(24)
        copula = norm.cdf(random_vals)
        distribution_objects = {}
        for x in Currencies:
            distribution_objects[x] = norm(result['returns_' + str(x)].mean(), result['returns_' + str(x)].std())

        copulas = {}
        for x in Currencies:
            copulas[x] = distribution_objects[str(x)].ppf(copula[:, Currencies.index(x)])
        
        copulas = pd.DataFrame(copulas)
        copulas = copulas + 1
        copulas.loc[0] = 1
        
        size_abs = list(map(abs, size))
        
        price_paths = np.full((hours, len(result_nostring.columns)), float(1))
        for t in range(1, 24):
            price_paths[t] = np.array(price_paths[t-1]*(1 + result_nostring.iloc[randrange(len(result_nostring))]), dtype=float)    
                
        maintenance_margin_fraction = [0.03 if x == "BTC-USD" or x == "ETH-USD" else 0.05 for x in Currencies]
        maintenance_margin = price_paths * size_abs * maintenance_margin_fraction
        maintenance_margin = maintenance_margin.sum(axis=1)
        
        Total_Account_Value = Initial_USDC + np.sum(price_paths * size, axis=1)
        
        portfolio_paths = pd.concat([portfolio_paths, pd.DataFrame(Total_Account_Value)], axis=1)
        liquidation_scenarios = pd.concat([liquidation_scenarios, pd.DataFrame(Total_Account_Value > maintenance_margin)], axis=1)

    df = liquidation_scenarios.apply(pd.Series.value_counts).T
    try:
        print("The portfolio would have been liquidated in ", df[False].count(), " scenarios out of ", iterations)
    except KeyError:
        print("Your portfolio was not liquidated in any scenarios ")

    print("The average portfolio value is: ", portfolio_paths.iloc[hours-1].mean())
    print("The median portfolio value is: ", portfolio_paths.iloc[hours-1].median())
    print("The maximum portfolio value is: ", portfolio_paths.iloc[hours-1].max())
    print("The minimum portfolio value is: ", max(int(portfolio_paths.iloc[hours-1].min()), 0))
    
    VaR = np.percentile(portfolio_paths.iloc[23], 5, axis=0)
    ES = portfolio_paths.iloc[23][portfolio_paths.iloc[23] <= np.percentile(portfolio_paths.iloc[23], 5, axis=0)].mean()

    print("\nPortfolio VaR: ", VaR, "\nA VaR of ", VaR, "  suggests that we are \
    95% certain that our portfolio will be greater than ", VaR, 
         "\n in the next 24 hours")

    print("\nExpected Shortfall: ", ES, "\nOn the condition that the 24h loss is greater than the 5th percentile" 
          " of the loss distribution, it is expected that \n the portfolio will be ", ES)

    sns.displot(portfolio_paths.iloc[23])
    plt.axvline(x=portfolio_paths.iloc[23].median())
    plt.xlabel('Portfolio Value')
    
    figure(figsize=(8, 6), dpi=80)
    plt.plot(portfolio_paths)
    plt.show()


#Run below functions to generate risk metrics
result = run_historical_simulation(hours, iterations, Currencies, size, Initial_USDC)
result = run_monte_simulation(hours, iterations, Currencies, size, Initial_USDC)
