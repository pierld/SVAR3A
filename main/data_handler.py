#%%
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import numpy as np
import matplotlib.pyplot as plt


#%%
df_q = pd.read_excel("data/data_flat.xlsx",sheet_name="QT_index")
df_p = pd.read_excel("data/data_flat.xlsx",sheet_name="P_index")
df_w = pd.read_excel("data/data_flat.xlsx",sheet_name="weights")

df_q.replace("European Union - 27 countries (from 2020)","EU27",inplace=True)
df_p.replace("European Union - 27 countries (from 2020)","EU27",inplace=True)
df_p = df_p.drop(['2023-10'],axis=1)
df_w.replace("European Union - 27 countries (from 2020)","EU27",inplace=True)


#%%
#TODO: /////////////
class CPIframe:
    def __init__(self, df_q, df_p, df_w, country):
        """
        Args:
            df_q (DataFrame): The Quantity data.
            df_p (DataFrame): The Price data.
            df_w (DataFrame): The respective Weights data.
            country (str): Location available (EU27,France,Germany,Spain).
        """
        self.country = country
        self.df_q = df_q
        self.df_p = df_p
        self.df_w = df_w
        self.sectors = dict(self.price().loc['HICP'])
    
    def qt(self):
        """
        Generates a subset of the dataframe `df_q` based on the specified `country`.
        Parameters:
            self (object): The current instance of the class `CPI`.
        Returns:
            pandas.DataFrame: Dataframe of Quantity series data with dates as rows.
        """
        x = self.df_q[self.df_q['Location']==self.country]
        x.reset_index(drop=True,inplace=True)
        x = x.drop(['Location'],axis=1)
        x = x.transpose()
        return x
    
    def price(self):
        """
        Generates a subset of the dataframe `df_p` based on the specified `country`.
        Parameters:
            self (object): The current instance of the class `CPI`.
        Returns:
            pandas.DataFrame: Dataframe of Price series data with dates as rows
        """
        x = self.df_p[self.df_p['Location']==self.country]
        x.reset_index(drop=True,inplace=True)
        x = x.drop(['Location'],axis=1)
        x = x.transpose()
        return x
    
    def weights(self):
        x = self.df_w[self.df_w['Location']==self.country]
        x.reset_index(drop=True,inplace=True)
        x = x.drop(['Location'],axis=1)
        x = x.transpose()
        return x
    
    def sector(self,col_num,transform=True):
        """
        Generates a two column [price,quantity] for the specified sector.
        Parameters:
            self (object): The current instance of the class `CPI`.
            col_num: column number in `price` & `qt` objects for the sector of interest.
            transform: 100*first diff then demean
        Returns:
            pandas.DataFrame: Dataframe of Quantity series data with dates as rows
        """
        x = pd.concat([self.price()[col_num],self.qt()[col_num]],axis=1)
        sec = list(x.loc['HICP'])[-1]
        x = x.drop(['HICP'],axis=0)
        x.columns = ['price','qt']
        x = x.apply(pd.to_numeric)
        
        if transform:
            x = 100*x.diff() 
            x = x - x.mean()
            x = x.dropna()
        x.coicop = sec
        #x.subdates = list(x.index)
        #x.reset_index(drop=True,inplace=True)
        return x

#TODO: /////////////
class sector_estimation:
    #? Besoin des résidus en fait
    
    def __init__(self,df_ts,order="auto",maxlag=24,trend="n"):
        """
        Args:
        df_ts: pandas.DataFrame from CPIframe.sector(transform=true) > demeaned
        """
        self.df_ts = df_ts.reset_index(drop=True)
        self.order = order
        self.maxlag = maxlag
        self.trend = trend
        self.model = VAR(endog=self.df_ts)
        #````
        self.estimation = self.run_estimate()
        self.aic = self.estimation['aic']
        self.bic = self.estimation['bic']
        
    def run_estimate(self):
        model_fit = {'aic':None,'bic':None}
        if self.order == "auto":
            for crit in model_fit.keys():
                model_fit[crit] = self.model.fit(maxlags=self.maxlag,ic=crit,trend=self.trend)
        else:
            for crit in model_fit.keys():
                model_fit[crit] = self.model.fit(self.order,trend=self.trend)
        return model_fit
        



#%%
# =====================================================================
eu = CPIframe(df_q, df_p, df_w, "EU27")
names = eu.price().loc['HICP']
data_test = eu.sector(56)

#%%
estimation_test = sector_estimation(data_test,order=6)

#%%
"""
for i in range(len(eu.price().columns)):
    x = eu.process(i).dropna()
    try:
        print(i," : ",x.index[0]," - ",x.index[-1]," -- ",len(x)," -- ",names[i])
        c+=1
        #if "2022" or "2023" in x.index[-1]:
        #    if "2000" in x.index[0]:
        #        print(i," : ",x.index[0]," - ",x.index[-1]," -- ",len(x)," -- ",names[i])
        #        c+=1
    except:
        print(i,' error')
print(str(c)+" variables")
"""