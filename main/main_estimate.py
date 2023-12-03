#%%
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, acf
import numpy as np
import matplotlib.pyplot as plt


#%%
#df_q = pd.read_excel("data/data_flat.xlsx",sheet_name="QT_log")
#df_p = pd.read_excel("data/data_flat.xlsx",sheet_name="P_log")
df_w = pd.read_excel("data/data_flat.xlsx",sheet_name="weights")
df_q_index = pd.read_excel("data/data_flat.xlsx",sheet_name="QT_index")
df_p_index = pd.read_excel("data/data_flat.xlsx",sheet_name="P_index")

#df_q.replace("European Union - 27 countries (from 2020)","EU27",inplace=True)
#df_p.replace("European Union - 27 countries (from 2020)","EU27",inplace=True)
#df_p = df_p.drop(['2023-10'],axis=1)
df_q_index.replace("European Union - 27 countries (from 2020)","EU27",inplace=True)
df_p_index.replace("European Union - 27 countries (from 2020)","EU27",inplace=True)
df_p_index = df_p_index.drop(['2023-10'],axis=1)
df_w.replace("European Union - 27 countries (from 2020)","EU27",inplace=True)

def log_transform(cell_value):
    try:
        return np.log(float(cell_value))
    except (ValueError, TypeError):
        return cell_value

#%%
#TODO: /////////////
class CPIframe:
    def __init__(self, df_q_index, df_p_index, df_w, country):
        """
        Args:
            df_q (DataFrame): The Quantity data (log).
            df_p (DataFrame): The Price data (log).
            df_w (DataFrame): The respective Weights data.
            country (str): Location available (EU27,France,Germany,Spain).
        """
        self.country = country
        self.df_q_index = df_q_index
        self.df_p_index = df_p_index
        self.df_q = self.df_q_index.applymap(log_transform)
        self.df_p = self.df_p_index.applymap(log_transform)
        self.df_w = df_w
        self.qt = self.qt_df(df=self.df_q)
        self.price = self.price_df(df=self.df_p)
        self.qt_index = self.qt_df(df=self.df_q_index)
        self.price_index = self.price_df(df=self.df_p_index)
        self.weights = self.weights_df()
        self.sectors = dict(self.qt.loc['HICP'])
    
    def qt_df(self,df):
        """
        Generates a subset of the dataframe `df_q` based on the specified `country`.
        Parameters:
            df: quantity dataframe attribute current instance of the class `CPI`.
        Returns:
            pandas.DataFrame: Dataframe of Quantity series data with dates as rows.
        """
        x = df[df['Location']==self.country]
        x.reset_index(drop=True,inplace=True)
        x = x.drop(['Location'],axis=1)
        x = x.transpose()
        return x
    
    def price_df(self,df):
        """
        Generates a subset of the dataframe `df` based on the specified `country`.
        Parameters:
            df: price dataframe attribute current instance of the class `CPI`.
        Returns:
            pandas.DataFrame: Dataframe of Price series data with dates as rows
        """
        x = df[df['Location']==self.country]
        x.reset_index(drop=True,inplace=True)
        x = x.drop(['Location'],axis=1)
        x = x.transpose()
        return x
    
    def weights_df(self):
        x = self.df_w[self.df_w['Location']==self.country]
        x.reset_index(drop=True,inplace=True)
        x = x.drop(['Location'],axis=1)
        x = x.transpose()
        return x
    
    def sector(self,col_num,transform=True):
        """
        Generates a two column [price,quantity] for the specified sector.
        Uses Log transformed data
        Parameters:
            col_num: column number in `price` & `qt` objects for the sector of interest.
            transform: if True: 
                    1. 100*first_diff 
                    2. Demean
        Returns:
            pandas.DataFrame: Dataframe of Quantity series data with dates as rows
        """
        x = pd.concat([self.price[col_num],self.qt[col_num]],axis=1)
        sec = list(x.loc['HICP'])[-1]
        x = x.drop(['HICP'],axis=0)
        x.columns = ['price','qt']
        x = x.apply(pd.to_numeric)
        
        if transform:
            x = 100*x.diff() 
            x = x - x.mean()
        x = x.dropna()
        x.coicop = sec
        x.col = col_num
        #x.subdates = list(x.index)
        #x.reset_index(drop=True,inplace=True)
        return x

#TODO: Sheremirov labeling 
class sector_estimation:
    def __init__(self,meta,col,transform=True,order="auto",maxlag=24,trend="n"):
        """
        Args:
            meta: CPIframe object
        """
        self.meta = meta
        self.df_ts_origin = meta.sector(col_num=col,transform=transform) #DataFrame from CPIframe.sector(...,transform=true) > demeaned
        self.df_ts = self.df_ts_origin.reset_index(drop=True)
        self.order = order
        self.maxlag = maxlag
        self.trend = trend
        #````
        self.model = VAR(endog=self.df_ts)
        self.estimation = self.run_estimate()
        self.aic = self.estimation['aic']
        self.bic = self.estimation['bic']
        self.aic.shapiro = self.shapiro_label(model_resid=self.aic.resid)
        self.bic.shapiro = self.shapiro_label(model_resid=self.bic.resid)
        self.sheremirov = self.sheremirov_label()
        
    def run_estimate(self):
        model_fit = {'aic':None,'bic':None}
        if self.order == "auto":
            for crit in model_fit.keys():
                model_fit[crit] = self.model.fit(maxlags=self.maxlag,ic=crit,trend=self.trend)
        else:
            for crit in model_fit.keys():
                model_fit[crit] = self.model.fit(self.order,trend=self.trend)
        return model_fit

    def shapiro_label(self,model_resid):
        x = model_resid.copy()
        x[['sup+','sup-','dem+','dem-']] = ""
        # demand shock
        x['dem+'] = np.where((x['qt']>0) & (x['price']>0),1,0)
        x['dem-'] = np.where((x['qt']<0) & (x['price']<0),1,0)
        # supply shock
        x['sup+'] = np.where((x['qt']>0) & (x['price']<0),1,0)
        x['sup-'] = np.where((x['qt']<0) & (x['price']>0),1,0)
        return x
    
    def sheremirov_label(self):
        return self.meta.df_p



     
#TODO: boucle sur l'ensemble des secteurs (tester si assez de données) puis extraire résidus des estimation > fichier excel 
#%%
#? =====================================================================
eu = CPIframe(df_q_index=df_q_index, df_p_index=df_p_index, df_w=df_w, country="EU27")
names = eu.price.loc['HICP']
data_test = eu.sector(56)

#? =====================================================================
#%%
etest = sector_estimation(meta=eu,col=56)
#print(estimation_test.aic.plot_acorr(25))
#restest = etest.aic.resid
#test = eu.sector(56,False)[['price']].copy()
#cycle, trend = sm.tsa.filters.hpfilter(test, 1600*3**4)






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