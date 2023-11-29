#%%
import pandas as pd

df_q = pd.read_excel("data/data_flat.xlsx",sheet_name="QT_index")
df_p = pd.read_excel("data/data_flat.xlsx",sheet_name="P_index")
df_w = pd.read_excel("data/data_flat.xlsx",sheet_name="weights")

df_q.replace("European Union - 27 countries (from 2020)","EU27",inplace=True)
df_p.replace("European Union - 27 countries (from 2020)","EU27",inplace=True)
df_p = df_p.drop(['2023-10'],axis=1)
df_w.replace("European Union - 27 countries (from 2020)","EU27",inplace=True)


#%%
class CPIframe:
    def __init__(self, df_q, df_p, df_w, country):
        self.country = country
        self.df_q = df_q
        self.df_p = df_p
        self.df_w = df_w
    
    def qt(self):
        x = self.df_q[self.df_q['Location']==self.country]
        x.reset_index(drop=True,inplace=True)
        x = x.drop(['Location'],axis=1)
        x = x.transpose()
        #x.columns = x.loc['HICP']
        #x = x.drop(['HICP'],axis=0)
        return x
    
    def price(self):
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
    
    def process(self,col_num):
        x = pd.concat([self.price()[col_num],self.qt()[col_num]],axis=1)
        x = x.drop(['HICP'],axis=0)
        x.columns = ['price','qt']
        return x
        


#%%
fr = CPIframe(df_q, df_p, df_w, "France")


