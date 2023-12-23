#%%
import pandas as pd 

df_w = pd.read_excel("data/data_flat.xlsx",sheet_name="weights")
df_q_index = pd.read_excel("data/data_flat.xlsx",sheet_name="QT_index")
df_p_index = pd.read_excel("data/data_flat.xlsx",sheet_name="P_index")
df_overall = pd.read_excel("data/overall_hicp.xlsx",sheet_name="overall")

#%%
df_w.to_csv("df_w.csv")
df_q_index.to_csv("df_q_index.csv")
df_p_index.to_csv("df_p_index.csv")
df_overall.to_csv("df_overall.csv")