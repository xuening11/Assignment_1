#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd

df_cluster = pd.read_csv('subscribers_cleaned_dummified.csv').fillna(0)
convert_TF = []
for i in range(0, 63417):
    if df_cluster.iloc[i]['payment_period'] == 0.0 and df_cluster.iloc[i]['current_sub_TF'] == 1.0:
        convert_TF.append('Check')
    elif df_cluster.iloc[i]['payment_period'] == 0.0 and df_cluster.iloc[i]['current_sub_TF'] == 0.0:
        convert_TF.append(False)
    else:
        convert_TF.append(True)

df_cluster['convert_TF'] = convert_TF

df_attri = df_cluster[(df_cluster['attribution_technical_bing'] == 1.0)
                      | (df_cluster['attribution_technical_display'] == 1.0)
                      | (df_cluster['attribution_technical_facebook'] == 1.0)
                      | (df_cluster['attribution_technical_search'] == 1.0)
                      | (df_cluster['attribution_technical_youtube'] == 1.0)]


print(df_attri)


# In[60]:


df_attri = df_attri.loc[:, ['subid', 'tier','convert_TF','attribution_technical_bing', 'attribution_technical_display', 
                 'attribution_technical_facebook', 'attribution_technical_youtube', 'attribution_technical_search',
                           'join_fee', 'monthly_price', 'payment_period']]


# In[84]:


attribution_technical = list()
for i in range(len(df_attri)):
    if df_attri.loc[i,:][3] == 1:
        attribution_technical.append('bing')
    elif df_attri.loc[i,:][4] == 1:
        attribution_technical.append('display')
    elif df_attri.loc[i,:][5] == 1:
        attribution_technical.append('facebook')
    elif df_attri.loc[i,:][6] == 1:
        attribution_technical.append('youtube')
    else: 
        attribution_technical.append('search')


# In[85]:


df_attri['attribution_technical'] = attribution_technical
df_attri


# In[86]:


import json
channel_spend_dict = {}
channel_spend = pd.read_csv('channel_spend_undergraduate.csv')
tier_list = ['tier1', 'tier2', 'tier3','tier4', 'tier5', 'tier6','tier7', 'tier8']
for i in range(8):
    channel_spend_dict[tier_list[i]] = json.loads(channel_spend['spend'][i].replace("'", '"'))
# channel_spend_dict
channel_spend = channel_spend_dict


# In[87]:


channel_spend['tier1'].keys()


# In[88]:


# ----- Set parameters -----
touch_col_prepend = 'touch'
direct_label = 'direct'
first_weight = 0.4
last_weight = 0.4
cred_col_post_pend = '_credit'
write_to_file = True

# total spending for all 8 tier experiments
channel_spend['total'] = dict()
for t_name, t in channel_spend.items():
    if t_name != 'total':
        for c in t.keys():
            try:
                channel_spend['total'][c] = channel_spend['total'][c] + t[c]
            except KeyError:
                channel_spend['total'].update({c: t[c]})


# In[89]:


channel_spend


# In[90]:


# ----- Format dataframe -----
# --- create credit columns
cred_col_names = ['bing_credit','display_credit','facebook_credit','search_credit','youtube_credit']


# In[91]:


print(df_attri[df_attri['attribution_technical_facebook'] == 1].shape)
print(df_attri[df_attri['attribution_technical_bing'] == 1].shape)
print(df_attri[df_attri['attribution_technical_display'] == 1].shape)
print(df_attri[df_attri['attribution_technical_search'] == 1].shape)
print(df_attri[df_attri['attribution_technical_youtube'] == 1].shape)


# In[95]:


def assign_credit(t_row, cred_col_names_f, cred_col_post_pend_f, model_type_f, first_weight_f=0.5, last_weight_f=0.5):
    # function assigns a credit to each relevant channel based on user specified model type, e.g. "last_touch_point", "first_touch_point", etc.
    t_dict = dict(zip(cred_col_names_f, [0]*len(cred_col_names_f)))

    if model_type_f == 'attribution_technical':
        if t_row['attribution_technical'] == 'bing' or t_row['attribution_technical'] == 'display'or t_row['attribution_technical'] ==  'facebook' or t_row['attribution_technical'] == 'search' or t_row['attribution_technical'] == 'youtube':
            t_dict.update({t_row['attribution_technical'] + cred_col_post_pend_f: 1})
        return t_dict
    
def get_attribution_by_channel(df_f, credit_col_postpend_f):
    allocated_conversions = df_f[cred_col_names].sum()
    print(allocated_conversions)
    n_allocated_conversions = df_f[cred_col_names].sum().sum()
    print(n_allocated_conversions)
    n_total_conversions = df_f['convert_TF'].sum()
    channel_allocation_f = pd.Series(dict(zip([x.split(credit_col_postpend_f)[0] for x in allocated_conversions.keys()], list(allocated_conversions.array))))
    return channel_allocation_f

def calc_avg_CAC(channel_allocation_f, channel_spend_f):
    t_df = pd.DataFrame(channel_allocation_f)
    t_df.columns = ['channel_allocation']
    for t_ind, _ in t_df.iterrows():
        t_df.loc[t_ind, 'channel_spend'] = channel_spend_f[t_ind]

    t_df['CAC'] = t_df['channel_spend'] / t_df['channel_allocation']
    t_df['CAC'].replace(np.inf, 0, inplace=True)
    return t_df

def calc_marginal_CAC(n_conversions_low_tier, spend_low_tier, n_conversions_high_tier, spend_high_tier):
    ##### fill in this code to create the three variables in output dictionary
    marginal_conversions = n_conversions_high_tier - n_conversions_low_tier
    marginal_spend = spend_high_tier - spend_low_tier
    marginal_CAC = marginal_spend/marginal_conversions
    return {'marginal_conversions': marginal_conversions, 'marginal_spend': marginal_spend,
            'marginal_CAC': marginal_CAC}


# In[98]:


df_attri.convert_TF = df_attri.convert_TF.fillna(False)
import numpy as np


# In[116]:


# ----- RUN MODELS -----
CAC_dfs = dict()
model_type = 'attribution_technical'
print('Processing model %s' % model_type)

# ----- Run attribution model -----
print('Running attribution model')
df_convert = df_attri.loc[df_attri['convert_TF']==True] # only run calculation for conversion rows
info_to_add = list()
for t_ind, t_row in df_convert.iterrows():
    t_credit_dict = assign_credit(t_row, cred_col_names, cred_col_post_pend, model_type, first_weight, last_weight)
    info_to_add.append(t_credit_dict)
df_convert = pd.concat([df_convert.reset_index(drop=True), pd.DataFrame(info_to_add).reset_index(drop=True)], axis=1)

# ----- Calculate CAC -----
print('Calculating average and marginal CAC')
# --- Average CAC ---
channel_allocation = get_attribution_by_channel(df_convert, credit_col_postpend_f='_credit')
df_CAC = calc_avg_CAC(channel_allocation, channel_spend['total'])
print('Average:') 
print(df_CAC)
df_CAC.to_csv('average_cac_technical.csv')

# --- Marginal CAC ---
credit_cols = [x for x in df_convert.columns if x.find('credit') > -1]
df_CAC = pd.DataFrame(index=[x.split('_credit')[0] for x in credit_cols])
base_col_names = ['marginal_conversions', 'marginal_spend', 'marginal_CAC']

df_tier_sum = df_convert[['tier']+credit_cols].groupby(['tier']).sum()
df_tier_sum.columns = [x.split('_credit')[0] for x in df_tier_sum.columns]
for t_tier in df_tier_sum.index:
    for t_channel in df_CAC.index:
        if t_tier > 1:
            n_conversions_low_tier = df_tier_sum.loc[t_tier - 1, t_channel]
            spend_low_tier = channel_spend['tier' + str(t_tier - 1)][t_channel]
            n_conversions_high_tier = df_tier_sum.loc[t_tier, t_channel]
            spend_high_tier = channel_spend['tier' + str(t_tier)][t_channel]
        else:
            n_conversions_low_tier = 0
            spend_low_tier = 0
            n_conversions_high_tier = df_tier_sum.loc[t_tier, t_channel]
            spend_high_tier = channel_spend['tier' + str(t_tier)][t_channel]

        t_df_CAC_colnames = [x + '_t' + str(t_tier) for x in base_col_names]
        for i in t_df_CAC_colnames:
            if i not in list(df_CAC.columns):
                df_CAC[i] = float('nan')

        t_marginal_dict = calc_marginal_CAC(n_conversions_low_tier, spend_low_tier, n_conversions_high_tier, spend_high_tier)
        df_CAC.loc[t_channel, t_df_CAC_colnames] = [t_marginal_dict[x] for x in base_col_names]
CAC_dfs.update({model_type: df_CAC})


# In[119]:


CAC_dfs


# In[120]:


CAC_dfs['attribution_technical'].loc[['bing', 'display', 'facebook', 'search', 'youtube']]


# In[121]:


# write marginal CAC output
if write_to_file:
    for key, value in CAC_dfs.items():
        with open(key + '_model_marginal_implied_CAC.csv', 'w') as f:
            value.to_csv(f)


# In[122]:


df_convert[['join_fee', 'payment_period','monthly_price']] = df_convert[['join_fee', 'payment_period','monthly_price']].apply(pd.to_numeric)


# In[123]:


df_convert['revenue'] = df_convert['join_fee']+df_convert['payment_period']*4*df_convert['monthly_price']


# In[124]:


cac_technical =[]

for i in range(len(df_convert)):
    if df_convert.loc[i,'attribution_technical'] == 'bing':
        tier = df_convert.loc[i,'tier']
        cac_technical.append(CAC_dfs['attribution_technical'].loc['bing','marginal_CAC_t'+str(tier)])
    elif df_convert.loc[i,'attribution_technical'] == 'youtube':
        tier = df_convert.loc[i,'tier']
        cac_technical.append(CAC_dfs['attribution_technical'].loc['youtube','marginal_CAC_t'+str(tier)])
    elif df_convert.loc[i,'attribution_technical'] == 'facebook':
        tier = df_convert.loc[i,'tier']
        cac_technical.append(CAC_dfs['attribution_technical'].loc['facebook','marginal_CAC_t'+str(tier)])
    elif df_convert.loc[i,'attribution_technical'] == 'search':
        tier = df_convert.loc[i,'tier']
        cac_technical.append(CAC_dfs['attribution_technical'].loc['search','marginal_CAC_t'+str(tier)])
    elif df_convert.loc[i,'attribution_technical'] == 'display':
        tier = df_convert.loc[i,'tier']
        cac_technical.append(CAC_dfs['attribution_technical'].loc['display','marginal_CAC_t'+str(tier)])
    else:
        cac_technical.append(0)


df_convert['marginal_cac_technical'] = cac_technical


# In[125]:


df_technical = pd.read_csv('attribution_technical_model_marginal_implied_CAC.csv')
df_convert['CLV_technical'] = df_convert['revenue']-df_convert['marginal_cac_technical']
df_convert['CLV/CAC_ratio_technical'] = df_convert['CLV_technical']/df_convert['marginal_cac_technical']
df_convert.to_csv('clv_cac_analysis.csv')
df_convert


# In[127]:


df_convert.groupby(['tier','attribution_technical'])['CLV_technical'].count()


# In[130]:


df_convert.groupby(['tier','attribution_technical'])['CLV_technical'].sum()


# In[132]:


tier_channel_clv_info = df_convert.groupby(['tier','attribution_technical'])['CLV_technical'].sum()/df_convert.groupby(['tier','attribution_technical'])['CLV_technical'].count()


# In[133]:


tier_channel_clv_info.to_csv('tier_channel_clv_info.csv')


# In[ ]:




