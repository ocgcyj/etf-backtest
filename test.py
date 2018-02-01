# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 11:52:14 2018

@author: YINGJIE
"""
#%% inti
from __future__ import division
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt 


PATH = r"emb.xlsx"

    
tick_size = 0.01
commission_rate = 0.0035# ibkr fix:0.005 tired:0.0035
initial_net_asset = 50000# usd
net_asset = initial_net_asset
cash = net_asset
share = 0

discount_threshold = 0.1
stop_loss_threshold = -0.003
tick_num = 3
position_flag = 0
stop_loss_flag = 1
    
    

def maxDrawDownChg(L):
    max_val = 0
    max_drawdown =0
    max_drawdown_chg = 0
    for i in range(0, len(L)):
        if L[i] > max_val:
            max_val = L[i]
        
        if max_val - L[i] > max_drawdown:
            max_drawdown = max_val - L[i]
            if i > 0:
                max_drawdown_chg = - max_drawdown / max_val
    return max_drawdown_chg



if __name__ == '__main__':
    
    
    etf_df = pd.read_excel(PATH, 'etf')
    etf_adj_df = pd.read_excel(PATH, 'etf')
    nav_df = pd.read_excel(PATH, 'nav')
    
    
    datetime_diff_list = list(set(nav_df.Date) - set(etf_df.Date))
    mask = nav_df.Date.isin(datetime_diff_list)
    nav_df = nav_df.loc[~mask, :].reset_index(drop=True)
    last_exe_px_adj = 0
    #%% backest
    trade_log_df = pd.DataFrame( columns = ['timestamp', 'exe_px', 'exe_px_adj', 'nav', 'net_diff', 'side', 'share', 'commission', 'market_val', 'cash', 'net_asset', 'net_asset_norm', 'chg'])
    hit_count = 0    
    last_exe_px_adj_b = 0
    headwind_buy_count = 0
    
    date_list = list(set(etf_df.Date.dt.date))
    date_list.sort()
    
    for date in date_list:
        mask = (etf_df.Date.dt.date == date)
        # clean the position flag        
        position_flag = 0
        buy_count = 0
        for id, row in etf_df.loc[mask, :].iterrows():
            net_diff = etf_df.loc[id, 'LAST_PRICE'] - nav_df.loc[id, 'LAST_PRICE']
            timestamp = etf_df.loc[id, 'Date']
            exe_px = etf_df.loc[id, 'LAST_PRICE']
            exe_px_adj = etf_adj_df.loc[id, 'LAST_PRICE']
            nav = nav_df.loc[id, 'LAST_PRICE']
#            if len(trade_log_df) > 0 and sum(trade_log_df.timestamp.dt.date == date):
#                try:
#                    last_exe_px_adj_b = trade_log_df.loc[(trade_log_df.timestamp.dt.date == date) & (trade_log_df.side == 'b'), 'exe_px_adj'].iloc[-1]
#                    last_exe_px_adj_s = trade_log_df.loc[(trade_log_df.timestamp.dt.date == date) & (trade_log_df.side == 's'), 'exe_px_adj'].iloc[-1]
#                except:
#                    pass
                
            # buy
            if (buy_count >= 0 and position_flag == 0 and net_diff < discount_threshold) :
                #or (buy_count > 0 and position_flag == 0 and net_diff < discount_threshold and exe_px_adj < last_exe_px_adj_b)
                if exe_px_adj > last_exe_px_adj_b:
                    headwind_buy_count += 1
                    
                position_flag = 1    
                side = 'b'
                share = int(cash / exe_px_adj)
                commission = max(0.35, min(commission_rate*share, exe_px_adj * share*0.005))
                market_val = exe_px_adj * share
                cash = cash - market_val - commission
                net_asset = cash + market_val
                net_asset_norm = net_asset / initial_net_asset
                chg = 0
                trade_log_df.loc[len(trade_log_df)] = [timestamp, exe_px, exe_px_adj, nav, net_diff, side, share, commission, market_val, cash, net_asset, net_asset_norm, chg]
                last_exe_px_adj_b = exe_px_adj
                buy_count += 1

                continue
                
            # sell
            if position_flag == 1 and exe_px_adj - last_exe_px_adj_b >= tick_size*tick_num and net_diff > 0.10:
                position_flag = 0
                side = 's'
                commission = max(0.35, min(commission_rate*share, exe_px_adj * share*0.005))
                market_val = exe_px_adj * share
                cash = cash + market_val - commission
                market_val = 0
                net_asset = cash + market_val
                net_asset_norm = net_asset / initial_net_asset
                chg = exe_px_adj / last_exe_px_adj_b - 1
                trade_log_df.loc[len(trade_log_df)] = [timestamp, exe_px, exe_px_adj, nav, net_diff, side, share, commission, market_val, cash, net_asset, net_asset_norm, chg]
                last_exe_px_adj_s = exe_px_adj                
                hit_count += 1
                continue
            
            # sell stop loss
            if stop_loss_flag == 1 and position_flag == 1 and (exe_px_adj / last_exe_px_adj_b) - 1 < stop_loss_threshold:# exe_px_adj - last_exe_px_adj_b <= -tick_size*2
                position_flag = 0
                side = 's'
                commission = max(0.35, min(commission_rate*share, exe_px_adj * share*0.005))
                market_val = exe_px_adj * share
                cash = cash + market_val - commission
                market_val = 0
                net_asset = cash + market_val
                net_asset_norm = net_asset / initial_net_asset
                chg = exe_px_adj / last_exe_px_adj_b - 1
                trade_log_df.loc[len(trade_log_df)] = [timestamp, exe_px, exe_px_adj, nav, net_diff, side, share, commission, market_val, cash, net_asset, net_asset_norm, chg]
                continue
            
        # close the position each day
        if position_flag == 1:
            position_flag = 0
            side = 's'
            commission = max(0.35, min(commission_rate*share, exe_px_adj * share*0.005))
            market_val = exe_px_adj * share
            cash = cash + market_val - commission
            market_val = 0
            net_asset = cash + market_val
            net_asset_norm = net_asset / initial_net_asset
            chg = exe_px_adj / last_exe_px_adj_b - 1
            trade_log_df.loc[len(trade_log_df)] = [timestamp, exe_px, exe_px_adj, nav, net_diff, side, share, commission, market_val, cash, net_asset, net_asset_norm, chg]
    
    #%% evaluate
    evaluate_df = pd.DataFrame( columns =
    ['trade_count', 'daily_count', 'max_drawdown_chg', 'sharp_ratio', 'hit_ratio', 'total_return', 
     'max_trade_gain', 'max_daily_gain', 'mean_trade_gain', 'mean_daily_gain', 
     'max_trade_loss', 'max_daily_loss', 'mean_trade_loss', 'mean_daily_loss'])
    
    daily_log_df = pd.DataFrame(columns = ['date', 'net_asset', 'net_asset_norm', 'chg'])
    daily_log_df.loc[:, 'date'] = date_list
    for id, row in daily_log_df.iterrows():
        date = daily_log_df.loc[id, 'date']
        mask = (trade_log_df.timestamp.dt.date == date)
        if sum(mask) > 0:
            daily_log_df.loc[id, 'net_asset'] = trade_log_df.loc[mask, :].iloc[-1]['net_asset']
            daily_log_df.loc[id, 'net_asset_norm'] = trade_log_df.loc[mask, :].iloc[-1]['net_asset_norm']
        elif sum(mask) == 0 and id > 0:
            daily_log_df.loc[id, 'net_asset'] = daily_log_df.loc[id - 1, 'net_asset']
            daily_log_df.loc[id, 'net_asset_norm'] = daily_log_df.loc[id - 1, 'net_asset_norm']
        else:
            daily_log_df.loc[id, 'net_asset'] = initial_net_asset
            daily_log_df.loc[id, 'net_asset_norm'] = 1
    
    daily_log_df.loc[0, 'chg'] = 0
    daily_log_df.loc[1:, 'chg'] = (daily_log_df.iloc[1:]['net_asset'].values / daily_log_df.iloc[:-1]['net_asset'].values) - 1
    
    trade_count = len(trade_log_df)/2
    daily_count = len(daily_log_df)
    max_drawdown_chg = maxDrawDownChg(daily_log_df.net_asset.tolist())
    sharp_ratio = np.sqrt(252)*daily_log_df.chg.mean() / daily_log_df.chg.std()
    hit_ratio = hit_count / (len(trade_log_df)/2)
    
    max_trade_gain = max(trade_log_df.chg)
    max_daily_gain = max(daily_log_df.chg)
    max_trade_loss = min(trade_log_df.chg)
    max_daily_loss = min(daily_log_df.chg)

    # chg not account for commission, trade_log_df
    # chg account for commission, daily_log_df
    mean_trade_gain = trade_log_df.loc[(trade_log_df.side == 's') & (trade_log_df.chg > 0), 'chg'].mean()
    mean_daily_gain = daily_log_df.loc[ (daily_log_df.chg > 0), 'chg'].mean()
    mean_trade_loss = trade_log_df.loc[(trade_log_df.side == 's') & (trade_log_df.chg < 0), 'chg'].mean()
    mean_daily_loss = daily_log_df.loc[ (daily_log_df.chg < 0), 'chg'].mean()
     
    total_return = daily_log_df.iloc[-1]['net_asset_norm'] - 1
    evaluate_df.loc[0] = \
    [trade_count, daily_count, max_drawdown_chg, sharp_ratio, hit_ratio, total_return, 
     max_trade_gain, max_daily_gain, mean_trade_gain, mean_daily_gain, 
     max_trade_loss, max_daily_loss, mean_trade_loss, mean_daily_loss]   
    
    #%% plot
    xs = daily_log_df.net_asset_norm.tolist()
    i = np.argmax(np.maximum.accumulate(xs) - xs) # end of the period
    j = np.argmax(xs[:i]) # start of period
#    maxdropdownchg = (xs[j] - xs[i]) / xs[j]
    
    plt.close("all")
    plt.figure()
    plt.plot(xs)
    plt.plot([i, j], [xs[i], xs[j]], 'o', color='Red', markersize=5)
    plt.grid()
