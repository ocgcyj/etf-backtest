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
import sys

PATH = r"emb.xlsx"
# 1. net dynmaic gain/loss, cut loss by abs px or net 2. last 30 min not do 3. next buy better than sell, use abs px or net
    
tick_size = 0.01
commission_rate = 0.0035# ibkr fix:0.005 tired:0.0035
initial_net_asset = 50000# usd
net_asset = initial_net_asset
cash = net_asset
share = 0

discount_threshold = 0.1
stop_loss_threshold = -0.0001
tick_num = 3
position_flag = 0
stop_loss_flag = 1
cut_loss_flag = 1
    
    

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

def plot_trade():
#        plt.ioff()
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(etf_df.loc[mask, 'LAST_PRICE'].tolist(), color='blue', linewidth=0.5, label = 'etf')
        ax1.plot(nav_df.loc[mask, 'LAST_PRICE'].tolist(), color='green', linewidth=0.5, label = 'nav')
        ax1.plot(trade_plot_dic['b']['x'], trade_plot_dic['b']['y'], 'o', color='Red', markersize=2, label = 'buy')
        ax1.plot(trade_plot_dic['s']['x'], trade_plot_dic['s']['y'], 'o', color='black', markersize=2, label = 'sell')
        ymin,ymax = ax1.get_ylim()        
        ax1.vlines(x=trade_plot_dic['b']['x'], ymin=ymin, ymax=ymax, color='k', linestyle=':', linewidth=0.5)
        ax1.set_ylabel('last_px')
        ax1.ticklabel_format(useOffset=False, style='plain')
                
        ax2 = ax1.twinx()
        ax2.plot( (etf_df.loc[mask, 'LAST_PRICE'] - nav_df.loc[mask, 'LAST_PRICE']).tolist(), '-', color='orchid', linewidth=0.5)
        ax1.plot(np.nan, '--', color='orchid',  label = 'net') # Make an agent in ax2
        ax2.plot(trade_plot_dic['b']['x'], trade_plot_dic['b']['net'], 'o', color='orange', markersize=2)
        for i,j in zip(trade_plot_dic['b']['x'],trade_plot_dic['b']['net']):
            ax2.annotate(str(j),xy=(i,j), size=5)
        xmin,xmax = ax2.get_xlim()        
#        ax2.hlines(y=trade_plot_dic['b']['net'], xmin=xmin, xmax=xmax, color='k', linestyle='--', linewidth=0.5)
        ax2.set_ylabel('net')
#        ax2.grid()
          
        ax1.legend(loc = 'best', fontsize = 'x-small')
        plt.title(date) 
        plt.savefig(r"plot" + '\\' + PATH.split('.')[0] + "\\" + str(date) + '.png',  dpi = 400)
        plt.close(fig)

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
    last_exe_px_adj_s = 0
    headwind_buy_count = 0
    
    date_list = list(set(etf_df.Date.dt.date))
    date_list.sort()
#    date_list = date_list[0:20]
    
    for date in date_list:
        mask = (etf_df.Date.dt.date == date)
        # clean the position flag        
        position_flag = 0
        buy_count = 0
        gain_count = 0
        loss_count = 0 
        trade_plot_dic = {'b':{'x':[], 'y':[], 'net': []}, 's':{'x':[], 'y':[], 'net': []}}
        start_id = etf_df.loc[mask, :].iloc[0:1].index[0]
        
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
            
            if cut_loss_flag == 1 and loss_count - gain_count >= 2:
                print(date)
                break
            # buy
            if (buy_count >= 0 and position_flag == 0 and net_diff < discount_threshold) \
            and (timestamp.time() <= dt.time(15, 30) ):
                #or (buy_count > 0 and position_flag == 0 and net_diff < discount_threshold and (exe_px_adj / last_exe_px_adj_s - 1) < -0.003):
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

                trade_plot_dic[side]['x'].append(id - start_id)
                trade_plot_dic[side]['y'].append(exe_px_adj)
                trade_plot_dic[side]['net'].append(net_diff)
                continue
                
            # sell
            if position_flag == 1 and exe_px_adj - last_exe_px_adj_b >= tick_size*tick_num and net_diff > discount_threshold:
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
                gain_count += 1
                
                trade_plot_dic[side]['x'].append(id - start_id)
                trade_plot_dic[side]['y'].append(exe_px_adj)
                trade_plot_dic[side]['net'].append(net_diff)
                continue
            
            # sell stop loss
            if stop_loss_flag == 1 and position_flag == 1 and exe_px_adj - last_exe_px_adj_b <= -tick_size*10: #and (exe_px_adj / last_exe_px_adj_b) - 1 < stop_loss_threshold:
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
                loss_count += 1                
                
                trade_plot_dic[side]['x'].append(id - start_id)
                trade_plot_dic[side]['y'].append(exe_px_adj)
                trade_plot_dic[side]['net'].append(net_diff)
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
            
            trade_plot_dic[side]['x'].append(id - start_id)
            trade_plot_dic[side]['y'].append(exe_px_adj)
            trade_plot_dic[side]['net'].append(net_diff)
        
#        if str(date) == '2018-02-05':
#            sys.exit(0)
        if len(trade_plot_dic['b']['x']) > 0:
            plot_trade()
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
