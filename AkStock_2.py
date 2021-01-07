import akshare as ak
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil 
# import time



class AkStock:
    INPUT   = './input/'                # All input file.
    BASIC   = '../AkShare/1.basic/'         # The self.basic document.
    AKSHARE = '../AkShare/2.raw/'  # Raw data .
    STOCK   = '../AkShare/3.stock/'         # self.Stock data for visualization.
    REPORT  = '../AkShare/4.report/'        # self.Stock self.Report

    def __init__(self):
        print('Algorithmic trading platform versoin 1.0 by Tao Zhang in Shanghai on December 8, 2020.')
        return


    def clean(self):
        for path in [self.AKSHARE, self.BASIC, self.STOCK, self.REPORT]:
            if os.path.isdir(path):
                shutil.rmtree(path)
                os.makedirs(path)
            else:
                os.makedirs(path)

        for path in [self.AKSHARE, self.STOCK, self.REPORT ]:
            os.makedirs(path +'trade/')
            os.makedirs(path +'favourite/')
            os.makedirs(path +'archive/')
        return 

    def code(self):
        etf = ak.fund_em_etf_fund_daily()         
        etf = etf[['基金代码', '基金简称', '类型', ]]
        etf.columns = ['code','name', 'market']
        etf['market'] = 'ETF'  
        etf['symbol'] = etf['code'].apply(lambda x:'sh' + str(x)[:6] if int(x) > 500000  else 'sz' + str(x)[:6])
        etf['tag'] = etf['symbol'].apply(lambda x: 'E_' + str(x))
        etf = etf[['name', 'code', 'symbol', 'tag', 'market']]
    
        index = ak.stock_zh_index_spot()        
        index['code'] = index['symbol'].apply(lambda x:str(x)[2:])
        index['tag'] = index['symbol'].apply(lambda x:'I_' + str(x))
        index.loc[index['name']=='','name']='TBD'
        index['market'] = 'INDEX'
        index = index[['name','code','symbol', 'tag','market']]
        
        stock = ak.stock_zh_a_spot()    
        stock['tag'] = stock['symbol'].apply(lambda x: 'S_' + str(x))
        stock.to_csv(self.BASIC + 'stock_spot.csv', encoding='gb18030', index=False)
        stock['tag'] = stock['symbol'].apply(lambda x:'S_' + str(x))
        stock['market'] = 'STOCK'
        stock = stock[['name','code','symbol', 'tag','market']]
        
        df = pd.concat([etf,index,stock],axis=0)   
        df = df.append({'name':'银证转账', 'code':'817909','symbol':'sh817909','tag':'C_sh817909','market':'TRADE'},ignore_index=True)
        df = df.append({'name':'测试账号','code':'817910', 'symbol':'sh817910','tag':'C_sh817910','market':'TRADE'},ignore_index=True) 
        df['name'] = df['name'].apply(lambda x: x[:-5] if x[-5:] == 'ETF行情' else x)  
        df['name'] = df['name'].apply(lambda x: x[:-2] if x[-2:] == '行情' else x)     
        
        df=df[df['name'].str.contains('ST')==False] 
        df.to_csv(self.BASIC + 'code_name.csv', encoding='gb18030', index=False)
        return 

    
    def tradelog(self):
        if os.path.exists(self.INPUT + 'self.stocklogfile.xls'):
            df = pd.read_excel(self.INPUT + 'self.stocklogfile.xls',index_col='交收日期', parse_dates=True).reset_index()
            df = df[['交收日期', '业务名称', '证券代码', '成交价格','成交数量','发生金额']]
            df.columns = ['date','type', 'tag', 'price', 'qty', 'amount']
    
            df['type'] = df['type'].apply(lambda x:
                        'buy'  if x == '证券买入'   else (
                        'sell' if x == '证券卖出'   else (
                        'sell' if x == '银行转证券' else (
                        'buy'  if x == '证券转银行' else (
                        'buy'  if x == '红利入账'   else (
                        'tax'  if x == '股息红利差异扣税' else
                        'tax' ))))))
        
            df['tag'] = df['tag'].apply(lambda x:
                    'S_' + 'sh' + str(x)[:6] if x >= 600000 else (
                    'E_' + 'sh' + str(x)[:6] if x > 510000  else (
                    'I_' + 'sz' + str(x)[:6] if x > 399000  else (
                    'E_' + 'sz' + str(x)[:6] if x > 150000  else (
                    'S_' + 'sz' + str(x)[:6] if x > 1000    else 'C_sh817909')))))
    
            df['amount'] = -1 * df['amount']
            df['qty'] = df['qty'] * df['amount'] /abs(df['amount'])
            df.loc[df['price'] == 0,'qty'] = 0
            df = df.drop(df[df.type == 'tax'].index)
            df = df.reset_index().pivot_table(index=['date','tag','type',],values=['qty','amount'],aggfunc=[np.sum])
            df.columns = ['amount','qty']
            df.to_csv(self.BASIC + 'trade_book.csv', encoding='gb18030')
    
            tickers = sorted(list(set([] + df.reset_index()['tag'].values.tolist())))
            tickers.remove('C_sh817909')
            self.stock = pd.DataFrame(tickers,columns =['tag'])
            self.stock.to_csv(self.BASIC + 'trade.csv', encoding='gb18030',index=False)
            print('tradlog')
        else:
            print('Please copy self.stock log file to input directory, and than run the application.')
            os._exit(0)

#plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
#plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置
#plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
#plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
#
# Starting an new self.stock research project.


    def industry(self):
        if os.path.exists(self.INPUT + 'cicslevel2.xls'):
            df=pd.read_excel(self.INPUT + 'cicslevel2.xls', converters={'证券代码\nSecurities Code': str})       
            df.columns = ['code','name','E-Name','Exchange','C-1','N-1','E-1','C-2','N-2','E-2','C-3','N-3','E-3','C-4','N-4','E-4','CSI300','CSI300E']
            df['symbol'] = df['Exchange'].apply(lambda x: 'sh' if x=='Shanghai' else 'sz')
            df['symbol'] = df['symbol'] + df['code']
            df['tag'] = df['symbol']
            df['tag'] = df['tag'].apply(lambda x: 'S_' + str(x))
            df = df[['code','symbol','tag','name','N-1','N-2','N-3','N-4','CSI300']]
            df.to_csv(self.BASIC + 'cicslevel2' + '.csv', encoding='gb18030', index=False)
        else:
            print('Please download industry catelog to input directory.')
            os._exit(0)  
        return

    def add(self,type=['I_sh000001']): 
        try:
            tickers = pd.read_csv(self.BASIC + 'favourite.csv', encoding='gb18030')['tag'].values.tolist()
        except :
            tickers = ['I_sh000001']
            
        tickers = tickers + type
        data = pd.DataFrame(list(set(tickers)),columns =['tag'])
        data.sort_values(by=['tag'],ascending=[True],inplace=True)
        data.to_csv(self.BASIC + 'favourite.csv', index=False)   
        return
    
    # Clean the track list.
    def reset(self):  
        tickers = ['I_sh000001']
        data = pd.DataFrame(list(set(tickers)),columns =['tag'])
        data.sort_values(by=['tag'],ascending=[True],inplace=True)
        data.to_csv(self.BASIC + 'favourite.csv', index=False)   
        return

    def download(self):
        for type in ['trade','favourite']:
            tickers = pd.read_csv(self.BASIC + type +'.csv', encoding='gb18030')['tag'].values.tolist()
    
            alltickers = tickers.copy()
            removed_tickers = []
 #           tickers = list(set(tickers).difference(set([i[:-4] for i in os.listdir(self.AKSHARE + type + '/') ])))
           
            for ticker in tickers:
                if ticker[:2]=='E_':
                    try:
                        data = ak.fund_etf_hist_sina(symbol=ticker[2:])
                        data  =  data.drop_duplicates()
                        data.to_csv(self.AKSHARE + type + '/' + ticker + '.csv', encoding='gb18030', index=False)
                    except :
                        print('Can not download ETF:',ticker)
                        removed_tickers.append(ticker)
                elif ticker[:2]=='S_':
                    try:
                        data = ak.stock_zh_a_daily(symbol=ticker[2:], adjust="qfq")
                        data.drop(['outstanding_share','turnover'],axis=1,inplace=True)
                        data  =  data.drop_duplicates()
                        data.to_csv(self.AKSHARE + type + '/' + ticker + '.csv', encoding='gb18030', index=True)
                    except :
                        print('Can not download stock:',ticker)
                        removed_tickers.append(ticker)
                elif ticker[:2]=='I_':
                    try:
                        data = ak.stock_zh_index_daily(symbol=ticker[2:])
                        data  =  data.drop_duplicates()
                        data.to_csv(self.AKSHARE + type + '/' + ticker + '.csv', encoding='gb18030', index=True)
                    except :
                        print('Can not download index:',ticker)
                        removed_tickers.append(ticker)
    
            for ticker in removed_tickers:  # Remove empty record
                try:
                    os.remove(raw_path +  ticker + '.csv')
                except:
                    pass
    
            if type == 'favourite':
                data = pd.DataFrame(list(set(alltickers).difference(set(removed_tickers))),columns =['tag'])
                data.sort_values(by=['tag'],ascending=[True],inplace=True)
                data.to_csv(self.BASIC + 'favourite.csv', encoding='gb18030',index=False)
            elif type == 'archive':
                data = pd.DataFrame(list(set(alltickers).difference(set(removed_tickers))),columns =['tag'])
                data.sort_values(by=['tag'],ascending=[True],inplace=True)
                data.to_csv(self.BASIC + 'archive.csv', encoding='gb18030',index=False)              
        return


    def mystock(self):
        print('Algorithmic trading platform versoin 1.0 by Tao Zhang in Shanghai on December 8, 2020.')
        
        clean()
        code()
        return

    def codename(self):
        codename = pd.read_csv(self.BASIC + 'code_name.csv', encoding='gb18030').set_index(['tag'])['name'].to_dict()
        
        for type in ['trade','favourite']:
            tickers = pd.read_csv(self.BASIC + type +'.csv', encoding='gb18030')['tag'].values.tolist()
        
            for ticker in tickers:
                df = pd.read_csv(self.AKSHARE + type + '/' + ticker + '.csv', encoding='gb18030',index_col='date',parse_dates=True)
                df['tag'] = ticker
                df['name'] = codename[ticker]
                df.to_csv(self.AKSHARE + type + '/' + ticker + '.csv', encoding='gb18030', index=True)
        return

    def tradelog(self):
        if os.path.exists(self.INPUT + 'stocklogfile.xls'):
            df = pd.read_excel(self.INPUT + 'stocklogfile.xls',index_col='交收日期', parse_dates=True).reset_index()
            df = df[['交收日期', '业务名称', '证券代码', '成交价格','成交数量','发生金额']]
            df.columns = ['date','type', 'tag', 'price', 'qty', 'amount']
    
            df['type'] = df['type'].apply(lambda x:
                        'buy'  if x == '证券买入'   else (
                        'sell' if x == '证券卖出'   else (
                        'sell' if x == '银行转证券' else (
                        'buy'  if x == '证券转银行' else (
                        'buy'  if x == '红利入账'   else (
                        'tax'  if x == '股息红利差异扣税' else
                        'tax' ))))))
        
            df['tag'] = df['tag'].apply(lambda x:
                    'S_' + 'sh' + str(x)[:6] if x >= 600000 else (
                    'E_' + 'sh' + str(x)[:6] if x > 510000  else (
                    'I_' + 'sz' + str(x)[:6] if x > 399000  else (
                    'E_' + 'sz' + str(x)[:6] if x > 150000  else (
                    'S_' + 'sz' + str(x)[:6] if x > 1000    else 'C_sh817909')))))
    
            df['amount'] = -1 * df['amount']
            df['qty'] = df['qty'] * df['amount'] /abs(df['amount'])
            df.loc[df['price'] == 0,'qty'] = 0
            df = df.drop(df[df.type == 'tax'].index)
            df = df.reset_index().pivot_table(index=['date','tag','type',],values=['qty','amount'],aggfunc=[np.sum])
            df.columns = ['amount','qty']
            df.to_csv(self.BASIC + 'trade_book.csv', encoding='gb18030')
    
            tickers = sorted(list(set([] + df.reset_index()['tag'].values.tolist())))
            tickers.remove('C_sh817909')
            stock = pd.DataFrame(tickers,columns =['tag'])
            stock.to_csv(self.BASIC + 'trade.csv', encoding='gb18030',index=False)
        else:
            print('Please copy stock log file to input directory, and than run the application.')
            os._exit(0)

# Add trade information for the further profit analysis.
    def tradeinfo(self):
        trade = pd.read_csv(self.BASIC + 'trade_book.csv', 
                           encoding='gb18030',
                           index_col='date', 
                           parse_dates=True).reset_index().pivot_table(
                           index=['date'], 
                           columns=["tag"],
                           values=['qty','amount'],
                           aggfunc=[np.sum])
    
        tickers = pd.read_csv(self.BASIC + 'trade.csv', encoding='gb18030')['tag'].values.tolist()
    
        for ticker in tickers:
            df = pd.read_csv(self.AKSHARE + 'trade/' + ticker + '.csv', encoding='gb18030',index_col='date',parse_dates=True)
            df['cost'] = trade['sum','amount',ticker] 
            df['qty'] = trade['sum','qty',ticker]
            df['Cost']= df['cost'].fillna(0).cumsum() 
            df['Qty'] = df['qty'].fillna(0).cumsum()    
            df['price']  = df['Cost'] / df['Qty']
            df['Amount'] = df['Qty'] * df['close']
            df['Profit'] = df['Amount'] - df['Cost'] 
            df['Rate']   = df['Profit'] / df['Cost']
            df['Color']  = df['Rate'].apply(lambda x:'g' if x <0 else 'r')    
            df.to_csv(self.AKSHARE + 'trade/' + ticker  +'.csv', encoding='gb18030', index=True)
        return

    def exposure(self):
        trades = pd.read_csv(self.BASIC + 'trade.csv', encoding='gb18030')['tag'].values.tolist()
        df = pd.concat([pd.read_csv(self.AKSHARE + 'trade/' + trade + '.csv', encoding='gb18030', parse_dates=True) for trade in trades]) 
        amount = df.reset_index().pivot_table(index=['date'], columns=['name','tag'],values=['Amount'])
        cost = df.reset_index().pivot_table(index=['date'], columns=['name','tag'],values=['Cost'])
        bank = pd.DataFrame(data=amount.sum(axis=1).values,index=amount.sum(axis=1).index,columns=['Amount'])
        bank['Exposure'] = cost.sum(axis=1)
        bank['Profit'] = bank['Amount'] - bank['Exposure']
        bank['Rate'] = bank['Profit'] / bank['Exposure']
        bank.to_csv(self.STOCK + 'trade/' + 'C_sh817909' + '.csv', encoding='gb18030', index=True)
    

    
    def signal(self):
        for type in ['trade','favourite']:
            tickers = pd.read_csv(self.BASIC + type +'.csv', encoding='gb18030')['tag'].values.tolist()
      
            for ticker in tickers:
                df = pd.read_csv(self.AKSHARE + type + '/' + ticker + '.csv', encoding='gb18030',index_col='date',parse_dates=True)
                
                ## MACD (12,26,9)
                df['DIF'] = df['close'].ewm(span=12,min_periods=1, adjust=False).mean() - df['close'].ewm(span=26,min_periods=1,adjust=False).mean()
                df['DEA'] = df['DIF'].ewm(span=9,min_periods=1,adjust=False).mean()
                df['MACD'] = (df['DIF'] - df['DEA']) * 2
                
                ## BBands (21)
                df['SMA-21D'] = df['close'].rolling(window = 21,min_periods=1).mean()
                df['UPPER'] = df['SMA-21D'] + df['close'].rolling(window = 21,min_periods=1).std() * 2
                df['LOWER'] = df['SMA-21D'] - df['close'].rolling(window = 21,min_periods=1).std() * 2
                
                ## RSI (21) 
                up = df['close'].diff().clip(lower=0).rolling(window=21,min_periods=1).mean()
                down = -1 *  df['close'].diff().clip(upper=0).rolling(window=21,min_periods=1).mean()
                rs = up/down
                df['RSI'] = 100-(100/(1+rs))
    
                ## BIAS(4,21)
                EMA_4D= df['close'].ewm(span=4,min_periods=1,adjust=False).mean()
                df['BIAS'] = (EMA_4D - df['SMA-21D'])/df['SMA-21D'] * 100
    
                ## MOM(21)
                df['MOM'] = df['close'] - df['close'].shift(axis=0,periods=21)
                df.to_csv(self.STOCK + type + '/' + ticker + '.csv', encoding='gb18030', index=True)
        return

    def signal_show(self):
        plt.rcParams['figure.figsize']=(16,18) # 设置缺省图片大小和像素
        plt.rcParams['figure.dpi'] = 100 #分辨率
        

        for type in ['trade','favourite']:
            tickers = pd.read_csv(self.BASIC + type +'.csv', encoding='gb18030')['tag'].values.tolist()
       
            for ticker in tickers:
                df = pd.read_csv(self.STOCK + type + '/' + ticker + '.csv', 
                                 encoding='gb18030',index_col='date',
                                 parse_dates=True)[dt.datetime.now() - dt.timedelta(weeks=48):dt.datetime.now() - dt.timedelta(weeks=0)]
                
                title = df.index[-1].strftime('%Y/%m/%d  ') + df.iloc[-1].loc['name'] + '[' + df.iloc[-1].loc['tag'][4:] +']'
                
                fig = plt.figure()
                ax1 = fig.add_subplot(811, ylabel='SMA')
                ax1.set_title(title, fontsize=12)
                ax1.plot(df.index,df['close'], color='b', lw=1.,label = 'Close')
                if type == 'trade':
                    ax1.plot(df.index,df['price'], color='k', lw=1.,label = 'Close',linestyle='--')
                ax1.plot(df.index,df['SMA-21D'], color='r', lw=1.,label = 'SMA-21D')
                ax1.grid(True,linestyle='--') 
                ax1.set(xlim=[df.index[0],df.index[-1] + dt.timedelta(days=2)])
                ax1.legend(loc='upper left')
    
                ax2 = fig.add_subplot(812,ylabel='Volume',sharex=ax1)     
                ax2.bar(df.index,df['volume'],color=['r' if x >0 else 'g' for x in df['close']-df['open']])        
                ax2.grid(True,linestyle='--') 
                          
                ax3 = fig.add_subplot(813, ylabel='MACD',sharex=ax1)
                ax3.plot(df.index,df['DIF'], color='r', lw=1.,label = 'DIF')
                ax3.plot(df.index,df['DEA'], color='g', lw=1.,label = 'DEA')
                ax3.bar(df['MACD'].index,df['MACD'],color=['r' if x >0 else 'g' for x in df['MACD'].values.tolist()],label = 'MACD')
                ax3.legend(loc='upper left')
                ax3.grid(True,linestyle='--') 
    
                ax4 = fig.add_subplot(814, ylabel='BIAS',sharex=ax1)
                ax4.bar(df.index,df['BIAS'],color=['r' if x > 0 else 'g' for x in df['BIAS'].values.tolist()])
                ax4.grid(True,linestyle='--')
                
                ax5 = fig.add_subplot(815, ylabel='BBANDS',sharex=ax1)
                ax5.plot(df.index,df['close'], color='b', lw=1.,label = 'Close')
                ax5.plot(df.index,df['SMA-21D'], color='k', lw=1.,label = 'SMA-21D')
                ax5.plot(df.index,df['UPPER'], color='r', lw=1.,label = 'UPPER')
                ax5.plot(df.index,df['LOWER'], color='g', lw=1.,label = 'LOWER')
                ax5.legend(loc='upper left')
                ax5.grid(True,linestyle='--') 
                
                ax6 = fig.add_subplot(816, ylabel='RSI',sharex=ax1)
                ax6.plot(df.index,df['RSI'], color='b', lw=1.,label = 'Close')
                ax6.set_ylim(0,100)
                ax6.axhline(30,lw=1, color='g', linestyle='--')
                ax6.axhline(70, lw=1,color='r', linestyle='--')
                ax6.grid(True,linestyle='--') 
                
                ax7 = fig.add_subplot(817, ylabel='Stddev in $',sharex=ax1)
                ax7.plot(df.index,df['MOM'], color='b', lw=1.,label = 'Close')
                ax7.grid(True,linestyle='--') 
                
                ax8 = fig.add_subplot(818, ylabel='MOM',sharex=ax1)
                ax8.plot(df.index,df['MOM'], color='b', lw=1.,label = 'mom')
                ax8.grid(True,linestyle='--') 
              
                plt.subplots_adjust(hspace=0.0)

                print('\n\n')

                plt.show()
                plt.close()    
            #

    def func(self,pct, allvals):
        absolute = int(pct/100.*np.sum(allvals))
        return "{:,}[{:.0f}%]".format(absolute,pct)
    
    def profit(self):
        plt.rcParams['figure.figsize']=(8,8) # 设置缺省图片大小和像素
        plt.rcParams['figure.dpi'] = 100 #分辨率
        tickers = pd.read_csv(self.BASIC + 'trade.csv', encoding='gb18030')['tag'].values.tolist()
    
        df = pd.concat([pd.read_csv(self.STOCK + 'trade/' + ticker + '.csv', encoding='gb18030', parse_dates=True)[-1:] for ticker in tickers])
        df = df[df.Qty>0][['date','name','tag', 'Cost','Qty','Amount','Profit','Rate','Color']]
        df.sort_values(by=['Rate'],ascending=[False],inplace=True)
        df.to_csv(self.STOCK + 'trade/' + 'daily_view.csv', encoding='gb18030', index=False)
        df = pd.read_csv(self.STOCK + 'trade/' + 'daily_view.csv', encoding='gb18030',index_col='date', parse_dates=True)

        
    
        fig = plt.figure()
    
        title = df.index[-1].strftime('%Y/%m/%d') + ' 股票收益一览'
        
        plt.title (title,fontsize=12)  
        
        plt.pie(df['Amount'].values.tolist(),
                   startangle = 45,
                   textprops  = {"fontsize":12},
                  labels     = (df['name'] + df['Rate'].apply(lambda x: '[{:,.0%}]'.format(x))).values.tolist() ,
                   colors     = df['Color'].values.tolist(),
                   explode    = [0.01]*len(df),
                   autopct    = lambda n: self.func(n, df['Amount'].values.tolist())            ) 
        
    
        df1 = pd.read_csv(self.STOCK + 'trade/' +  'C_sh817909.csv', encoding='gb18030', parse_dates=True)
        plt.xlabel ( '盈利' + '[{:,.0f}元]'.format(df1.iloc[-1].loc['Profit']) + ' = ' +
                     '股票市值' + '[{:,.0f}元]'.format(df1.iloc[-1].loc['Amount']) + ' - ' + 
                     '资金投入' + '[{:,.0f}元]'.format(df1.iloc[-1].loc['Exposure']) +
                 '    盈利率' + '[{:,.2%}]'.format(df1.iloc[-1].loc['Rate'])  ,  fontsize=10)  
    
       # plt.save
        plt.savefig(self.REPORT + 'Stock_' + df.index[-1].strftime('%Y%m%d') + '.jpg', dpi=200)
        plt.show()

        
        # pack_toolbar=False will make it easier to use a layout manager later on.
       # toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
       # toolbar.update()
        
        
    #    canvas.mpl_connect(
    #        "key_press_event", lambda event: print(f"you pressed {event.key}"))
    #    canvas.mpl_connect("key_press_event", key_press_handler)
        

        
        # Packing order is important. Widgets are processed sequentially and if there
        # is no space left, because the window is too small, they are not displayed.
        # The canvas is rather flexible in its size, so we pack it last which makes
        # sure the UI controls are displayed as long as possible.

       # toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)


       # plt.show()
       # plt.close()
    
        return
    ###
    
    def set(self):
        self.clean()
        self.code()
        self.tradelog()      
        self.add([])
        self.download()
        self.codename()
        self.tradeinfo()
        self.exposure()
        self.signal()
    #    self.profit()
    #    self.signal_show()
        return

    def show(self):
        self.set()
        self.profit()
 #       self.signal_show()
        return




# Run program
if __name__ == '__main__':
    s = AkStock()
    s.show()

