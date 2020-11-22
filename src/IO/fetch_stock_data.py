
from datetime import datetime, timedelta
from yahoo_fin import stock_info as si


def get_last_stock_price(ticker, last=False):
    if last:
        now = datetime.now()
        start_date = now - timedelta(days=2190)
        return si.get_data(ticker, start_date, interval = "1d")
    return si.get_data(ticker)

def get_sp500_tickers():
    # Obtain the SP 500 Tickers 
    sp_500 = si.tickers_sp500()
    #Removing problematic tickers
    sp_500.remove("BF.B")
    sp_500.remove("BRK.B")
    sp_500.remove("VNT")

    return sp_500
