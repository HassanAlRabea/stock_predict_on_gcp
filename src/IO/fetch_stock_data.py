
from datetime import datetime, timedelta
from yahoo_fin import stock_info as si


def get_last_stock_price(ticker, last=False):
    if last:
        now = datetime.now()
        start_date = now - timedelta(days=500)
        return si.get_data(ticker, start_date, interval = "1d")
    return si.get_data(ticker)
