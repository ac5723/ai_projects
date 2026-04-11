import yfinance as yf
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class StockDataInput(BaseModel):
    symbol: str = Field(
        description="NSE stock symbol e.g. RELIANCE.NS, TCS.NS, INFY.NS"
    )
    period: str = Field(
        default="6mo",
        description="Data period: 1mo, 3mo, 6mo, 1y, 2y"
    )


class StockDataFetcher(BaseTool):
    name: str = "Stock Data Fetcher"
    description: str = """Fetches historical stock price data and 
    basic info for Indian NSE stocks. Input should be NSE stock 
    symbol like RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS"""
    args_schema: type[BaseModel] = StockDataInput

    def _run(self, symbol: str, period: str = "6mo") -> str:
        try:
            print(f"📊 Fetching data for {symbol}...")
            ticker = yf.Ticker(symbol)

            # fetch historical data
            df = ticker.history(period=period)

            if df.empty:
                return f"No data found for {symbol}. Check symbol name!"

            # fetch company info
            info = ticker.info

            # basic stats
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            price_change = ((current_price - prev_price) / prev_price) * 100

            high_52w = df['High'].max()
            low_52w = df['Low'].min()
            avg_volume = df['Volume'].mean()
            current_volume = df['Volume'].iloc[-1]

            result = f"""
📈 STOCK DATA: {symbol}
{'='*50}
Company: {info.get('longName', symbol)}
Sector: {info.get('sector', 'N/A')}
Industry: {info.get('industry', 'N/A')}

PRICE INFO:
Current Price: ₹{current_price:.2f}
Previous Close: ₹{prev_price:.2f}
Change: {price_change:+.2f}%

52 WEEK RANGE:
High: ₹{high_52w:.2f}
Low: ₹{low_52w:.2f}

VOLUME:
Current Volume: {current_volume:,.0f}
Average Volume: {avg_volume:,.0f}
Volume Ratio: {(current_volume/avg_volume):.2f}x

DATA PERIOD: {period}
Total Trading Days: {len(df)}
Start Date: {df.index[0].strftime('%d-%m-%Y')}
End Date: {df.index[-1].strftime('%d-%m-%Y')}
"""
            return result

        except Exception as e:
            return f"Error fetching data for {symbol}: {str(e)}"