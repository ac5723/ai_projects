import yfinance as yf
import pandas as pd
import numpy as np
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class IndicatorInput(BaseModel):
    symbol: str = Field(
        description="NSE stock symbol e.g. RELIANCE.NS, TCS.NS"
    )
    period: str = Field(
        default="6mo",
        description="Data period: 1mo, 3mo, 6mo, 1y"
    )


class TechnicalIndicatorsTool(BaseTool):
    name: str = "Technical Indicators Calculator"
    description: str = """Calculates technical indicators for a stock
    including RSI, MACD, Bollinger Bands, and Moving Averages.
    Use this to analyze momentum, trend and volatility."""
    args_schema: type[BaseModel] = IndicatorInput

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def _calculate_macd(self, prices: pd.Series):
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        return macd.iloc[-1], signal.iloc[-1], histogram.iloc[-1]

    def _calculate_bollinger(self, prices: pd.Series, period: int = 20):
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        return upper.iloc[-1], sma.iloc[-1], lower.iloc[-1]

    def _calculate_ema(self, prices: pd.Series, period: int) -> float:
        return prices.ewm(span=period).mean().iloc[-1]

    def _run(self, symbol: str, period: str = "6mo") -> str:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)

            if df.empty:
                return f"No data found for {symbol}"

            prices = df['Close']
            current_price = prices.iloc[-1]

            # calculate indicators
            rsi = self._calculate_rsi(prices)
            macd, signal, histogram = self._calculate_macd(prices)
            bb_upper, bb_mid, bb_lower = self._calculate_bollinger(prices)
            ema20 = self._calculate_ema(prices, 20)
            ema50 = self._calculate_ema(prices, 50)
            ema200 = self._calculate_ema(prices, 200)

            # interpret RSI
            if rsi >= 70:
                rsi_signal = "🔴 OVERBOUGHT - Potential sell signal"
            elif rsi <= 30:
                rsi_signal = "🟢 OVERSOLD - Potential buy signal"
            else:
                rsi_signal = "🟡 NEUTRAL"

            # interpret MACD
            if macd > signal and histogram > 0:
                macd_signal = "🟢 BULLISH - MACD above signal line"
            elif macd < signal and histogram < 0:
                macd_signal = "🔴 BEARISH - MACD below signal line"
            else:
                macd_signal = "🟡 NEUTRAL - Crossover zone"

            # interpret Bollinger Bands
            if current_price > bb_upper:
                bb_signal = "🔴 Price above upper band - Overbought"
            elif current_price < bb_lower:
                bb_signal = "🟢 Price below lower band - Oversold"
            else:
                bb_position = ((current_price - bb_lower) /
                               (bb_upper - bb_lower)) * 100
                bb_signal = f"🟡 Price within bands ({bb_position:.0f}% position)"

            # interpret Moving Averages
            if current_price > ema20 > ema50 > ema200:
                ma_signal = "🟢 STRONG BULLISH - Price above all EMAs"
            elif current_price < ema20 < ema50 < ema200:
                ma_signal = "🔴 STRONG BEARISH - Price below all EMAs"
            elif current_price > ema50:
                ma_signal = "🟡 MODERATELY BULLISH"
            else:
                ma_signal = "🟡 MODERATELY BEARISH"

            result = f"""
📊 TECHNICAL INDICATORS: {symbol}
{'='*50}
Current Price: ₹{current_price:.2f}

RSI (14):
Value: {rsi:.2f}
Signal: {rsi_signal}

MACD (12,26,9):
MACD Line: {macd:.4f}
Signal Line: {signal:.4f}
Histogram: {histogram:.4f}
Signal: {macd_signal}

BOLLINGER BANDS (20,2):
Upper Band: ₹{bb_upper:.2f}
Middle Band: ₹{bb_mid:.2f}
Lower Band: ₹{bb_lower:.2f}
Signal: {bb_signal}

MOVING AVERAGES:
EMA 20: ₹{ema20:.2f}
EMA 50: ₹{ema50:.2f}
EMA 200: ₹{ema200:.2f}
Signal: {ma_signal}
"""
            return result

        except Exception as e:
            return f"Error calculating indicators for {symbol}: {str(e)}"