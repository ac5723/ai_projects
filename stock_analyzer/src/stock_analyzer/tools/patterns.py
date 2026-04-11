import yfinance as yf
import pandas as pd
import numpy as np
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class PatternInput(BaseModel):
    symbol: str = Field(
        description="NSE stock symbol e.g. RELIANCE.NS"
    )
    period: str = Field(
        default="6mo",
        description="Data period: 3mo, 6mo, 1y"
    )


class ChartPatternsTool(BaseTool):
    name: str = "Chart Pattern Detector"
    description: str = """Detects chart patterns in stock price data
    including Golden Cross, Death Cross, trends, support and 
    resistance levels. Use this to identify trading patterns."""
    args_schema: type[BaseModel] = PatternInput

    def _run(self, symbol: str, period: str = "6mo") -> str:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)

            if df.empty:
                return f"No data found for {symbol}"

            prices = df['Close']
            highs = df['High']
            lows = df['Low']
            volumes = df['Volume']
            current_price = prices.iloc[-1]

            patterns_found = []
            bullish_count = 0
            bearish_count = 0

            # ─────────────────────────────
            # Golden Cross / Death Cross
            # ─────────────────────────────
            ema50 = prices.ewm(span=50).mean()
            ema200 = prices.ewm(span=200).mean()

            if (ema50.iloc[-1] > ema200.iloc[-1] and
                    ema50.iloc[-5] <= ema200.iloc[-5]):
                patterns_found.append(
                    "🟢 GOLDEN CROSS detected! (EMA50 crossed above EMA200)"
                    " - Strong bullish signal"
                )
                bullish_count += 2

            elif (ema50.iloc[-1] < ema200.iloc[-1] and
                  ema50.iloc[-5] >= ema200.iloc[-5]):
                patterns_found.append(
                    "🔴 DEATH CROSS detected! (EMA50 crossed below EMA200)"
                    " - Strong bearish signal"
                )
                bearish_count += 2

            elif ema50.iloc[-1] > ema200.iloc[-1]:
                patterns_found.append(
                    "🟢 BULLISH ALIGNMENT (EMA50 above EMA200)"
                )
                bullish_count += 1
            else:
                patterns_found.append(
                    "🔴 BEARISH ALIGNMENT (EMA50 below EMA200)"
                )
                bearish_count += 1

            # ─────────────────────────────
            # Trend Analysis
            # ─────────────────────────────
            recent_highs = highs.tail(20)
            recent_lows = lows.tail(20)

            hh = recent_highs.iloc[-1] > recent_highs.iloc[-10]
            hl = recent_lows.iloc[-1] > recent_lows.iloc[-10]
            lh = recent_highs.iloc[-1] < recent_highs.iloc[-10]
            ll = recent_lows.iloc[-1] < recent_lows.iloc[-10]

            if hh and hl:
                patterns_found.append(
                    "🟢 UPTREND - Higher Highs and Higher Lows"
                )
                bullish_count += 1
            elif lh and ll:
                patterns_found.append(
                    "🔴 DOWNTREND - Lower Highs and Lower Lows"
                )
                bearish_count += 1
            else:
                patterns_found.append(
                    "🟡 SIDEWAYS/CONSOLIDATION - No clear trend"
                )

            # ─────────────────────────────
            # Support and Resistance
            # ─────────────────────────────
            resistance = highs.tail(60).max()
            support = lows.tail(60).min()
            mid_point = (resistance + support) / 2

            dist_to_resistance = ((resistance - current_price)
                                  / current_price * 100)
            dist_to_support = ((current_price - support)
                               / current_price * 100)

            patterns_found.append(
                f"📊 SUPPORT LEVEL: ₹{support:.2f} "
                f"({dist_to_support:.1f}% below current)"
            )
            patterns_found.append(
                f"📊 RESISTANCE LEVEL: ₹{resistance:.2f} "
                f"({dist_to_resistance:.1f}% above current)"
            )

            if current_price > mid_point:
                patterns_found.append(
                    "🟢 Price in UPPER HALF of range - Bullish bias"
                )
                bullish_count += 1
            else:
                patterns_found.append(
                    "🔴 Price in LOWER HALF of range - Bearish bias"
                )
                bearish_count += 1

            # ─────────────────────────────
            # Volume Analysis
            # ─────────────────────────────
            avg_volume = volumes.mean()
            recent_volume = volumes.tail(5).mean()
            volume_ratio = recent_volume / avg_volume

            if volume_ratio > 1.5:
                patterns_found.append(
                    f"🟢 HIGH VOLUME ({volume_ratio:.1f}x average)"
                    " - Strong conviction"
                )
                bullish_count += 1
            elif volume_ratio < 0.5:
                patterns_found.append(
                    f"🔴 LOW VOLUME ({volume_ratio:.1f}x average)"
                    " - Weak conviction"
                )
                bearish_count += 1
            else:
                patterns_found.append(
                    f"🟡 NORMAL VOLUME ({volume_ratio:.1f}x average)"
                )

            # ─────────────────────────────
            # Overall Pattern Summary
            # ─────────────────────────────
            if bullish_count > bearish_count + 1:
                overall = "🟢 OVERALL: MORE BULLISH PATTERNS"
            elif bearish_count > bullish_count + 1:
                overall = "🔴 OVERALL: MORE BEARISH PATTERNS"
            else:
                overall = "🟡 OVERALL: MIXED PATTERNS"

            result = f"""
🔍 CHART PATTERNS: {symbol}
{'='*50}
Current Price: ₹{current_price:.2f}

PATTERNS DETECTED:
{chr(10).join(f'  • {p}' for p in patterns_found)}

SUMMARY:
Bullish Signals: {bullish_count}
Bearish Signals: {bearish_count}
{overall}
"""
            return result

        except Exception as e:
            return f"Error detecting patterns for {symbol}: {str(e)}"