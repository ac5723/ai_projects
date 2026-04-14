import os
import sys
import json
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# add src to path so we can import stock_analyzer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from stock_analyzer.crew import StockAnalyzer

app = Flask(__name__)
CORS(app)  # allow frontend to call backend

# NSE Stocks organized by sector
STOCKS = {
    "Information Technology": [
        {"symbol": "TCS.NS", "name": "Tata Consultancy Services"},
        {"symbol": "INFY.NS", "name": "Infosys"},
        {"symbol": "WIPRO.NS", "name": "Wipro"},
        {"symbol": "HCLTECH.NS", "name": "HCL Technologies"},
        {"symbol": "TECHM.NS", "name": "Tech Mahindra"},
        {"symbol": "LTIM.NS", "name": "LTIMindtree"},
    ],
    "Banking & Finance": [
        {"symbol": "HDFCBANK.NS", "name": "HDFC Bank"},
        {"symbol": "ICICIBANK.NS", "name": "ICICI Bank"},
        {"symbol": "SBIN.NS", "name": "State Bank of India"},
        {"symbol": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank"},
        {"symbol": "AXISBANK.NS", "name": "Axis Bank"},
        {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance"},
    ],
    "Energy & Oil": [
        {"symbol": "RELIANCE.NS", "name": "Reliance Industries"},
        {"symbol": "ONGC.NS", "name": "ONGC"},
        {"symbol": "POWERGRID.NS", "name": "Power Grid"},
        {"symbol": "NTPC.NS", "name": "NTPC"},
        {"symbol": "BPCL.NS", "name": "BPCL"},
    ],
    "Automobile": [
        {"symbol": "TATAMOTORS.NS", "name": "Tata Motors"},
        {"symbol": "MARUTI.NS", "name": "Maruti Suzuki"},
        {"symbol": "BAJAJ-AUTO.NS", "name": "Bajaj Auto"},
        {"symbol": "HEROMOTOCO.NS", "name": "Hero MotoCorp"},
        {"symbol": "EICHERMOT.NS", "name": "Eicher Motors"},
    ],
    "Pharma & Healthcare": [
        {"symbol": "SUNPHARMA.NS", "name": "Sun Pharmaceutical"},
        {"symbol": "DRREDDY.NS", "name": "Dr Reddy's Labs"},
        {"symbol": "CIPLA.NS", "name": "Cipla"},
        {"symbol": "DIVISLAB.NS", "name": "Divi's Laboratories"},
        {"symbol": "APOLLOHOSP.NS", "name": "Apollo Hospitals"},
    ],
    "FMCG": [
        {"symbol": "HINDUNILVR.NS", "name": "Hindustan Unilever"},
        {"symbol": "ITC.NS", "name": "ITC"},
        {"symbol": "NESTLEIND.NS", "name": "Nestle India"},
        {"symbol": "BRITANNIA.NS", "name": "Britannia Industries"},
        {"symbol": "DABUR.NS", "name": "Dabur India"},
    ],
    "Metals & Mining": [
        {"symbol": "TATASTEEL.NS", "name": "Tata Steel"},
        {"symbol": "JSWSTEEL.NS", "name": "JSW Steel"},
        {"symbol": "HINDALCO.NS", "name": "Hindalco Industries"},
        {"symbol": "COALINDIA.NS", "name": "Coal India"},
        {"symbol": "VEDL.NS", "name": "Vedanta"},
    ],
    "Infrastructure": [
        {"symbol": "ADANIENT.NS", "name": "Adani Enterprises"},
        {"symbol": "ADANIPORTS.NS", "name": "Adani Ports"},
        {"symbol": "DLF.NS", "name": "DLF"},
        {"symbol": "LT.NS", "name": "Larsen & Toubro"},
        {"symbol": "ULTRACEMCO.NS", "name": "UltraTech Cement"},
    ],
}


@app.route('/api/sectors', methods=['GET'])
def get_sectors():
    """Return all sectors"""
    return jsonify(list(STOCKS.keys()))


@app.route('/api/stocks/<sector>', methods=['GET'])
def get_stocks(sector):
    """Return stocks for a sector"""
    stocks = STOCKS.get(sector, [])
    return jsonify(stocks)


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Run stock analysis"""
    try:
        data = request.json
        symbol = data.get('symbol')
        stock_name = data.get('name')

        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400

        print(f"\n🚀 Starting analysis for {symbol}...")

        # run crew analysis
        os.makedirs("output", exist_ok=True)
        inputs = {"symbol": symbol}
        StockAnalyzer().crew().kickoff(inputs=inputs)

        # read generated report
        report_path = os.path.join(
            os.path.dirname(__file__), '..', 'output', 'signal_report.md'
        )

        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                report = f.read()
        else:
            report = "Report generation failed. Please try again."

        return jsonify({
            'success': True,
            'symbol': symbol,
            'name': stock_name,
            'report': report
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("🚀 Starting Stock Analyzer UI...")
    print("📊 Open http://localhost:5000 in your browser")
    app.run(debug=False, port=5000)