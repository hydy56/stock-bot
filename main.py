# main.py - Zintegrowany bot inwestycyjny z pełnym zapisem analiz i szacowaniem czasu trzymania pozycji

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib
import requests
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import linregress
import talib

CONFIG = {
    'news_api_keys': ['437d13a52a1a4cde982ef2fcb7d448f3'],
    'analysis_params': {
        'min_market_cap': 10e9
    },
    'reco_dir': 'recommendations',
    'ml_model_path': 'model_rf.pkl',
    'model_retrain_hours': 12
}

class StockBot:

    def load_watchlist(self, path='watchlist.txt'):
        try:
            with open(path, 'r') as f:
                return set(line.strip().upper() for line in f if line.strip())
        except FileNotFoundError:
            return set()

    def get_top_100_tickers(self):
        try:
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            tickers = table['Symbol'].tolist()[:100]
            return [t.replace('.', '-') for t in tickers]
        except Exception as e:
            logging.error(f"Błąd pobierania top 100 companies: {e}")
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'JNJ', 'V']

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.model = None
        self.last_model_train_time = None
        self.watchlist = self.load_watchlist()
        os.makedirs(CONFIG['reco_dir'], exist_ok=True)
        logging.basicConfig(level=logging.INFO)
        self.load_or_train_model()

    def load_or_train_model(self):
        try:
            if os.path.exists(CONFIG['ml_model_path']):
                self.model = joblib.load(CONFIG['ml_model_path'])
                logging.info("Existing ML model loaded.")
                self.last_model_train_time = datetime.now()
            else:
                self.train_model()
        except Exception as e:
            logging.error(f"Error loading model, training a new one: {e}")
            self.train_model()

    def train_model(self):
        start_time = datetime.now()
        training_msg = f"Started training ML model... Czas rozpoczęcia: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        print(training_msg)
        logging.info(training_msg)
        
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'] + list(self.watchlist)
        tickers = list(set(tickers))
        
        X, y = [], []
        processed_tickers = []
        
        for ticker in tickers:
            try:
                yfinance_ticker = ticker.replace('.', '-')
                data = yf.Ticker(yfinance_ticker).history(period='6mo')
                if data.empty or len(data) < 40:
                    continue

                data['RSI'] = data['Close'].pct_change().rolling(14).mean()
                data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
                data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
                bb_upper = data['Close'].rolling(20).mean() + 2 * data['Close'].rolling(20).std()
                bb_lower = data['Close'].rolling(20).mean() - 2 * data['Close'].rolling(20).std()
                data['BB_Pos'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower)
                data['Target'] = (data['Close'].shift(-5) > data['Close']).astype(int)
                data = data.dropna()

                for _, row in data.iterrows():
                    X.append([row['RSI'], row['MACD'], row['MACD_Signal'], row['BB_Pos']])
                    y.append(row['Target'])
                
                processed_tickers.append(ticker)
            except Exception as e:
                logging.error(f"Error processing data for {ticker}: {e}")
                continue

        ticker_info = f"Processed data for {len(processed_tickers)} companies: {', '.join(processed_tickers[:5])}{'...' if len(processed_tickers) > 5 else ''}"
        print(ticker_info)
        logging.info(ticker_info)

        if len(X) < 100:
            warning_msg = f"Not enough data to train model (tylko {len(X)} próbek), używam domyślnego"
            print(warning_msg)
            logging.warning(warning_msg)
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            X = np.random.rand(100, 4)
            y = np.random.randint(0, 2, 100)
            self.model.fit(X, y)
        else:
            X, y = np.array(X), np.array(y)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            joblib.dump(model, CONFIG['ml_model_path'])
            self.model = model

        self.last_model_train_time = datetime.now()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        completion_msg = f" Finished training ML model! Duration: {duration:.2f} sekund. Model saved and ready."
        print(completion_msg)
        logging.info(completion_msg)

    async def send_message(self, text):
        pass

    async def get_sentiment_score(self, ticker):
        try:
            key = CONFIG['news_api_keys'][0]
            url = f'https://newsapi.org/v2/everything?q={ticker}&language=en&apiKey={key}&pageSize=10'
            articles = requests.get(url).json().get('articles', [])
            if not articles:
                return 0

            scores = []
            for a in articles:
                title = a.get('title', '') or ''
                description = a.get('description', '') or ''
                text = f"{title}. {description}".strip()
                if text:
                    scores.append(self.analyzer.polarity_scores(text)['compound'])

            return np.mean(scores) if scores else 0
        except Exception as e:
            logging.error(f"Sentiment error for {ticker}: {e}")
            return 0

    async def estimate_holding_period(self, ticker, data):
        """Szacuje optymalny czas trzymania pozycji w dniach"""
        try:
            # Konwersja na numpy array dla lepszej wydajności
            highs = data['High'].values
            lows = data['Low'].values
            closes = data['Close'].values
            
            # 1. Analiza trendu (regresja liniowa)
            x = np.arange(len(closes))
            slope, _, _, _, _ = linregress(x, closes)
            trend_strength = abs(slope) * 100  # Siła trendu w %
            
            # 2. Wskaźnik ATR (Average True Range) - poprawione użycie iloc
            atr = talib.ATR(highs, lows, closes, timeperiod=14)
            atr_value = atr[-1] if len(atr) > 0 else 0
            volatility_ratio = atr_value / closes[-1] if closes[-1] != 0 else 0
            
            # 3. Analiza średnich kroczących
            sma_50 = talib.SMA(closes, timeperiod=50)[-1] if len(closes) >= 50 else closes[-1]
            sma_200 = talib.SMA(closes, timeperiod=200)[-1] if len(closes) >= 200 else closes[-1]
            ma_ratio = sma_50 / sma_200 if sma_200 != 0 else 1
            
            # 4. Ocena fundamentów
            info = yf.Ticker(ticker).info
            pe = info.get('trailingPE', 30)
            peg = info.get('pegRatio', 2)
            
            # Algorytm szacowania czasu
            if trend_strength > 1.5 and volatility_ratio < 0.03:
                base_days = 90  # Silny trend, niska zmienność
            elif ma_ratio > 1 and pe < 25 and peg < 1.5:
                base_days = 30  # Byczy rynek, dobre wyceny
            else:
                base_days = 14  # Wysoka zmienność/słaby trend
                
            # Korekta na podstawie RSI
            rsi = talib.RSI(closes)[-1] if len(closes) >= 14 else 50
            if rsi > 70:
                base_days *= 0.7  # Skróć przy przewartościowaniu
            elif rsi < 30:
                base_days *= 1.3  # Wydłuż przy niedowartościowaniu
                
            return max(7, min(180, int(base_days)))  # Ogranicz do 7-180 dni
            
        except Exception as e:
            logging.error(f"Error holding period dla {ticker}: {e}")
            return 30  # Domyślna wartość

    async def analyze_stock(self, ticker):
        try:
            yfinance_ticker = ticker.replace('.', '-')
            stock = yf.Ticker(yfinance_ticker)
            data = stock.history(period='2mo')
            if data.empty or len(data) < 30:
                logging.warning(f"No data for {ticker} (konwertowano na {yfinance_ticker})")
                return None

            info = stock.info
            market_cap = info.get('marketCap', 0)
            if market_cap < CONFIG['analysis_params']['min_market_cap']:
                return None

            data['RSI'] = data['Close'].pct_change().rolling(14).mean()
            data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            bb_upper = data['Close'].rolling(20).mean() + 2 * data['Close'].rolling(20).std()
            bb_lower = data['Close'].rolling(20).mean() - 2 * data['Close'].rolling(20).std()

            last = data.iloc[-1]
            rsi = last['RSI']
            macd = last['MACD']
            macd_signal = last['MACD_Signal']
            bb_pos = (last['Close'] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

            sentiment = await self.get_sentiment_score(ticker)

            pe = info.get('trailingPE', None)
            roe = info.get('returnOnEquity', None)
            de = info.get('debtToEquity', None)
            growth = info.get('earningsQuarterlyGrowth', None)

            score = 0
            if rsi < 30: score += 0.1
            if macd > macd_signal: score += 0.1
            if bb_pos < 0.25: score += 0.05
            if sentiment > 0.2: score += 0.1
            if pe and 5 < pe < 25: score += 0.05
            if roe and roe > 0.15: score += 0.05
            if de and de < 1: score += 0.05
            if growth and growth > 0: score += 0.05

            features = np.array([[rsi, macd, macd_signal, bb_pos]])
            prob_up = self.model.predict_proba(features)[0][1]
            score += (prob_up - 0.5) * 0.2

            recommendation = 'HOLD'
            if score > 0.4: recommendation = 'BUY'
            elif score < -0.4: recommendation = 'SELL'

            # Szacowanie czasu trzymania pozycji
            holding_days = await self.estimate_holding_period(ticker, data)

            return {
                'ticker': ticker,
                'price': last['Close'],
                'sentiment': sentiment,
                'pe': pe,
                'roe': roe,
                'de': de,
                'growth': growth,
                'prob_up': prob_up,
                'score': score,
                'recommendation': recommendation,
                'positioning': 'LONG' if recommendation == 'BUY' else 'SHORT' if recommendation == 'SELL' else 'NEUTRAL',
                'holding_days': holding_days,
                'watchlist': ticker in self.watchlist
            }
        except Exception as e:
            logging.error(f"Error analyzing {ticker}: {e}")
            return None

    async def run_once(self):
        tickers = list(set(self.get_top_100_tickers() + list(self.watchlist)))
        
        if (self.last_model_train_time is None or 
            (datetime.now() - self.last_model_train_time) > timedelta(hours=CONFIG['model_retrain_hours'])):
            logging.info("Time to retrain the model...")
            self.train_model()

        async def analyze(t):
            return await self.analyze_stock(t)

        results_raw = await asyncio.gather(*(analyze(t) for t in tickers))
        
        all_results = []
        for r in results_raw:
            if r:
                r['strong_recommendation'] = abs(r['score']) > 0.3
                all_results.append(r)
        
        df = pd.DataFrame(all_results)
        df['analysis_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df['watchlist'] = df['watchlist'].apply(lambda x: '⭐' if x else '')
        df['strong_recommendation'] = df['strong_recommendation'].apply(lambda x: 'STRONG' if x else 'weak')
        
        # Nowe sortowanie - najpierw LONG/SHORT, potem NEUTRAL, w obrębie każdej grupy sortowane po score
        df['positioning_priority'] = df['positioning'].apply(
            lambda x: 0 if x in ['LONG', 'SHORT'] else 1)
        
        df = df.sort_values(
            by=['positioning_priority', 'strong_recommendation', 'score'], 
            ascending=[True, False, False]
        ).drop(columns=['positioning_priority'])
        
        filename = os.path.join(CONFIG['reco_dir'], f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx")
        
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Analiza')
            
            workbook = writer.book
            worksheet = writer.sheets['Analiza']
            
            # Formatowanie nagłówków
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#4472C4',
                'font_color': 'white',
                'border': 1
            })
            
            # Formatowanie dla LONG (zielone tło)
            long_format = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
            # Formatowanie dla SHORT (czerwone tło)
            short_format = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
            # Formatowanie dla strong recommendations (złote tło)
            strong_format = workbook.add_format({'bg_color': '#FFD700'})
            
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            for row_num in range(1, len(df)+1):
                positioning = df.iloc[row_num-1]['positioning']
                is_strong = df.iloc[row_num-1]['strong_recommendation'] == 'STRONG'
                
                if positioning == 'LONG':
                    worksheet.set_row(row_num, None, long_format)
                elif positioning == 'SHORT':
                    worksheet.set_row(row_num, None, short_format)
                elif is_strong:
                    worksheet.set_row(row_num, None, strong_format)
            
            for i, col in enumerate(df.columns):
                max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(i, i, max_len)
            
            # Dodajemy wykresy/spisy treści
            worksheet.freeze_panes(1, 0)  # Zablokuj nagłówki
        
        logging.info(f"Full analysis saved {len(df)} companies do: {filename}")
        
        strong_count = len(df[df['strong_recommendation'] == 'STRONG'])
        long_count = len(df[df['positioning'] == 'LONG'])
        short_count = len(df[df['positioning'] == 'SHORT'])
        
        print(f"\nAnalysis complete. Analyzed {len(df)} companies.")
        print(f"Found: {long_count} LONG, {short_count} SHORT, {strong_count} strong recommendations")
        print(f"Full results saved to: {filename}")

        return [r for r in all_results if r['strong_recommendation']]

if __name__ == '__main__':
    bot = StockBot()

    async def main_loop():
        print('Bot started and is running in the background!')
        while True:
            try:
                await bot.run_once()
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                print(f' An error occurred: {str(e)}')
            await asyncio.sleep(60 * 60)
    asyncio.run(main_loop())