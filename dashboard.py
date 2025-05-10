import pandas as pd
import streamlit as st
import os
import glob
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
import pyarrow as pa

# Konfiguracja strony
st.set_page_config(page_title="📈 StockBot Dashboard", layout="wide")

# Style CSS
st.markdown("""
<style>
    .stMetric {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        background-color: #f9f9f9;
    }
    .stMetric label {
        font-size: 0.9rem !important;
        color: #333 !important;
    }
    .stMetric div {
        font-size: 1.2rem !important;
        font-weight: bold !important;
        color: #000000 !important;
    }
    .holding-days {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 10px;
        border-radius: 10px;
        background-color: #e3f2fd;
    }
    .dataframe th {
        background-color: #4472C4 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Nagłówek
st.title("📈 StockBot — Zaawansowany Dashboard")
st.markdown("---")

# Cache danych
@st.cache_data(ttl=3600)
def load_data(files):
    all_data = []
    for file in files:
        try:
            temp_df = pd.read_excel(file)
            temp_df['date'] = os.path.basename(file).replace("recommendations_", "").replace(".xlsx", "")
            all_data.append(temp_df)
        except Exception as e:
            st.warning(f"Błąd przy wczytywaniu pliku {file}: {e}")
    return pd.concat(all_data) if all_data else pd.DataFrame()

# Funkcja do bezpiecznego konwertowania DataFrame
def safe_convert_df(df):
    converted_df = df.copy()
    for col in converted_df.columns:
        # Konwertuj kolumny numeryczne
        if pd.api.types.is_numeric_dtype(converted_df[col]):
            converted_df[col] = pd.to_numeric(converted_df[col], errors='coerce')
        # Konwertuj kolumny tekstowe
        else:
            converted_df[col] = converted_df[col].astype(str)
    return converted_df

# 📁 Wczytaj listę dostępnych plików Excel
folder = "recommendations"
files = sorted(glob.glob(os.path.join(folder, "recommendations_*.xlsx")), reverse=True)

if not files:
    st.warning("Brak plików z rekomendacjami.")
    st.stop()

# Funkcja formatująca nazwy plików
def format_filename(filename):
    base = os.path.basename(filename)
    date_str = base.replace("recommendations_", "").replace(".xlsx", "")
    try:
        date = datetime.strptime(date_str, "%Y%m%d_%H%M")
        return date.strftime("%d %b %Y %H:%M") + (" (Najnowsze)" if filename == files[0] else "")
    except ValueError:
        return base

# 🗂️ Selektor plików
selected_file = st.selectbox(
    "Wybierz datę rekomendacji:",
    files,
    format_func=format_filename,
    key="file_selector"
)

# Wczytaj wybrany plik
try:
    df = pd.read_excel(selected_file)
    df = safe_convert_df(df)  # Konwersja typów danych
    st.caption(f"📂 Źródło danych: `{os.path.basename(selected_file)}`")
except Exception as e:
    st.error(f"Błąd przy wczytywaniu pliku: {e}")
    st.stop()

# Emoji i kolory dla rekomendacji
emoji_map = {
    "BUY": "📈",
    "SELL": "📉",
    "HOLD": "🔄"
}

positioning_colors = {
    "LONG": "#b9f6ca",
    "SHORT": "#ffcdd2",
    "NEUTRAL": "#e0e0e0"
}

if 'recommendation' in df.columns:
    df["emoji"] = df["recommendation"].map(emoji_map).fillna("")

# Funkcja do kolorowania tabeli
def highlight(row):
    rec_colors = {
        "BUY": "#ccff90",
        "SELL": "#ffe082",
        "HOLD": "#e0e0e0"
    }
    recommendation = str(row.get("recommendation", "HOLD"))
    color = rec_colors.get(recommendation, "#ffffff")
    return [f"background-color: {color}; color: black;" for _ in row]

# 🧭 Główny układ
tab1, tab2 = st.tabs(["Aktualna rekomendacja", "Historia rekomendacji"])

with tab1:
    col1, col2 = st.columns([1, 3])

    with col1:
        tickers = df["ticker"].unique().tolist()
        current_index = st.session_state.get("current_ticker_index", 0)

        # Nawigacja między spółkami
        col_prev, col_mid, col_next = st.columns([1, 6, 1])
        with col_prev:
            if st.button("⬅️"):
                current_index = (current_index - 1) % len(tickers)
        with col_next:
            if st.button("➡️"):
                current_index = (current_index + 1) % len(tickers)

        st.session_state["current_ticker_index"] = current_index
        selected_ticker = tickers[current_index]
        selected_data = df[df["ticker"] == selected_ticker].iloc[0]

        with col_mid:
            try:
                company_name = yf.Ticker(selected_ticker).info.get('shortName', selected_ticker)
                st.markdown(f"<h3 style='text-align: center;'>📌 {selected_ticker} - {company_name}</h3>", unsafe_allow_html=True)
            except:
                st.markdown(f"<h3 style='text-align: center;'>📌 {selected_ticker}</h3>", unsafe_allow_html=True)

        # Metryki - sprawdzamy jakie kolumny są dostępne
        if 'recommendation' in selected_data and 'emoji' in df.columns:
            st.metric("💡 Rekomendacja", f"{selected_data['recommendation']} {selected_data['emoji']}")
        
        if 'price' in selected_data:
            st.metric("💰 Cena", f"${float(selected_data['price']):.2f}")
        
        cols = st.columns(2)
        with cols[0]:
            if 'rsi' in selected_data:
                st.metric("📉 RSI", f"{float(selected_data['rsi']):.1f}" if pd.notna(selected_data['rsi']) else "N/A")
            if 'macd' in selected_data:
                st.metric("📊 MACD", f"{float(selected_data['macd']):.2f}" if pd.notna(selected_data['macd']) else "N/A")
        with cols[1]:
            if 'holding_days' in selected_data:
                st.metric("📅 Holding Days", int(selected_data['holding_days']) if pd.notna(selected_data['holding_days']) else "N/A")
            if 'positioning' in selected_data:
                st.metric("📊 Pozycja", str(selected_data['positioning']))
        
        if 'score' in selected_data:
            st.metric("🏆 Siła sygnału", f"{float(selected_data['score']):.2f}" if pd.notna(selected_data['score']) else "N/A")

        # Wykres sentimentu
        if 'sentiment' in selected_data and pd.notna(selected_data['sentiment']):
            try:
                sentiment = float(selected_data['sentiment'])
                fig_sentiment = px.bar(
                    x=["Sentiment"],
                    y=[sentiment],
                    range_y=[-1, 1],
                    color=[sentiment > 0],
                    color_discrete_map={True: "#4CAF50", False: "#F44336"},
                    labels={'y': 'Wartość', 'x': ''},
                    title="📰 Analiza Sentimentu"
                )
                fig_sentiment.update_layout(showlegend=False)
                st.plotly_chart(fig_sentiment, use_container_width=True)
            except:
                pass

        # Analiza fundamentalna
        with st.expander("🔍 Analiza fundamentalna"):
            try:
                stock = yf.Ticker(selected_ticker)
                info = stock.info
                
                # Przygotuj dane w sposób zapewniający poprawne typy
                fundamental_data = pd.DataFrame({
                    "Metric": ["Kapitalizacja", "PE Ratio", "ROE", "Dywidenda", "Debt/Equity", "Sektor"],
                    "Value": [
                        f"{float(info.get('marketCap', 0))/1e9:.2f}B" if info.get('marketCap') else 'N/A',
                        str(info.get('trailingPE', 'N/A')),
                        f"{float(info.get('returnOnEquity', 0))*100:.2f}%" if info.get('returnOnEquity') else 'N/A',
                        f"{float(info.get('dividendYield', 0))*100:.2f}%" if info.get('dividendYield') else 'N/A',
                        str(info.get('debtToEquity', 'N/A')),
                        str(info.get('sector', 'N/A'))
                    ]
                })
                
                st.dataframe(fundamental_data, hide_index=True)
                
            except Exception as e:
                st.warning(f"Nie udało się pobrać danych fundamentalnych: {e}")

    with col2:
        st.subheader("📊 Zaawansowane wykresy techniczne")
        
        indicator_choice = st.selectbox(
            "Wybierz wskaźnik do wyświetlenia",
            ["Wszystkie", "RSI + MACD", "Wolumen", "Średnie kroczące"],
            key="indicator_choice"
        )
        
        try:
            stock = yf.Ticker(selected_ticker)
            hist = stock.history(period="3mo")
            
            if len(hist) < 30:
                st.warning("Za mało danych do wyświetlenia wykresu.")
            else:
                fig = plt.figure(figsize=(10, 12 if indicator_choice == "Wszystkie" else 6))
                plot_index = 1
                
                if indicator_choice in ["Wszystkie", "RSI + MACD"]:
                    # Oblicz wskaźniki na podstawie danych historycznych
                    delta = hist["Close"].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    avg_gain = gain.rolling(window=14).mean()
                    avg_loss = loss.rolling(window=14).mean()
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    ema12 = hist["Close"].ewm(span=12).mean()
                    ema26 = hist["Close"].ewm(span=26).mean()
                    macd = ema12 - ema26
                    signal = macd.ewm(span=9).mean()
                    
                    ma20 = hist["Close"].rolling(window=20).mean()
                    std20 = hist["Close"].rolling(window=20).std()
                    upper_bb = ma20 + 2 * std20
                    lower_bb = ma20 - 2 * std20
                    
                    if indicator_choice == "Wszystkie":
                        ax1 = fig.add_subplot(411)
                    else:
                        ax1 = fig.add_subplot(111)
                    
                    ax1.plot(hist.index, hist["Close"], label="Cena", color="blue")
                    ax1.plot(hist.index, upper_bb, label="Górna BB", linestyle="--", color="gray")
                    ax1.plot(hist.index, lower_bb, label="Dolna BB", linestyle="--", color="gray")
                    ax1.set_title("Cena i Bollinger Bands")
                    ax1.legend()
                    
                    if indicator_choice == "Wszystkie":
                        ax2 = fig.add_subplot(412)
                        ax2.plot(hist.index, rsi, label="RSI", color="purple")
                        ax2.axhline(70, color="red", linestyle="--", linewidth=1)
                        ax2.axhline(30, color="green", linestyle="--", linewidth=1)
                        ax2.set_title("RSI (14)")
                        
                        ax3 = fig.add_subplot(413)
                        ax3.plot(hist.index, macd, label="MACD", color="black")
                        ax3.plot(hist.index, signal, label="Sygnał", color="orange")
                        ax3.set_title("MACD")
                        ax3.legend()
                    
                    plot_index += 3
                
                if indicator_choice in ["Wszystkie", "Wolumen"]:
                    if indicator_choice == "Wszystkie":
                        ax = fig.add_subplot(414)
                    else:
                        ax = fig.add_subplot(111)
                    
                    ax.bar(hist.index, hist['Volume'], color='blue', alpha=0.3)
                    ax.set_title("Wolumen obrotu")
                    plot_index += 1
                
                if indicator_choice in ["Wszystkie", "Średnie kroczące"]:
                    if indicator_choice == "Średnie kroczące":
                        fig = plt.figure(figsize=(10, 6))
                        ax = fig.add_subplot(111)
                    
                    hist['MA50'] = hist['Close'].rolling(50).mean()
                    hist['MA200'] = hist['Close'].rolling(200).mean()
                    ax.plot(hist.index, hist['Close'], label='Cena')
                    ax.plot(hist.index, hist['MA50'], label='MA50')
                    ax.plot(hist.index, hist['MA200'], label='MA200')
                    ax.set_title("Średnie kroczące")
                    ax.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
        except Exception as e:
            st.warning(f"Błąd przy generowaniu wykresów: {e}")

    # Tabela z rekomendacjami
    with st.expander("📋 Filtrowane rekomendacje", expanded=True):
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            min_score = st.slider(
                "Minimalny score", 
                min_value=float(df['score'].min()) if 'score' in df.columns else 0.0,
                max_value=float(df['score'].max()) if 'score' in df.columns else 1.0,
                value=float(df['score'].min()) if 'score' in df.columns else 0.0,
                key="min_score"
            )
        
        with col_f2:
            if 'recommendation' in df.columns:
                selected_recommendations = st.multiselect(
                    "Filtruj rekomendacje",
                    options=df['recommendation'].unique(),
                    default=df['recommendation'].unique(),
                    key="rec_filter"
                )
        
        with col_f3:
            if 'positioning' in df.columns:
                selected_positioning = st.multiselect(
                    "Filtruj pozycje",
                    options=df['positioning'].unique(),
                    default=df['positioning'].unique(),
                    key="pos_filter"
                )
        
        # Filtrowanie danych
        filter_conditions = []
        if 'score' in df.columns:
            filter_conditions.append(df['score'] >= min_score)
        
        if 'recommendation' in df.columns and 'rec_filter' in st.session_state:
            filter_conditions.append(df['recommendation'].isin(selected_recommendations))
        
        if 'positioning' in df.columns and 'pos_filter' in st.session_state:
            filter_conditions.append(df['positioning'].isin(selected_positioning))
        
        filtered_df = df.copy()
        if filter_conditions:
            filtered_df = filtered_df[pd.concat(filter_conditions, axis=1).all(axis=1)]
        
        if 'score' in filtered_df.columns:
            filtered_df = filtered_df.sort_values(by="score", ascending=False)
        
        # Wybierz kolumny do wyświetlenia (tylko te które istnieją)
        possible_columns = {
            "emoji": "emoji" if 'emoji' in df.columns else None,
            "ticker": "ticker",
            "price": "price" if 'price' in df.columns else None,
            "rsi": "rsi" if 'rsi' in df.columns else None,
            "macd": "macd" if 'macd' in df.columns else None,
            "recommendation": "recommendation" if 'recommendation' in df.columns else None,
            "positioning": "positioning" if 'positioning' in df.columns else None,
            "holding_days": "holding_days" if 'holding_days' in df.columns else None,
            "score": "score" if 'score' in df.columns else None
        }
        
        display_cols = [col for col in possible_columns.values() if col is not None and col in filtered_df.columns]
        
        if len(display_cols) > 0:
            try:
                # Konwersja danych przed wyświetleniem
                display_df = filtered_df[display_cols].copy()
                
                # Konwersja kolumn numerycznych
                num_cols = ['price', 'rsi', 'macd', 'score', 'holding_days']
                for col in num_cols:
                    if col in display_df.columns:
                        display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                
                # Konwersja kolumn tekstowych
                text_cols = ['ticker', 'recommendation', 'positioning', 'emoji']
                for col in text_cols:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].astype(str)
                
                st.dataframe(
                    display_df.style.apply(highlight, axis=1),
                    use_container_width=True,
                    height=600
                )
            except Exception as e:
                st.error(f"Błąd przy wyświetlaniu danych: {e}")
        else:
            st.warning("Brak dostępnych kolumn do wyświetlenia")

with tab2:
    st.subheader("🕰️ Historia rekomendacji dla wybranej spółki")
    
    # Wczytaj wszystkie dane historyczne
    full_df = load_data(files)
    
    if full_df.empty:
        st.warning("Brak danych historycznych do wyświetlenia.")
    else:
        # Wybór spółki
        history_ticker = st.selectbox(
            "Wybierz spółkę do analizy historycznej",
            options=full_df['ticker'].unique(),
            index=full_df['ticker'].unique().tolist().index(selected_ticker) if selected_ticker in full_df['ticker'].unique() else 0,
            key="history_ticker"
        )
        
        history_df = full_df[full_df['ticker'] == history_ticker].sort_values('date')
        
        if not history_df.empty:
            # Przygotuj dane do wykresu - poprawione parsowanie daty
            try:
                plot_data = history_df.copy()
                plot_data['date'] = pd.to_datetime(
                    plot_data['date'],
                    format='%Y%m%d_%H%M',
                    errors='coerce'
                )
                plot_data = plot_data.dropna(subset=['date'])
                
                # Wykres zmian score i ceny
                if not plot_data.empty:
                    fig = px.line(
                        plot_data, 
                        x='date', 
                        y=['score', 'price'],
                        title=f"Historia rekomendacji dla {history_ticker}",
                        labels={'value': 'Wartość', 'variable': 'Metryka'},
                        color_discrete_map={'score': '#FFA000', 'price': '#1976D2'}
                    )
                    fig.update_layout(yaxis2=dict(title='Cena', overlaying='y', side='right'))
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Błąd przy przetwarzaniu danych historycznych: {e}")
            
            # Wykres pozycji
            if 'positioning' in history_df.columns:
                try:
                    fig_pos = px.bar(
                        plot_data,
                        x='date',
                        y='positioning',
                        color='positioning',
                        color_discrete_map=positioning_colors,
                        title="Zmiany pozycji w czasie"
                    )
                    st.plotly_chart(fig_pos, use_container_width=True)
                except:
                    pass
            
            # Tabela historyczna
            history_cols = ['date', 'recommendation', 'positioning', 'score', 'price']
            history_cols = [col for col in history_cols if col in history_df.columns]
            
            if 'rsi' in history_df.columns:
                history_cols.append('rsi')
            if 'macd' in history_df.columns:
                history_cols.append('macd')
            if 'holding_days' in history_df.columns:
                history_cols.append('holding_days')
            
            if len(history_cols) > 0:
                try:
                    st.dataframe(
                        history_df[history_cols]
                        .reset_index(drop=True)
                        .style.apply(highlight, axis=1),
                        use_container_width=True
                    )
                except:
                    st.dataframe(history_df[history_cols], use_container_width=True)
        else:
            st.warning(f"Brak danych historycznych dla spółki {history_ticker}")

st.markdown("---")
st.caption("© 2023 StockBot — Zaawansowany system rekomendacji giełdowych")