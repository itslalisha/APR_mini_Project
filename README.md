# Stock Candle Detector

A powerful web application that automatically detects key **candlestick patterns** such as **Doji**, **Hammer**, **Engulfing**, and more — helping traders visualize potential market reversals and trends directly from stock data.

---

##  Overview

**Stock Candle Detector** fetches live or historical stock data and applies **technical candlestick pattern recognition** using `TA-Lib`.  
The app visualizes results using `Matplotlib` and provides interactive pattern analysis for better trading insights.

---

##  Features

-  Detects major candlestick patterns (Doji, Hammer, Engulfing, etc.)
-  Fetches real-time stock data using **yFinance**
-  Interactive charts and visualizations with **Matplotlib**
-  Fast and lightweight backend built with **Python**
-  Data analysis powered by **NumPy** and **Pandas**
-  User-friendly web interface for pattern exploration

---

##  Tech Stack

**Frontend:**  
- HTML / CSS / JavaScript (or Streamlit/Flask if applicable)

**Backend:**  
- Python  
- Pandas  
- NumPy  
- TA-Lib  
- yFinance  
- Matplotlib

---

##  Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/itslalisha/APR_mini_Project.git
cd stock-candle-detector
```

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

Install required dependencies:

```bash
pip install uv
```

---

##  Dependencies

Below are the major dependencies used in this project:

| Library     | Purpose |
|--------------|----------|
| **pandas** | Data manipulation and analysis |
| **numpy** | Numerical operations |
| **matplotlib** | Plotting stock candlestick charts |
| **TA-Lib** | Technical analysis and pattern detection |
| **yfinance** | Fetching live/historical stock data |
| **flask / streamlit** *(if used)* | Web framework for serving the app |

---

##  Running the App

Start the web application:

```bash
python app.py
```

or (if using Streamlit):

```bash
streamlit run app.py
```

Then open your browser and visit:

```
http://localhost:5000
```
*(or the Streamlit URL shown in your terminal)*

---

## Project Structure

```
stock-candle-detector/
│
├── app.py                # Main web app script
├── requirements.txt      # List of dependencies
├── static/               # CSS, JS, images
├── templates/            # HTML templates (if Flask)
├── utils/                # Helper functions and pattern detection scripts
└── README.md             # Project documentation
```

---

## Contributing

Contributions are welcome!  
If you’d like to improve detection accuracy, add more candlestick types, or enhance visualization:

1. Fork the repository  
2. Create a new branch (`feature/new-pattern`)  
3. Commit your changes  
4. Open a Pull Request

---

##  License

This project is licensed under the **MIT License** – you’re free to use, modify, and distribute it with attribution.

---

##  Future Improvements

- Add more advanced candlestick recognition patterns  
- Integrate AI-based trend prediction  
- Add backtesting and portfolio simulation  
- Deploy to a cloud platform (e.g., Render, Streamlit Cloud, or AWS)

---



> “Trade with logic, not emotion — let the candles tell their story.”
