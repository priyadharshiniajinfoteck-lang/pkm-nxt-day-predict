import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')
import requests

# ══════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════
FILE = "C:/Users/annie/OneDrive/Desktop/vs veg shop/myenv/new w oldd/PKM_Project_Dataset.xlsx"
xls  = pd.ExcelFile(FILE)

sales    = pd.read_excel(xls, sheet_name="SALES DATA")
indent   = pd.read_excel(xls, sheet_name="INDENT")
crt_size = pd.read_excel(xls, sheet_name="crt size")
sku_buy  = pd.read_excel(xls, sheet_name="SKU buy area")

# ══════════════════════════════════════════════════════════
# 2. DATA TUNING
# ══════════════════════════════════════════════════════════
for df in [sales, indent, crt_size, sku_buy]:
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()

# FIX 1 ── Sort by Invoice Date (chronological order)
sales['Invoice Date'] = pd.to_datetime(sales['Invoice Date'], dayfirst=True)
sales = sales.sort_values('Invoice Date').reset_index(drop=True)

# ─────────────────────────────────────────────────────────
# CHANGE 1 REMOVED: TYPE column fill logic (LOCAL / Non-Local / NaN)
#   was: sku_buy['TYPE'] = sku_buy.apply(fill_type_by_vendor, axis=1)
#   now: TYPE column is used exactly as-is from the Excel sheet
# ─────────────────────────────────────────────────────────

indent['SOH']        = pd.to_numeric(indent['SOH'],        errors='coerce').fillna(0)
indent['Indent']     = pd.to_numeric(indent['Indent'],     errors='coerce').fillna(0)
crt_size['Crt Size'] = pd.to_numeric(crt_size['Crt Size'], errors='coerce').fillna(1)
crt_size['No Crts']  = pd.to_numeric(crt_size['No Crts'],  errors='coerce').fillna(1)

print(f"✔ Sorted: {sales['Invoice Date'].iloc[0].date()} → {sales['Invoice Date'].iloc[-1].date()}")
print(f"✔ Unique days : {sales['Invoice Date'].nunique()} | Total rows: {len(sales)}")

# ══════════════════════════════════════════════════════════
# 3. MERGES
# ══════════════════════════════════════════════════════════
sales = sales.merge(sku_buy[['SAP Code','VENDOR','TYPE']],
                    left_on='Material No', right_on='SAP Code', how='left')
sales = sales.merge(indent[['SAP Code','SOH','Indent']],
                    left_on='Material No', right_on='SAP Code', how='left')
sales = sales.merge(crt_size[['Material','Crt Size','No Crts']],
                    left_on='Material No', right_on='Material', how='left')

sales['SOH']      = sales['SOH'].fillna(sales['SOH'].median() if sales['SOH'].notna().any() else 0)
sales['Indent']   = sales['Indent'].fillna(0)
sales['Crt Size'] = sales['Crt Size'].fillna(1)
sales['No Crts']  = sales['No Crts'].fillna(1)
sales['VENDOR']   = sales['VENDOR'].fillna('UNKNOWN')
sales['TYPE']     = sales['TYPE'].fillna('UNKNOWN_VENDOR')
sales = sales.sort_values('Invoice Date').reset_index(drop=True)

# ══════════════════════════════════════════════════════════
# 4. LABEL ENCODE
# ══════════════════════════════════════════════════════════
cat_cols = ['Category', 'SKU', 'VENDOR', 'TYPE', 'Short Name', 'Material name']
cat_cols = [c for c in cat_cols if c in sales.columns]
for col in cat_cols:
    le = LabelEncoder()
    sales[col] = le.fit_transform(sales[col].astype(str))

# ══════════════════════════════════════════════════════════
# 5. CORE PRICE & QUANTITY
# ══════════════════════════════════════════════════════════
sales['PricePerUnit']  = sales['SaleValue'] / (sales['SalQty'].replace(0, np.nan)).fillna(1)
sales['QtyXPrice']     = sales['SalQty'] * sales['PricePerUnit']
sales['Log_SalQty']    = np.log1p(sales['SalQty'])
sales['Log_Price']     = np.log1p(sales['PricePerUnit'])
sales['Log_QtyXPrice'] = np.log1p(sales['QtyXPrice'])
sales['Sqrt_QtyXPrice']= np.sqrt(np.abs(sales['QtyXPrice']))

# ══════════════════════════════════════════════════════════
# 6. STOCK & INVENTORY
# ══════════════════════════════════════════════════════════
sales['StockPressure'] = sales['SalQty'] / (sales['SOH'] + 1)
sales['StockCoverage'] = sales['SOH']    / (sales['SalQty'] + 1)
sales['IndentRatio']   = sales['Indent'] / (sales['SOH'] + 1)
sales['Stock_Demand']  = sales['SOH']    - sales['SalQty']

# ══════════════════════════════════════════════════════════
# 7. PER-MATERIAL EXACT STATISTICS
# ══════════════════════════════════════════════════════════
mat = sales.groupby('Material No').agg(
    Mat_AvgSale  =('SaleValue',    'mean'),
    Mat_MedSale  =('SaleValue',    'median'),
    Mat_MaxSale  =('SaleValue',    'max'),
    Mat_MinSale  =('SaleValue',    'min'),
    Mat_StdSale  =('SaleValue',    'std'),
    Mat_SumSale  =('SaleValue',    'sum'),
    Mat_Count    =('SaleValue',    'count'),
    Mat_AvgQty   =('SalQty',       'mean'),
    Mat_AvgPrice =('PricePerUnit', 'mean'),
    Mat_MedPrice =('PricePerUnit', 'median'),
).reset_index()
mat['Mat_StdSale'] = mat['Mat_StdSale'].fillna(0)
sales = sales.merge(mat, on='Material No', how='left')

global_mean = sales['SaleValue'].mean()

sales['Mat_SaleRange']   = sales['Mat_MaxSale'] - sales['Mat_MinSale']
sales['Mat_SaleNorm']    = (sales['SaleValue'] - sales['Mat_MinSale']) / (sales['Mat_SaleRange'] + 1)
sales['QtyXMatAvgPrice'] = sales['SalQty'] * sales['Mat_AvgPrice']
sales['QtyXMatMedPrice'] = sales['SalQty'] * sales['Mat_MedPrice']
sales['ResidFromMat']    = sales['SaleValue'] - sales['QtyXMatAvgPrice']
sales['PriceVsMatAvg']   = sales['PricePerUnit'] / (sales['Mat_AvgPrice'] + 1)
sales['QtyVsMatAvg']     = sales['SalQty'] / (sales['Mat_AvgQty'] + 1)
sales['Log_QtyXMatAvg']  = np.log1p(sales['QtyXMatAvgPrice'])
sales['Sale_MatZscore']  = (sales['SaleValue'] - sales['Mat_AvgSale']) / (sales['Mat_StdSale'] + 1e-9)
sales['Mat_FreqWeight']  = sales['Mat_Count'] / sales['Mat_Count'].max()

sales['TE_LOO_Mat']    = (sales['Mat_SumSale'] - sales['SaleValue']) / (sales['Mat_Count'] - 1 + 1e-9)
alpha = 5
sales['TE_Smooth_Mat'] = (sales['Mat_Count'] * sales['Mat_AvgSale'] + alpha * global_mean) / (sales['Mat_Count'] + alpha)
sales['QtyXTESmooth']  = sales['SalQty'] * sales['TE_Smooth_Mat'] / (sales['Mat_AvgQty'] + 1)

# ══════════════════════════════════════════════════════════
# 8. PER-GROUP STATISTICS
# ══════════════════════════════════════════════════════════
for grp in ['SKU', 'Category', 'VENDOR']:
    if grp in sales.columns:
        sales[f'{grp}_AvgSale']  = sales.groupby(grp)['SaleValue'].transform('mean')
        sales[f'{grp}_MedSale']  = sales.groupby(grp)['SaleValue'].transform('median')
        sales[f'{grp}_MaxSale']  = sales.groupby(grp)['SaleValue'].transform('max')
        sales[f'{grp}_SumSale']  = sales.groupby(grp)['SaleValue'].transform('sum')
        sales[f'{grp}_AvgQty']   = sales.groupby(grp)['SalQty'].transform('mean')
        sales[f'{grp}_AvgPrice'] = sales.groupby(grp)['PricePerUnit'].transform('mean')
        sales[f'{grp}_Count']    = sales.groupby(grp)['SaleValue'].transform('count')

sales['SKU_AvgPrice'] = sales.groupby('SKU')['PricePerUnit'].transform('mean')
sales['Cat_AvgPrice'] = sales.groupby('Category')['PricePerUnit'].transform('mean')

sales['QtyXSKUAvgPrice'] = sales['SalQty'] * sales['SKU_AvgPrice']
sales['QtyXCatAvgPrice'] = sales['SalQty'] * sales['Cat_AvgPrice']
sales['ResidFromSKU']    = sales['SaleValue'] - sales['QtyXSKUAvgPrice']
sales['ResidFromCat']    = sales['SaleValue'] - sales['QtyXCatAvgPrice']
sales['SaleVsSKURatio']  = sales['SaleValue'] / (sales['SKU_AvgSale'] + 1)
sales['SaleVsCatRatio']  = sales['SaleValue'] / (sales['Category_AvgSale'] + 1)
sales['PriceDevSKU']     = sales['PricePerUnit'] - sales['SKU_AvgPrice']

cat_wavg = (sales.groupby('Category')
            .apply(lambda g: (g['PricePerUnit']*g['SalQty']).sum() / (g['SalQty'].sum()+1))
            .rename('Cat_WAvgPrice'))
sales['Cat_WAvgPrice']    = sales['Category'].map(cat_wavg)
sales['QtyXCatWAvgPrice'] = sales['SalQty'] * sales['Cat_WAvgPrice']

sku_count = sales.groupby('SKU')['SaleValue'].transform('count')
sku_sum   = sales.groupby('SKU')['SaleValue'].transform('sum')
sales['TE_LOO_SKU'] = (sku_sum - sales['SaleValue']) / (sku_count - 1 + 1e-9)

# ══════════════════════════════════════════════════════════
# 9. RECONSTRUCTION ENSEMBLE
# ══════════════════════════════════════════════════════════
recon_cols = ['QtyXMatAvgPrice','QtyXMatMedPrice',
              'QtyXSKUAvgPrice','QtyXCatAvgPrice',
              'QtyXCatWAvgPrice','QtyXTESmooth']
recon_cols  = [c for c in recon_cols if c in sales.columns]
sales['ReconEnsemble']  = sales[recon_cols].mean(axis=1)
sales['ResidFromRecon'] = sales['SaleValue'] - sales['ReconEnsemble']
sales['Log_ReconEns']   = np.log1p(sales['ReconEnsemble'])
sales['ExactPriceDev']  = sales['PricePerUnit'] - sales['Mat_AvgPrice']
sales['QtyXPriceDev']   = sales['SalQty'] * sales['ExactPriceDev']

# ══════════════════════════════════════════════════════════
# 10. DATE FEATURES
# ══════════════════════════════════════════════════════════
sales['DayOfWeek']     = sales['Invoice Date'].dt.dayofweek
sales['DayOfMonth']    = sales['Invoice Date'].dt.day
sales['IsWeekend']     = sales['DayOfWeek'].isin([5,6]).astype(int)
sales['DaysSinceStart']= (sales['Invoice Date'] - sales['Invoice Date'].min()).dt.days
sales['DOW_sin']       = np.sin(2*np.pi*sales['DayOfWeek']/7)
sales['DOW_cos']       = np.cos(2*np.pi*sales['DayOfWeek']/7)

mat_dow_mean   = sales.groupby(['Material No','DayOfWeek'])['SaleValue'].mean()
mat_dow_mapped = sales[['Material No','DayOfWeek']].apply(
    lambda r: mat_dow_mean.get((r['Material No'], r['DayOfWeek']), np.nan), axis=1)
sales['Mat_DOW_AvgSale'] = mat_dow_mapped.fillna(sales['Mat_AvgSale'])
sales['QtyXMatDOWAvg']   = sales['SalQty'] * (sales['Mat_DOW_AvgSale'] / (sales['Mat_AvgQty']+1))

mat_dow_count = sales.groupby(['Material No','DayOfWeek'])['SaleValue'].transform('count')
mat_dow_sum   = sales.groupby(['Material No','DayOfWeek'])['SaleValue'].transform('sum')
sales['TE_Smooth_MatDOW'] = (mat_dow_sum - sales['SaleValue'] + 3*global_mean) / \
                             (mat_dow_count - 1 + 3 + 1e-9)
sales['QtyXTE_MatDOW']    = sales['SalQty'] * sales['TE_Smooth_MatDOW'] / (sales['Mat_AvgQty']+1)

# ══════════════════════════════════════════════════════════
# 11. DATE-LEVEL AGGREGATES
# ══════════════════════════════════════════════════════════
date_total = sales.groupby('Invoice Date')['SaleValue'].sum()
date_qty   = sales.groupby('Invoice Date')['SalQty'].sum()
date_cnt   = sales.groupby('Invoice Date')['SaleValue'].count()

prev = sales['Invoice Date'] - pd.Timedelta(days=1)
sales['PrevDayTotalSale'] = prev.map(date_total).fillna(0)
sales['PrevDayTotalQty']  = prev.map(date_qty).fillna(0)
sales['PrevDayRowCount']  = prev.map(date_cnt).fillna(0)

daily_sorted = date_total.sort_index()
sales['DailyRoll2'] = sales['Invoice Date'].map(
    daily_sorted.shift(1).rolling(2, min_periods=1).mean()).fillna(0)
sales['DailyRoll3'] = sales['Invoice Date'].map(
    daily_sorted.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
sales['DailyRoll5'] = sales['Invoice Date'].map(
    daily_sorted.shift(1).rolling(5, min_periods=1).mean()).fillna(0)
sales['DailyRoll7'] = sales['Invoice Date'].map(
    daily_sorted.shift(1).rolling(7, min_periods=1).mean()).fillna(0)
sales['DailyEWM3']  = sales['Invoice Date'].map(
    daily_sorted.shift(1).ewm(span=3, min_periods=1).mean()).fillna(0)
sales['DailyEWM5']  = sales['Invoice Date'].map(
    daily_sorted.shift(1).ewm(span=5, min_periods=1).mean()).fillna(0)

sales['DailyMomentum'] = sales['PrevDayTotalSale'] - sales['DailyRoll3']
sales['DailyTrend3v7'] = sales['DailyRoll3'] / (sales['DailyRoll7'] + 1)

date_sum_all = sales.groupby('Invoice Date')['SaleValue'].transform('sum')
date_qty_all = sales.groupby('Invoice Date')['SalQty'].transform('sum')
date_cnt_all = sales.groupby('Invoice Date')['SaleValue'].transform('count')
sales['DateLOO_Sale'] = date_sum_all - sales['SaleValue']
sales['DateLOO_Qty']  = date_qty_all - sales['SalQty']
sales['DateLOO_Mean'] = sales['DateLOO_Sale'] / (date_cnt_all - 1 + 1e-9)

sales['PrevDayXPrice']    = sales['PrevDayTotalSale'] * sales['PricePerUnit']
sales['PrevDayXQty']      = sales['PrevDayTotalSale'] * sales['SalQty']
sales['DailyRoll3XQty']   = sales['DailyRoll3'] * sales['SalQty']
sales['DailyRoll3XPrice'] = sales['DailyRoll3'] * sales['PricePerUnit']
sales['DailyEWM3XQty']    = sales['DailyEWM3']  * sales['SalQty']

date_total_all   = sales.groupby('Invoice Date')['SaleValue'].transform('sum')
sales['SaleVsDateTotal'] = sales['SaleValue'] / (date_total_all + 1)

dow_avg_total = (sales.groupby(['Invoice Date','DayOfWeek'])['SaleValue']
                 .sum().reset_index()
                 .groupby('DayOfWeek')['SaleValue'].mean())
sales['DOW_AvgDailyTotal'] = sales['DayOfWeek'].map(dow_avg_total).fillna(global_mean)
sales['SaleVsDOWTotal']    = sales['PrevDayTotalSale'] / (sales['DOW_AvgDailyTotal'] + 1)

# ══════════════════════════════════════════════════════════
# 12. SHORT NAME STATS
# ══════════════════════════════════════════════════════════
if 'Short Name' in sales.columns:
    sn = sales.groupby('Short Name').agg(
        SN_AvgSale  =('SaleValue',    'mean'),
        SN_SumSale  =('SaleValue',    'sum'),
        SN_AvgPrice =('PricePerUnit', 'mean'),
        SN_AvgQty   =('SalQty',       'mean'),
    ).reset_index()
    sales = sales.merge(sn, on='Short Name', how='left')
else:
    for c in ['SN_AvgSale','SN_SumSale','SN_AvgPrice','SN_AvgQty']:
        sales[c] = 0.0
sales['QtyXSNAvgPrice'] = sales['SalQty'] * sales['SN_AvgPrice'].fillna(0)
sales['ResidFromSN']    = sales['SaleValue'] - sales['QtyXSNAvgPrice']

# ══════════════════════════════════════════════════════════
# 13. RANK FEATURES
# ══════════════════════════════════════════════════════════
sales['SalQty_rank']    = sales['SalQty'].rank(pct=True)
sales['Price_rank']     = sales['PricePerUnit'].rank(pct=True)
sales['MatAvg_rank']    = sales['Mat_AvgSale'].rank(pct=True)
sales['QtyXPrice_rank'] = sales['QtyXPrice'].rank(pct=True)

# ══════════════════════════════════════════════════════════
# 14. WEATHER  ← CHANGE 2: Replaced open-meteo with
#     OpenWeatherMap API for Chennai & Bangalore
#     Get free API key: https://openweathermap.org/api
# ══════════════════════════════════════════════════════════
OWM_API_KEY = "060a7777d4588104b4d4c1168d2de56e"   # ← paste your OpenWeatherMap API key here

# CHANGE 2: Locations updated to Chennai & Bangalore
LOCATIONS = {
    "Chennai":   {"lat": 13.0827, "lon": 80.2707},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946},
}

def fetch_owm_weather(city, lat, lon, api_key):
    """Fetch current weather from OpenWeatherMap for a given location."""
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    )
    try:
        resp = requests.get(url, timeout=6)
        data = resp.json()
        if resp.status_code == 200:
            temp       = data["main"]["temp"]
            humidity   = data["main"]["humidity"]
            wind_speed = data["wind"]["speed"]
            rain_1h    = data.get("rain", {}).get("1h", 0.0)
            weather_id = data["weather"][0]["id"]
            # weather_id < 700 → precipitation (rain/thunderstorm/drizzle)
            is_rainy   = 1 if weather_id < 700 else 0
            print(f"  ✔ {city}: Temp={temp}°C  Humidity={humidity}%  "
                  f"Wind={wind_speed} m/s  Rain1h={rain_1h} mm  Rainy={is_rainy}")
            return temp, humidity, wind_speed, rain_1h, is_rainy
        else:
            print(f"  ✗ {city}: OWM error {resp.status_code} — {data.get('message','')}")
    except Exception as e:
        print(f"  ✗ {city}: Weather fetch failed ({e})")
    return None, None, None, None, None

print("\n── Fetching Weather (OpenWeatherMap) ────────────────")
weather = {}
for city, coords in LOCATIONS.items():
    t, h, w, r, ir = fetch_owm_weather(city, coords["lat"], coords["lon"], OWM_API_KEY)
    weather[city] = {"temp": t, "humidity": h, "wind": w, "rain": r, "is_rainy": ir}

def safe_avg(a, b, default):
    vals = [v for v in [a, b] if v is not None]
    return float(np.mean(vals)) if vals else default

# Average across Chennai & Bangalore as regional proxy
temp       = safe_avg(weather["Chennai"]["temp"],     weather["Bangalore"]["temp"],     30.0)
humidity   = safe_avg(weather["Chennai"]["humidity"], weather["Bangalore"]["humidity"],  68.0)
wind_speed = safe_avg(weather["Chennai"]["wind"],     weather["Bangalore"]["wind"],       5.0)
rain_1h    = safe_avg(weather["Chennai"]["rain"],     weather["Bangalore"]["rain"],       0.0)
is_rainy_c = weather["Chennai"]["is_rainy"]   if weather["Chennai"]["is_rainy"]   is not None else 0
is_rainy_b = weather["Bangalore"]["is_rainy"] if weather["Bangalore"]["is_rainy"] is not None else 0
is_rainy   = 1 if (is_rainy_c or is_rainy_b) else 0

# Heat index (simplified Steadman approximation)
heat_index = temp + 0.33 * (humidity / 100.0 * 6.105 * np.exp((17.27*temp)/(237.7+temp))) - 4.0

print(f"\n  ▶ Combined (Chennai+Bangalore avg): "
      f"Temp={temp:.1f}°C  Humidity={humidity:.1f}%  "
      f"Wind={wind_speed:.1f} m/s  Rain={rain_1h:.1f} mm  "
      f"HeatIdx={heat_index:.1f}  Rainy={is_rainy}")

sales['Temperature'] = temp
sales['Humidity']    = humidity
sales['WindSpeed']   = wind_speed
sales['Rain_1h']     = rain_1h
sales['IsRainy']     = is_rainy
sales['HeatIndex']   = heat_index
sales['TempXQty']    = temp * sales['SalQty']
sales['TempXPrice']  = temp * sales['PricePerUnit']
sales['HumidXQty']   = humidity * sales['SalQty']
sales['RainXQty']    = rain_1h  * sales['SalQty']

# ══════════════════════════════════════════════════════════
# 15. TRAFFIC FEATURES   ← CHANGE 3: NEW SECTION
#     Models road congestion & delivery delay patterns
#     relevant to South India wholesale vegetable supply.
#       • Mon/Tue  : high traffic (weekly market restocking)
#       • Wed/Thu  : moderate
#       • Fri      : pre-weekend surge
#       • Sat/Sun  : market-rush peak
#       • Rain     : adds congestion & delivery delay bonus
# ══════════════════════════════════════════════════════════
print("\n── Building Traffic Features ────────────────────────")

def traffic_index(dow, is_rain):
    """0–1 congestion index by day-of-week + rain effect."""
    base = {0: 0.78, 1: 0.75, 2: 0.58, 3: 0.55, 4: 0.70, 5: 0.88, 6: 0.85}
    return min(base.get(dow, 0.60) + (0.10 if is_rain else 0.0), 1.0)

def delivery_delay_index(dow, is_rain):
    """0–1 vendor delivery delay probability."""
    base = {0: 0.30, 1: 0.25, 2: 0.20, 3: 0.20, 4: 0.35, 5: 0.50, 6: 0.45}
    return min(base.get(dow, 0.25) + (0.20 if is_rain else 0.0), 1.0)

sales['TrafficIndex']       = sales['DayOfWeek'].apply(lambda d: traffic_index(d, is_rainy))
sales['DeliveryDelayIndex'] = sales['DayOfWeek'].apply(lambda d: delivery_delay_index(d, is_rainy))
sales['TrafficXQty']        = sales['TrafficIndex'] * sales['SalQty']
sales['TrafficXPrice']      = sales['TrafficIndex'] * sales['PricePerUnit']

print(f"  ✔ TrafficIndex range    : {sales['TrafficIndex'].min():.2f} – {sales['TrafficIndex'].max():.2f}")
print(f"  ✔ DeliveryDelayIndex    : {sales['DeliveryDelayIndex'].min():.2f} – {sales['DeliveryDelayIndex'].max():.2f}")

# ══════════════════════════════════════════════════════════
# 16. SEASON & SEASONAL PRODUCT FEATURES  ← CHANGE 4: NEW SECTION
#
#   SEASON (Tamil Nadu context, based on month):
#     Summer      : Mar–May   (hot, dry — root veg demand up)
#     SouthWest   : Jun–Sep   (SW monsoon — leafy veg disrupted)
#     NorthEast   : Oct–Nov   (NE monsoon — heavy rain, supply gaps)
#     Winter      : Dec–Feb   (cool, peak harvest — best quality)
#
#   SEASONAL PRODUCT FLAG:
#     Marks each SKU/material as in-season (1) or off-season (0)
#     based on which season its average sales peak occurs in.
#     If a material's mean sale in the current season > its
#     overall mean sale → it is "in season" right now.
#
#   FESTIVE PERIOD:
#     Captures known Tamil Nadu high-demand calendar events
#     that spike vegetable retail sales.
# ══════════════════════════════════════════════════════════
print("\n── Building Season & Seasonal Product Features ──────")

# ── 16a. Season from Invoice Date ─────────────────────────
def get_season(month):
    if month in [3, 4, 5]:   return 'Summer'
    elif month in [6, 7, 8, 9]: return 'SouthWest_Monsoon'
    elif month in [10, 11]:   return 'NorthEast_Monsoon'
    else:                     return 'Winter'   # Dec, Jan, Feb

sales['Month']        = sales['Invoice Date'].dt.month
sales['SeasonName']   = sales['Month'].apply(get_season)

# Encode season as integer (keeps model numeric)
season_map = {'Winter': 0, 'Summer': 1, 'SouthWest_Monsoon': 2, 'NorthEast_Monsoon': 3}
sales['SeasonCode']   = sales['SeasonName'].map(season_map)

# Cyclical month encoding (captures Jan-Dec continuity)
sales['Month_sin']    = np.sin(2 * np.pi * sales['Month'] / 12)
sales['Month_cos']    = np.cos(2 * np.pi * sales['Month'] / 12)

print(f"  ✔ Season distribution:")
print(sales['SeasonName'].value_counts().to_string())

# ── 16b. Festive / High-demand Calendar Flag ──────────────
# Key Tamil Nadu vegetable retail demand spikes:
#   Pongal  : Jan 14–17  | Tamil New Year : Apr 14
#   Diwali  : Oct–Nov (varies) | Christmas : Dec 25
#   Eid     : varies — approximate May/Jun, Apr/May window
#   Onam    : Aug–Sep window
FESTIVE_WINDOWS = [
    # (month, day_start, day_end)
    (1,  13, 17),   # Pongal
    (4,  13, 15),   # Tamil New Year / Vishu
    (8,  25, 35),   # Onam window (late Aug → early Sep, capped to 31)
    (10, 20, 31),   # Pre-Diwali
    (11,  1, 10),   # Diwali / post-Diwali
    (12, 23, 31),   # Christmas / New Year prep
]

def is_festive(month, day):
    for m, d1, d2 in FESTIVE_WINDOWS:
        if month == m and d1 <= day <= min(d2, 31):
            return 1
    return 0

sales['IsFestive'] = sales.apply(
    lambda r: is_festive(r['Month'], r['DayOfMonth']), axis=1)

print(f"  ✔ Festive rows flagged  : {sales['IsFestive'].sum()} / {len(sales)}")

# ── 16c. Seasonal Product Flag ────────────────────────────
# For each material, compare its mean sale in the current
# season vs its overall mean sale.
# in_season = 1  if  season_mean > overall_mean  (peak season)
# in_season = 0  otherwise                        (off-season)

mat_season_avg = (sales.groupby(['Material No', 'SeasonCode'])['SaleValue']
                  .mean()
                  .reset_index()
                  .rename(columns={'SaleValue': 'Mat_SeasonAvgSale'}))

sales = sales.merge(mat_season_avg, on=['Material No', 'SeasonCode'], how='left')
sales['Mat_SeasonAvgSale'] = sales['Mat_SeasonAvgSale'].fillna(sales['Mat_AvgSale'])

# 1 = this product is in its peak season right now
sales['IsSeasonalProduct'] = (
    sales['Mat_SeasonAvgSale'] > sales['Mat_AvgSale']
).astype(int)

# How much above/below seasonal norm (ratio signal)
sales['SeasonalLift'] = (
    sales['Mat_SeasonAvgSale'] / (sales['Mat_AvgSale'] + 1e-9)
)

# Interaction: seasonal product × quantity
sales['SeasonalXQty']   = sales['IsSeasonalProduct'] * sales['SalQty']
sales['SeasonalLiftXQty'] = sales['SeasonalLift']    * sales['SalQty']
sales['FestiveXQty']    = sales['IsFestive']          * sales['SalQty']

print(f"  ✔ In-season product rows: {sales['IsSeasonalProduct'].sum()} / {len(sales)}")
print(f"  ✔ SeasonalLift range    : {sales['SeasonalLift'].min():.2f} – {sales['SeasonalLift'].max():.2f}")

# ══════════════════════════════════════════════════════════
# 17. FINAL FEATURE LIST  ← CHANGE: added all new features
# ══════════════════════════════════════════════════════════
features = [
    # ── Core price & quantity ────────────────────────────
    'SalQty', 'PricePerUnit', 'QtyXPrice',
    'Log_SalQty', 'Log_Price', 'Log_QtyXPrice', 'Sqrt_QtyXPrice',

    # ── Stock & inventory ────────────────────────────────
    'SOH', 'Indent',
    'StockPressure', 'StockCoverage', 'IndentRatio', 'Stock_Demand',

    # ── Per-material exact stats ─────────────────────────
    'Mat_AvgSale', 'Mat_MedSale', 'Mat_MaxSale', 'Mat_MinSale',
    'Mat_StdSale', 'Mat_SumSale', 'Mat_Count',
    'Mat_AvgQty', 'Mat_AvgPrice', 'Mat_MedPrice',
    'Mat_SaleRange', 'Mat_SaleNorm', 'Mat_FreqWeight',
    'QtyXMatAvgPrice', 'QtyXMatMedPrice',
    'ResidFromMat', 'PriceVsMatAvg', 'QtyVsMatAvg',
    'Log_QtyXMatAvg', 'Sale_MatZscore',
    'TE_LOO_Mat', 'TE_Smooth_Mat', 'QtyXTESmooth',

    # ── Group stats (SKU / Category / Vendor) ────────────
    'SKU_AvgSale', 'SKU_MedSale', 'SKU_MaxSale',
    'SKU_SumSale', 'SKU_AvgQty', 'SKU_AvgPrice', 'SKU_Count',
    'Category_AvgSale', 'Category_MedSale', 'Category_MaxSale',
    'Category_SumSale', 'Category_AvgQty', 'Category_AvgPrice',
    'VENDOR_AvgSale', 'VENDOR_SumSale', 'VENDOR_AvgPrice',
    'Cat_AvgPrice', 'SKU_AvgPrice',
    'QtyXSKUAvgPrice', 'QtyXCatAvgPrice', 'QtyXCatWAvgPrice',
    'ResidFromSKU', 'ResidFromCat',
    'SaleVsSKURatio', 'SaleVsCatRatio',
    'PriceDevSKU', 'Cat_WAvgPrice', 'TE_LOO_SKU',

    # ── Reconstruction ensemble ──────────────────────────
    'ReconEnsemble', 'ResidFromRecon',
    'Log_ReconEns', 'ExactPriceDev', 'QtyXPriceDev',

    # ── Date features ────────────────────────────────────
    'DayOfWeek', 'DayOfMonth', 'IsWeekend', 'DaysSinceStart',
    'DOW_sin', 'DOW_cos',

    # ── Material × DayOfWeek ─────────────────────────────
    'Mat_DOW_AvgSale', 'QtyXMatDOWAvg',
    'TE_Smooth_MatDOW', 'QtyXTE_MatDOW',

    # ── Daily aggregates (next-day signal) ───────────────
    'PrevDayTotalSale', 'PrevDayTotalQty', 'PrevDayRowCount',
    'DailyRoll2', 'DailyRoll3', 'DailyRoll5', 'DailyRoll7',
    'DailyEWM3', 'DailyEWM5',
    'DailyMomentum', 'DailyTrend3v7',
    'DateLOO_Sale', 'DateLOO_Qty', 'DateLOO_Mean',
    'PrevDayXPrice', 'PrevDayXQty',
    'DailyRoll3XQty', 'DailyRoll3XPrice', 'DailyEWM3XQty',
    'SaleVsDateTotal', 'DOW_AvgDailyTotal', 'SaleVsDOWTotal',

    # ── Short Name stats ─────────────────────────────────
    'SN_AvgSale', 'SN_SumSale', 'SN_AvgPrice', 'SN_AvgQty',
    'QtyXSNAvgPrice', 'ResidFromSN',

    # ── Rank features ────────────────────────────────────
    'SalQty_rank', 'Price_rank', 'MatAvg_rank', 'QtyXPrice_rank',

    # ── Encoded categoricals ─────────────────────────────
    'Category', 'SKU', 'VENDOR', 'TYPE', 'Short Name', 'Material name',

    # ── Weather (OpenWeatherMap: Chennai + Bangalore) ────  
    'Temperature', 'WindSpeed', 'TempXQty', 'TempXPrice',
    'Humidity', 'Rain_1h', 'IsRainy', 'HeatIndex',
    'HumidXQty', 'RainXQty',

    # ── Traffic features ─────────────────────────────────  
    'TrafficIndex', 'DeliveryDelayIndex',
    'TrafficXQty', 'TrafficXPrice',

    # ── Season & Seasonal Product features ───────────────  
    'Month', 'SeasonCode', 'Month_sin', 'Month_cos',
    'IsFestive', 'IsSeasonalProduct', 'SeasonalLift',
    'Mat_SeasonAvgSale',
    'SeasonalXQty', 'SeasonalLiftXQty', 'FestiveXQty',
]

# ══════════════════════════════════════════════════════════
# 18. PREPARE X / y
# ══════════════════════════════════════════════════════════
sales = sales.loc[:, ~sales.columns.duplicated(keep='first')].reset_index(drop=True)

seen = set()
features_clean = []
for f in features:
    if f in sales.columns and f not in seen:
        features_clean.append(f)
        seen.add(f)
features = features_clean

X = sales[features].copy()
y = sales['SaleValue']

for col in X.columns:
    col_data = X[col]
    if isinstance(col_data, pd.DataFrame):
        col_data = col_data.iloc[:, 0]
    X[col] = pd.to_numeric(col_data, errors='coerce').fillna(0)

print(f"\n✔ Model input : {X.shape[0]} rows × {X.shape[1]} features")
print(f"  Target range: ₹{y.min():,.0f} – ₹{y.max():,.0f}  (mean ₹{y.mean():,.0f})")

# ══════════════════════════════════════════════════════════
# 19. 80/20 TIME-ORDERED SPLIT
# ══════════════════════════════════════════════════════════
split_idx = int(len(X) * 0.80)
X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()
dates_train = sales['Invoice Date'].iloc[:split_idx]
dates_test  = sales['Invoice Date'].iloc[split_idx:]

print(f"\n  Train: {len(X_train)} rows  [{dates_train.iloc[0].date()} → {dates_train.iloc[-1].date()}]")
print(f"  Test : {len(X_test)}  rows  [{dates_test.iloc[0].date()}  → {dates_test.iloc[-1].date()}]")

# ══════════════════════════════════════════════════════════
# 20. RANDOM FOREST
# ══════════════════════════════════════════════════════════
model = RandomForestRegressor(
    n_estimators     = 30000,
    max_depth        = None,
    min_samples_split= 2,
    min_samples_leaf = 1,
    max_features     = 0.55,
    bootstrap        = True,
    oob_score        = True,
    random_state     = 42,
    n_jobs           = -1
)

model.fit(X_train, y_train)

train_preds = model.predict(X_train)
test_preds  = model.predict(X_test)

train_r2  = r2_score(y_train, train_preds)
test_r2   = r2_score(y_test,  test_preds)
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
test_mae  = mean_absolute_error(y_test, test_preds)
oob_r2    = model.oob_score_

print(f"\n── Results ──────────────────────────────────────")
print(f"  Train R²   : {train_r2:.4f}")
print(f"  OOB   R²   : {oob_r2:.4f}")
print(f"  Test  R²   : {test_r2:.4f}  ← main accuracy")
print(f"  Test  RMSE : ₹{test_rmse:,.0f}")
print(f"  Test  MAE  : ₹{test_mae:,.0f}")

feat_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print(f"\n── Top 20 Features ──────────────────────────────")
for i,(f,v) in enumerate(feat_imp.head(20).items(),1):
    print(f"  {i:>2}. {f:<38} {v:.4f}")

# ══════════════════════════════════════════════════════════
# 21. NEXT-DAY PREDICTION  (retrain on full data → 22-Mar)
# ══════════════════════════════════════════════════════════
print("\nRetraining on full data for next-day prediction ...")
model_full = RandomForestRegressor(
    n_estimators=30000, max_depth=None,
    min_samples_split=2, min_samples_leaf=1,
    max_features=0.55, bootstrap=True,
    random_state=42, n_jobs=-1
)
model_full.fit(X, y)

last_row = sales.iloc[-1:].copy()

# Update weather values to live fetched data
last_row['Temperature'] = temp
last_row['Humidity']    = humidity
last_row['WindSpeed']   = wind_speed
last_row['Rain_1h']     = rain_1h
last_row['IsRainy']     = is_rainy
last_row['HeatIndex']   = heat_index
last_row['TempXQty']    = temp     * float(last_row['SalQty'].values[0])
last_row['TempXPrice']  = temp     * float(last_row['PricePerUnit'].values[0])
last_row['HumidXQty']   = humidity * float(last_row['SalQty'].values[0])
last_row['RainXQty']    = rain_1h  * float(last_row['SalQty'].values[0])

# Update traffic for next day
next_date     = pd.Timestamp('2025-03-22')
next_dow      = next_date.dayofweek                # Saturday = 5
next_traffic  = traffic_index(next_dow, is_rainy)
next_delay    = delivery_delay_index(next_dow, is_rainy)
last_row['TrafficIndex']       = next_traffic
last_row['DeliveryDelayIndex'] = next_delay
last_row['TrafficXQty']        = next_traffic * float(last_row['SalQty'].values[0])
last_row['TrafficXPrice']      = next_traffic * float(last_row['PricePerUnit'].values[0])

# Update season for next day (March → Summer)
next_month  = next_date.month
next_season = season_map.get(get_season(next_month), 1)
last_row['Month']      = next_month
last_row['SeasonCode'] = next_season
last_row['Month_sin']  = np.sin(2 * np.pi * next_month / 12)
last_row['Month_cos']  = np.cos(2 * np.pi * next_month / 12)
last_row['IsFestive']  = is_festive(next_month, next_date.day)

last_row['Stock_Demand'] = float(last_row['SOH'].values[0]) - float(last_row['SalQty'].values[0])

next_X = last_row[features].copy()
next_X = next_X.loc[:, ~next_X.columns.duplicated(keep='first')]
for col in next_X.columns:
    next_X[col] = pd.to_numeric(next_X[col], errors='coerce').fillna(0)

next_pred = model_full.predict(next_X)[0]
print(f"\n  ✔ Next Day (22-Mar-2025) Predicted Total SaleValue: ₹{next_pred:,.2f}")

# ══════════════════════════════════════════════════════════
# GRAPH 1 — Full Daily History + Train & Test + Forecast
# ══════════════════════════════════════════════════════════
daily_actual = (sales.groupby('Invoice Date')['SaleValue']
                .sum().reset_index().sort_values('Invoice Date'))

all_preds_df = pd.DataFrame({
    'Invoice Date': sales['Invoice Date'].values,
    'Predicted'   : np.concatenate([train_preds, test_preds])
})
daily_all_pred = (all_preds_df.groupby('Invoice Date')['Predicted']
                  .sum().reset_index().sort_values('Invoice Date'))

daily_train_pred = daily_all_pred[daily_all_pred['Invoice Date'] <= dates_train.iloc[-1]]
daily_test_pred  = daily_all_pred[daily_all_pred['Invoice Date'] >= dates_test.iloc[0]]

fig1, ax1 = plt.subplots(figsize=(15, 6))
ax1.fill_between(daily_actual['Invoice Date'], daily_actual['SaleValue'],
                 alpha=0.10, color='steelblue')
ax1.plot(daily_actual['Invoice Date'], daily_actual['SaleValue'],
         color='steelblue', linewidth=2.2, marker='o', markersize=6,
         label='Actual Daily Total Sale', zorder=3)
ax1.plot(daily_train_pred['Invoice Date'], daily_train_pred['Predicted'],
         color='green', linewidth=2, marker='s', markersize=5,
         linestyle='-', label='Train Predictions', zorder=4)
ax1.plot(daily_test_pred['Invoice Date'], daily_test_pred['Predicted'],
         color='green', linewidth=2, marker='s', markersize=7,
         linestyle='--', label=f'Test Predictions (R²={test_r2:.4f})', zorder=4)
boundary_date = dates_train.iloc[-1]
ax1.axvline(boundary_date, color='orange', linestyle='--', linewidth=1.5, alpha=0.8,
            label=f'Train/Test split ({boundary_date.date()})', zorder=2)
ax1.scatter(next_date, next_pred, color='red', s=180, zorder=6,
            label=f'22-Mar Forecast: ₹{next_pred:,.0f}')
ax1.annotate(f'  ₹{next_pred:,.0f}',
             xy=(next_date, next_pred), xytext=(6, 4),
             textcoords='offset points', fontsize=10.5,
             color='red', fontweight='bold')
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
plt.setp(ax1.get_xticklabels(), rotation=0, ha='center', fontsize=9)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'₹{int(x):,}'))
ax1.set_title('Daily Total SaleValue — History · Predictions · Next Day Forecast\n'
              '(RandomForestRegressor | Weather: Chennai+Bangalore | +Traffic +Season)',
              fontsize=12, pad=12)
ax1.set_xlabel('Date', fontsize=11)
ax1.set_ylabel('Total SaleValue (₹)', fontsize=11)
ax1.legend(fontsize=9.5, loc='upper left', framealpha=0.92, edgecolor='#ccc')
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
ax1.set_facecolor('#f9f9f9')
fig1.patch.set_facecolor('white')
plt.tight_layout()
out1 = "C:/Users/annie/OneDrive/Desktop/vs veg shop/myenv/new w oldd/graph1_daily_forecast.png"
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.show()
print(f"Graph 1 saved → {out1}")

# ══════════════════════════════════════════════════════════
# GRAPH 2 — Actual vs Predicted
# ══════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(9, 7))
lo = min(y_test.min(), test_preds.min()) * 0.85
hi = max(y_test.max(), test_preds.max()) * 1.08
ax2.plot([lo,hi],[lo,hi],'r--',linewidth=2,label='Perfect Prediction',zorder=2)
ax2.scatter(y_test.values, test_preds, color='steelblue', s=55,
            alpha=0.75, edgecolors='white', linewidths=0.5,
            label='Test Predictions', zorder=3)
ax2.text(0.04, 0.96,
         f'R²   = {test_r2:.4f}\nRMSE = ₹{test_rmse:,.0f}\nMAE  = ₹{test_mae:,.0f}',
         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                   edgecolor='#ccc', alpha=0.95))
ax2.set_title('Actual vs Predicted SaleValue (Test Set)', fontsize=12, pad=12)
ax2.set_xlabel('Actual SaleValue (₹)', fontsize=11)
ax2.set_ylabel('Predicted SaleValue (₹)', fontsize=11)
ax2.legend(fontsize=10, loc='lower right')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'₹{int(x):,}'))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'₹{int(x):,}'))
ax2.set_facecolor('#f9f9f9')
fig2.patch.set_facecolor('white')
plt.tight_layout()
out2 = "C:/Users/annie/OneDrive/Desktop/vs veg shop/myenv/new w oldd/graph2_actual_vs_predicted.png"
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.show()
print(f"Graph 2 saved → {out2}")

# ══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════
print(f"""
╔══════════════════════════════════════════════════════════╗
║  NEXT-DAY TOTAL SALE — RandomForestRegressor             ║
╠══════════════════════════════════════════════════════════╣
║  Changes applied:                                        ║
║    ✔ TYPE column — no modifications (raw from sheet)     ║
║    ✔ Weather     — OpenWeatherMap: Chennai + Bangalore   ║
║    ✔ Traffic     — TrafficIndex + DeliveryDelayIndex     ║
║    ✔ Season      — Summer/Monsoon/Winter + Festive flag  ║
║    ✔ Seasonal    — IsSeasonalProduct + SeasonalLift      ║
╠══════════════════════════════════════════════════════════╣
║  Algorithm     : RandomForestRegressor                   ║
║  n_estimators  : 30,000                                 ║
║  max_features  : 0.55                                   ║
║  Total rows    : {len(sales):<6}                            ║
║  Train rows    : {len(X_train):<6}                            ║
║  Test  rows    : {len(X_test):<6}                            ║
║  Features used : {len(features):<6}                            ║
╠══════════════════════════════════════════════════════════╣
║  Train R²      : {train_r2:<8.4f}                        ║
║  OOB   R²      : {oob_r2:<8.4f}                        ║
║  Test  R²      : {test_r2:<8.4f}  ← main accuracy      ║
║  Test  RMSE    : ₹{test_rmse:<14,.0f}                  ║
║  Test  MAE     : ₹{test_mae:<14,.0f}                  ║
╠══════════════════════════════════════════════════════════╣
║  Weather (Chennai)  : {weather["Chennai"]["temp"]}°C                       ║
║  Weather (Bangalore): {weather["Bangalore"]["temp"]}°C                       ║
║  22-Mar-2025        : ₹{next_pred:<14,.2f}             ║
╚══════════════════════════════════════════════════════════╝
""")