import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
import requests

# ══════════════════════════════════════════════════════════
# All helper functions from new.py
# ══════════════════════════════════════════════════════════
OWM_API_KEY = "060a7777d4588104b4d4c1168d2de56e"

LOCATIONS = {
    "Chennai":   {"lat": 13.0827, "lon": 80.2707},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946},
}

def fetch_owm_weather(city, lat, lon, api_key):
    url = (f"https://api.openweathermap.org/data/2.5/weather"
           f"?lat={lat}&lon={lon}&appid={api_key}&units=metric")
    try:
        resp = requests.get(url, timeout=6)
        data = resp.json()
        if resp.status_code == 200:
            return (data["main"]["temp"], data["main"]["humidity"],
                    data["wind"]["speed"], data.get("rain",{}).get("1h",0.0),
                    1 if data["weather"][0]["id"] < 700 else 0)
    except Exception as e:
        print(f"  ✗ {city}: {e}")
    return None, None, None, None, None

def safe_avg(a, b, default):
    vals = [v for v in [a, b] if v is not None]
    return float(np.mean(vals)) if vals else default

def get_season(month):
    if month in [3, 4, 5]:       return 'Summer'
    elif month in [6, 7, 8, 9]:  return 'SouthWest_Monsoon'
    elif month in [10, 11]:      return 'NorthEast_Monsoon'
    else:                        return 'Winter'

season_map = {'Winter': 0, 'Summer': 1, 'SouthWest_Monsoon': 2, 'NorthEast_Monsoon': 3}

FESTIVE_WINDOWS = [
    (1, 13, 17), (4, 13, 15), (8, 25, 35),
    (10, 20, 31), (11, 1, 10), (12, 23, 31),
]

def is_festive(month, day):
    for m, d1, d2 in FESTIVE_WINDOWS:
        if month == m and d1 <= day <= min(d2, 31):
            return 1
    return 0

def traffic_index(dow, is_rain):
    base = {0: 0.78, 1: 0.75, 2: 0.58, 3: 0.55, 4: 0.70, 5: 0.88, 6: 0.85}
    return min(base.get(dow, 0.60) + (0.10 if is_rain else 0.0), 1.0)

def delivery_delay_index(dow, is_rain):
    base = {0: 0.30, 1: 0.25, 2: 0.20, 3: 0.20, 4: 0.35, 5: 0.50, 6: 0.45}
    return min(base.get(dow, 0.25) + (0.20 if is_rain else 0.0), 1.0)


# ══════════════════════════════════════════════════════════
# run_pipeline() — new.py logic wrapped as a function
# Only difference from new.py:
#   1. FILE path comes from argument instead of hardcoded
#   2. next_date comes from argument instead of hardcoded
#   3. plt.show() / savefig() removed (Streamlit handles charts)
#   4. Per-product predictions added at the end
#   5. Returns a dict instead of just printing
# Everything else is IDENTICAL to new.py
# ══════════════════════════════════════════════════════════
def run_pipeline(file_path: str, next_date_str: str = "2025-03-22"):

    next_date = pd.Timestamp(next_date_str)

    # ── 1. LOAD ───────────────────────────────────────────
    xls      = pd.ExcelFile(file_path)
    sales    = pd.read_excel(xls, sheet_name="SALES DATA")
    indent   = pd.read_excel(xls, sheet_name="INDENT")
    crt_size = pd.read_excel(xls, sheet_name="crt size")
    sku_buy  = pd.read_excel(xls, sheet_name="SKU buy area")

    # ── 2. DATA TUNING ────────────────────────────────────
    for df in [sales, indent, crt_size, sku_buy]:
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype(str).str.strip()

    sales['Invoice Date'] = pd.to_datetime(sales['Invoice Date'], dayfirst=True)
    sales = sales.sort_values('Invoice Date').reset_index(drop=True)

    indent['SOH']        = pd.to_numeric(indent['SOH'],        errors='coerce').fillna(0)
    indent['Indent']     = pd.to_numeric(indent['Indent'],     errors='coerce').fillna(0)
    crt_size['Crt Size'] = pd.to_numeric(crt_size['Crt Size'], errors='coerce').fillna(1)
    crt_size['No Crts']  = pd.to_numeric(crt_size['No Crts'],  errors='coerce').fillna(1)

    # ── Save product display names BEFORE label encoding ──
    # Category + Material Description from SKU buy area
    desc_col = next((c for c in ['Material Description','Material description',
                                  'MATERIAL DESCRIPTION','Material Name','Material name']
                     if c in sku_buy.columns), None)
    cat_col  = next((c for c in ['Category','CATEGORY','category']
                     if c in sku_buy.columns), None)

    pm = sku_buy[['SAP Code']].copy()
    pm['category_label']  = sku_buy[cat_col].astype(str).str.strip()  if cat_col  else 'Unknown'
    pm['mat_description'] = sku_buy[desc_col].astype(str).str.strip() if desc_col else ''
    pm['vendor_name']     = sku_buy['VENDOR'].astype(str).str.strip() if 'VENDOR' in sku_buy.columns else ''
    pm['display_name']    = pm['category_label'] + ' — ' + pm['mat_description']
    product_meta = (pm.rename(columns={'SAP Code':'Material No'})
                      .drop_duplicates('Material No')
                      .set_index('Material No'))

    # ── 3. MERGES ─────────────────────────────────────────
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

    # ── 4. LABEL ENCODE ───────────────────────────────────
    cat_cols = ['Category', 'SKU', 'VENDOR', 'TYPE', 'Short Name', 'Material name']
    cat_cols = [c for c in cat_cols if c in sales.columns]
    for col in cat_cols:
        le = LabelEncoder()
        sales[col] = le.fit_transform(sales[col].astype(str))

    # ── 5. CORE PRICE & QUANTITY ──────────────────────────
    sales['PricePerUnit']   = sales['SaleValue'] / (sales['SalQty'].replace(0, np.nan)).fillna(1)
    sales['QtyXPrice']      = sales['SalQty'] * sales['PricePerUnit']
    sales['Log_SalQty']     = np.log1p(sales['SalQty'])
    sales['Log_Price']      = np.log1p(sales['PricePerUnit'])
    sales['Log_QtyXPrice']  = np.log1p(sales['QtyXPrice'])
    sales['Sqrt_QtyXPrice'] = np.sqrt(np.abs(sales['QtyXPrice']))

    # ── 6. STOCK & INVENTORY ──────────────────────────────
    sales['StockPressure'] = sales['SalQty'] / (sales['SOH'] + 1)
    sales['StockCoverage'] = sales['SOH']    / (sales['SalQty'] + 1)
    sales['IndentRatio']   = sales['Indent'] / (sales['SOH'] + 1)
    sales['Stock_Demand']  = sales['SOH']    - sales['SalQty']

    # ── 7. PER-MATERIAL STATISTICS ────────────────────────
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
    sales['TE_LOO_Mat']      = (sales['Mat_SumSale'] - sales['SaleValue']) / (sales['Mat_Count'] - 1 + 1e-9)
    alpha = 5
    sales['TE_Smooth_Mat']   = (sales['Mat_Count'] * sales['Mat_AvgSale'] + alpha * global_mean) / (sales['Mat_Count'] + alpha)
    sales['QtyXTESmooth']    = sales['SalQty'] * sales['TE_Smooth_Mat'] / (sales['Mat_AvgQty'] + 1)

    # ── 8. PER-GROUP STATISTICS ───────────────────────────
    for grp in ['SKU', 'Category', 'VENDOR']:
        if grp in sales.columns:
            sales[f'{grp}_AvgSale']  = sales.groupby(grp)['SaleValue'].transform('mean')
            sales[f'{grp}_MedSale']  = sales.groupby(grp)['SaleValue'].transform('median')
            sales[f'{grp}_MaxSale']  = sales.groupby(grp)['SaleValue'].transform('max')
            sales[f'{grp}_SumSale']  = sales.groupby(grp)['SaleValue'].transform('sum')
            sales[f'{grp}_AvgQty']   = sales.groupby(grp)['SalQty'].transform('mean')
            sales[f'{grp}_AvgPrice'] = sales.groupby(grp)['PricePerUnit'].transform('mean')
            sales[f'{grp}_Count']    = sales.groupby(grp)['SaleValue'].transform('count')

    sales['SKU_AvgPrice']    = sales.groupby('SKU')['PricePerUnit'].transform('mean')
    sales['Cat_AvgPrice']    = sales.groupby('Category')['PricePerUnit'].transform('mean')
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

    # ── 9. RECONSTRUCTION ENSEMBLE ───────────────────────
    recon_cols = ['QtyXMatAvgPrice','QtyXMatMedPrice','QtyXSKUAvgPrice',
                  'QtyXCatAvgPrice','QtyXCatWAvgPrice','QtyXTESmooth']
    recon_cols = [c for c in recon_cols if c in sales.columns]
    sales['ReconEnsemble']  = sales[recon_cols].mean(axis=1)
    sales['ResidFromRecon'] = sales['SaleValue'] - sales['ReconEnsemble']
    sales['Log_ReconEns']   = np.log1p(sales['ReconEnsemble'])
    sales['ExactPriceDev']  = sales['PricePerUnit'] - sales['Mat_AvgPrice']
    sales['QtyXPriceDev']   = sales['SalQty'] * sales['ExactPriceDev']

    # ── 10. DATE FEATURES ─────────────────────────────────
    sales['DayOfWeek']      = sales['Invoice Date'].dt.dayofweek
    sales['DayOfMonth']     = sales['Invoice Date'].dt.day
    sales['IsWeekend']      = sales['DayOfWeek'].isin([5,6]).astype(int)
    sales['DaysSinceStart'] = (sales['Invoice Date'] - sales['Invoice Date'].min()).dt.days
    sales['DOW_sin']        = np.sin(2*np.pi*sales['DayOfWeek']/7)
    sales['DOW_cos']        = np.cos(2*np.pi*sales['DayOfWeek']/7)
    mat_dow_mean   = sales.groupby(['Material No','DayOfWeek'])['SaleValue'].mean()
    mat_dow_mapped = sales[['Material No','DayOfWeek']].apply(
        lambda r: mat_dow_mean.get((r['Material No'], r['DayOfWeek']), np.nan), axis=1)
    sales['Mat_DOW_AvgSale']  = mat_dow_mapped.fillna(sales['Mat_AvgSale'])
    sales['QtyXMatDOWAvg']    = sales['SalQty'] * (sales['Mat_DOW_AvgSale'] / (sales['Mat_AvgQty']+1))
    mat_dow_count             = sales.groupby(['Material No','DayOfWeek'])['SaleValue'].transform('count')
    mat_dow_sum               = sales.groupby(['Material No','DayOfWeek'])['SaleValue'].transform('sum')
    sales['TE_Smooth_MatDOW'] = (mat_dow_sum - sales['SaleValue'] + 3*global_mean) / (mat_dow_count - 1 + 3 + 1e-9)
    sales['QtyXTE_MatDOW']    = sales['SalQty'] * sales['TE_Smooth_MatDOW'] / (sales['Mat_AvgQty']+1)

    # ── 11. DATE-LEVEL AGGREGATES ─────────────────────────
    date_total = sales.groupby('Invoice Date')['SaleValue'].sum()
    date_qty   = sales.groupby('Invoice Date')['SalQty'].sum()
    date_cnt   = sales.groupby('Invoice Date')['SaleValue'].count()
    prev = sales['Invoice Date'] - pd.Timedelta(days=1)
    sales['PrevDayTotalSale'] = prev.map(date_total).fillna(0)
    sales['PrevDayTotalQty']  = prev.map(date_qty).fillna(0)
    sales['PrevDayRowCount']  = prev.map(date_cnt).fillna(0)
    daily_sorted = date_total.sort_index()
    sales['DailyRoll2'] = sales['Invoice Date'].map(daily_sorted.shift(1).rolling(2,min_periods=1).mean()).fillna(0)
    sales['DailyRoll3'] = sales['Invoice Date'].map(daily_sorted.shift(1).rolling(3,min_periods=1).mean()).fillna(0)
    sales['DailyRoll5'] = sales['Invoice Date'].map(daily_sorted.shift(1).rolling(5,min_periods=1).mean()).fillna(0)
    sales['DailyRoll7'] = sales['Invoice Date'].map(daily_sorted.shift(1).rolling(7,min_periods=1).mean()).fillna(0)
    sales['DailyEWM3']  = sales['Invoice Date'].map(daily_sorted.shift(1).ewm(span=3,min_periods=1).mean()).fillna(0)
    sales['DailyEWM5']  = sales['Invoice Date'].map(daily_sorted.shift(1).ewm(span=5,min_periods=1).mean()).fillna(0)
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
    date_total_all = sales.groupby('Invoice Date')['SaleValue'].transform('sum')
    sales['SaleVsDateTotal'] = sales['SaleValue'] / (date_total_all + 1)
    dow_avg_total = (sales.groupby(['Invoice Date','DayOfWeek'])['SaleValue']
                     .sum().reset_index().groupby('DayOfWeek')['SaleValue'].mean())
    sales['DOW_AvgDailyTotal'] = sales['DayOfWeek'].map(dow_avg_total).fillna(global_mean)
    sales['SaleVsDOWTotal']    = sales['PrevDayTotalSale'] / (sales['DOW_AvgDailyTotal'] + 1)

    # ── 12. SHORT NAME STATS ──────────────────────────────
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

    # ── 13. RANK FEATURES ─────────────────────────────────
    sales['SalQty_rank']    = sales['SalQty'].rank(pct=True)
    sales['Price_rank']     = sales['PricePerUnit'].rank(pct=True)
    sales['MatAvg_rank']    = sales['Mat_AvgSale'].rank(pct=True)
    sales['QtyXPrice_rank'] = sales['QtyXPrice'].rank(pct=True)

    # ── 14. WEATHER ───────────────────────────────────────
    weather = {}
    for city, coords in LOCATIONS.items():
        t, h, w, r, ir = fetch_owm_weather(city, coords["lat"], coords["lon"], OWM_API_KEY)
        weather[city] = {"temp": t, "humidity": h, "wind": w, "rain": r, "is_rainy": ir}

    temp       = safe_avg(weather["Chennai"]["temp"],     weather["Bangalore"]["temp"],     30.0)
    humidity   = safe_avg(weather["Chennai"]["humidity"], weather["Bangalore"]["humidity"],  68.0)
    wind_speed = safe_avg(weather["Chennai"]["wind"],     weather["Bangalore"]["wind"],       5.0)
    rain_1h    = safe_avg(weather["Chennai"]["rain"],     weather["Bangalore"]["rain"],       0.0)
    is_rainy_c = weather["Chennai"]["is_rainy"]   if weather["Chennai"]["is_rainy"]   is not None else 0
    is_rainy_b = weather["Bangalore"]["is_rainy"] if weather["Bangalore"]["is_rainy"] is not None else 0
    is_rainy   = 1 if (is_rainy_c or is_rainy_b) else 0
    heat_index = temp + 0.33 * (humidity/100.0 * 6.105 * np.exp((17.27*temp)/(237.7+temp))) - 4.0

    sales['Temperature'] = temp;       sales['Humidity']  = humidity
    sales['WindSpeed']   = wind_speed; sales['Rain_1h']   = rain_1h
    sales['IsRainy']     = is_rainy;   sales['HeatIndex'] = heat_index
    sales['TempXQty']    = temp     * sales['SalQty']
    sales['TempXPrice']  = temp     * sales['PricePerUnit']
    sales['HumidXQty']   = humidity * sales['SalQty']
    sales['RainXQty']    = rain_1h  * sales['SalQty']

    # ── 15. TRAFFIC ───────────────────────────────────────
    sales['TrafficIndex']       = sales['DayOfWeek'].apply(lambda d: traffic_index(d, is_rainy))
    sales['DeliveryDelayIndex'] = sales['DayOfWeek'].apply(lambda d: delivery_delay_index(d, is_rainy))
    sales['TrafficXQty']        = sales['TrafficIndex'] * sales['SalQty']
    sales['TrafficXPrice']      = sales['TrafficIndex'] * sales['PricePerUnit']

    # ── 16. SEASON & SEASONAL PRODUCT ────────────────────
    sales['Month']      = sales['Invoice Date'].dt.month
    sales['SeasonName'] = sales['Month'].apply(get_season)
    sales['SeasonCode'] = sales['SeasonName'].map(season_map)
    sales['Month_sin']  = np.sin(2 * np.pi * sales['Month'] / 12)
    sales['Month_cos']  = np.cos(2 * np.pi * sales['Month'] / 12)
    sales['IsFestive']  = sales.apply(lambda r: is_festive(r['Month'], r['DayOfMonth']), axis=1)
    mat_season_avg = (sales.groupby(['Material No','SeasonCode'])['SaleValue']
                     .mean().reset_index().rename(columns={'SaleValue':'Mat_SeasonAvgSale'}))
    sales = sales.merge(mat_season_avg, on=['Material No','SeasonCode'], how='left')
    sales['Mat_SeasonAvgSale'] = sales['Mat_SeasonAvgSale'].fillna(sales['Mat_AvgSale'])
    sales['IsSeasonalProduct'] = (sales['Mat_SeasonAvgSale'] > sales['Mat_AvgSale']).astype(int)
    sales['SeasonalLift']      = sales['Mat_SeasonAvgSale'] / (sales['Mat_AvgSale'] + 1e-9)
    sales['SeasonalXQty']      = sales['IsSeasonalProduct'] * sales['SalQty']
    sales['SeasonalLiftXQty']  = sales['SeasonalLift']      * sales['SalQty']
    sales['FestiveXQty']       = sales['IsFestive']         * sales['SalQty']

    # ── 17. FEATURE LIST (identical to new.py) ────────────
    features = [
        'SalQty','PricePerUnit','QtyXPrice',
        'Log_SalQty','Log_Price','Log_QtyXPrice','Sqrt_QtyXPrice',
        'SOH','Indent','StockPressure','StockCoverage','IndentRatio','Stock_Demand',
        'Mat_AvgSale','Mat_MedSale','Mat_MaxSale','Mat_MinSale','Mat_StdSale',
        'Mat_SumSale','Mat_Count','Mat_AvgQty','Mat_AvgPrice','Mat_MedPrice',
        'Mat_SaleRange','Mat_SaleNorm','Mat_FreqWeight',
        'QtyXMatAvgPrice','QtyXMatMedPrice','ResidFromMat','PriceVsMatAvg',
        'QtyVsMatAvg','Log_QtyXMatAvg','Sale_MatZscore',
        'TE_LOO_Mat','TE_Smooth_Mat','QtyXTESmooth',
        'SKU_AvgSale','SKU_MedSale','SKU_MaxSale','SKU_SumSale',
        'SKU_AvgQty','SKU_AvgPrice','SKU_Count',
        'Category_AvgSale','Category_MedSale','Category_MaxSale',
        'Category_SumSale','Category_AvgQty','Category_AvgPrice',
        'VENDOR_AvgSale','VENDOR_SumSale','VENDOR_AvgPrice',
        'Cat_AvgPrice','SKU_AvgPrice',
        'QtyXSKUAvgPrice','QtyXCatAvgPrice','QtyXCatWAvgPrice',
        'ResidFromSKU','ResidFromCat','SaleVsSKURatio','SaleVsCatRatio',
        'PriceDevSKU','Cat_WAvgPrice','TE_LOO_SKU',
        'ReconEnsemble','ResidFromRecon','Log_ReconEns','ExactPriceDev','QtyXPriceDev',
        'DayOfWeek','DayOfMonth','IsWeekend','DaysSinceStart','DOW_sin','DOW_cos',
        'Mat_DOW_AvgSale','QtyXMatDOWAvg','TE_Smooth_MatDOW','QtyXTE_MatDOW',
        'PrevDayTotalSale','PrevDayTotalQty','PrevDayRowCount',
        'DailyRoll2','DailyRoll3','DailyRoll5','DailyRoll7','DailyEWM3','DailyEWM5',
        'DailyMomentum','DailyTrend3v7',
        'DateLOO_Sale','DateLOO_Qty','DateLOO_Mean',
        'PrevDayXPrice','PrevDayXQty','DailyRoll3XQty','DailyRoll3XPrice','DailyEWM3XQty',
        'SaleVsDateTotal','DOW_AvgDailyTotal','SaleVsDOWTotal',
        'SN_AvgSale','SN_SumSale','SN_AvgPrice','SN_AvgQty','QtyXSNAvgPrice','ResidFromSN',
        'SalQty_rank','Price_rank','MatAvg_rank','QtyXPrice_rank',
        'Category','SKU','VENDOR','TYPE','Short Name','Material name',
        'Temperature','WindSpeed','TempXQty','TempXPrice',
        'Humidity','Rain_1h','IsRainy','HeatIndex','HumidXQty','RainXQty',
        'TrafficIndex','DeliveryDelayIndex','TrafficXQty','TrafficXPrice',
        'Month','SeasonCode','Month_sin','Month_cos',
        'IsFestive','IsSeasonalProduct','SeasonalLift',
        'Mat_SeasonAvgSale','SeasonalXQty','SeasonalLiftXQty','FestiveXQty',
    ]

    # ── 18. PREPARE X / y ─────────────────────────────────
    sales = sales.loc[:, ~sales.columns.duplicated(keep='first')].reset_index(drop=True)
    seen = set(); features_clean = []
    for f in features:
        if f in sales.columns and f not in seen:
            features_clean.append(f); seen.add(f)
    features = features_clean

    X = sales[features].copy()
    y = sales['SaleValue']
    for col in X.columns:
        col_data = X[col]
        if isinstance(col_data, pd.DataFrame): col_data = col_data.iloc[:, 0]
        X[col] = pd.to_numeric(col_data, errors='coerce').fillna(0)

    # ── 19. 80/20 SPLIT ───────────────────────────────────
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()
    dates_train = sales['Invoice Date'].iloc[:split_idx]
    dates_test  = sales['Invoice Date'].iloc[split_idx:]

    # ── 20. RANDOM FOREST ─────────────────────────────────
    model = RandomForestRegressor(
        n_estimators=30000, max_depth=None,
        min_samples_split=2, min_samples_leaf=1,
        max_features=0.55, bootstrap=True,
        oob_score=True, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    test_preds  = model.predict(X_test)

    train_r2  = r2_score(y_train, train_preds)
    test_r2   = r2_score(y_test,  test_preds)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
    test_mae  = float(mean_absolute_error(y_test, test_preds))
    oob_r2    = model.oob_score_

    # ── 21. RETRAIN ON FULL DATA ──────────────────────────
    model_full = RandomForestRegressor(
        n_estimators=30000, max_depth=None,
        min_samples_split=2, min_samples_leaf=1,
        max_features=0.55, bootstrap=True,
        random_state=42, n_jobs=-1)
    model_full.fit(X, y)

    # ── NEXT-DAY CONTEXT (same as new.py section 21) ──────
    next_dow     = next_date.dayofweek
    next_month   = next_date.month
    next_season  = season_map.get(get_season(next_month), 1)
    next_traffic = traffic_index(next_dow, is_rainy)
    next_delay   = delivery_delay_index(next_dow, is_rainy)
    next_festive = is_festive(next_month, next_date.day)

    # ── PER-PRODUCT PREDICTIONS ───────────────────────────
    results = []
    for mat_no in sales['Material No'].unique():
        mat_rows = sales[sales['Material No'] == mat_no]
        if len(mat_rows) == 0: continue
        row = mat_rows.iloc[[-1]].copy()

        sq  = float(row['SalQty'].values[0])
        ppu = float(row['PricePerUnit'].values[0])
        row['Temperature']=temp;       row['Humidity']   =humidity
        row['WindSpeed']  =wind_speed; row['Rain_1h']    =rain_1h
        row['IsRainy']    =is_rainy;   row['HeatIndex']  =heat_index
        row['TempXQty']   =temp*sq;    row['TempXPrice'] =temp*ppu
        row['HumidXQty']  =humidity*sq; row['RainXQty']  =rain_1h*sq
        row['TrafficIndex']      =next_traffic
        row['DeliveryDelayIndex']=next_delay
        row['TrafficXQty']       =next_traffic*sq
        row['TrafficXPrice']     =next_traffic*ppu
        row['Month']     =next_month;  row['SeasonCode']=next_season
        row['Month_sin'] =np.sin(2*np.pi*next_month/12)
        row['Month_cos'] =np.cos(2*np.pi*next_month/12)
        row['IsFestive'] =next_festive
        row['Stock_Demand']=float(row['SOH'].values[0])-sq

        nx = row[features].copy()
        nx = nx.loc[:, ~nx.columns.duplicated(keep='first')]
        for col in nx.columns:
            nx[col] = pd.to_numeric(nx[col], errors='coerce').fillna(0)

        pred      = model_full.predict(nx)[0]
        avg_price = float(mat_rows['PricePerUnit'].mean())
        pred_qty  = pred / avg_price if avg_price > 0 else 0.0

        if mat_no in product_meta.index:
            display_name  = product_meta.loc[mat_no, 'display_name']
            category_lbl  = product_meta.loc[mat_no, 'category_label']
            mat_desc      = product_meta.loc[mat_no, 'mat_description']
            vendor_name   = product_meta.loc[mat_no, 'vendor_name']
        else:
            display_name = str(mat_no)
            category_lbl = 'Unknown'
            mat_desc     = str(mat_no)
            vendor_name  = 'Unknown'

        results.append(dict(
            material_no    = mat_no,
            display_name   = display_name,
            category       = category_lbl,
            mat_description= mat_desc,
            vendor         = vendor_name,
            predicted_sale = round(float(pred), 2),
            predicted_qty  = round(max(float(pred_qty), 0.0), 2),
            avg_price      = round(avg_price, 2),
            last_actual    = round(float(mat_rows['SaleValue'].iloc[-1]), 2),
            avg_sale       = round(float(mat_rows['SaleValue'].mean()), 2),
            season         = get_season(next_month),
            is_seasonal    = int(mat_rows['IsSeasonalProduct'].iloc[-1]),
            seasonal_lift  = round(float(mat_rows['SeasonalLift'].iloc[-1]), 2),
            soh            = round(float(mat_rows['SOH'].iloc[-1]), 2),
            indent         = round(float(mat_rows['Indent'].iloc[-1]), 2),
            is_festive_day = next_festive,
            traffic_index  = round(next_traffic, 2),
            delay_index    = round(next_delay, 2),
        ))

    per_product_df = (pd.DataFrame(results)
                      .sort_values('predicted_sale', ascending=False)
                      .reset_index(drop=True))

    # ── CHART DATA ────────────────────────────────────────
    daily_actual = (sales.groupby('Invoice Date')['SaleValue']
                    .sum().reset_index().sort_values('Invoice Date'))
    all_preds_df = pd.DataFrame({
        'Invoice Date': sales['Invoice Date'].values,
        'Predicted':    np.concatenate([train_preds, test_preds]),
    })
    daily_pred_all = (all_preds_df.groupby('Invoice Date')['Predicted']
                      .sum().reset_index().sort_values('Invoice Date'))

    return dict(
        per_product_df = per_product_df,
        total_pred     = float(per_product_df['predicted_sale'].sum()),
        metrics        = dict(train_r2=float(train_r2), test_r2=float(test_r2),
                              rmse=test_rmse, mae=test_mae, oob_r2=float(oob_r2),
                              n_features=len(features)),
        daily_actual   = daily_actual,
        daily_pred_all = daily_pred_all,
        dates_train    = dates_train,
        dates_test     = dates_test,
        weather        = dict(temp=temp, humidity=humidity, wind_speed=wind_speed,
                              rain_1h=rain_1h, is_rainy=is_rainy, heat_index=heat_index,
                              chennai_temp=weather["Chennai"]["temp"],
                              bangalore_temp=weather["Bangalore"]["temp"]),
        sales_raw      = sales,
        next_date      = next_date,
    )