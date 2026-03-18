import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os, tempfile

st.set_page_config(page_title="PKM Veg Shop — Sales Forecast", page_icon="🥦", layout="wide")

st.markdown("""
<style>
.main-title{font-size:2rem;font-weight:800;color:#1a5c2a;margin-bottom:0}
.sub-title{font-size:0.92rem;color:#666;margin-bottom:1.5rem}
.kpi-card{background:linear-gradient(135deg,#f0faf3,#e6f7ec);border:1px solid #b2dfbb;
          border-radius:12px;padding:1rem 0.8rem;text-align:center;
          box-shadow:0 2px 6px rgba(0,0,0,0.06)}
.kpi-val{font-size:1.6rem;font-weight:800;color:#1a5c2a}
.kpi-lbl{font-size:0.72rem;color:#555;margin-top:3px;text-transform:uppercase;letter-spacing:.4px}
.wx-card{background:linear-gradient(135deg,#e3f2fd,#f0f4ff);border:1px solid #90caf9;
         border-radius:12px;padding:1rem 1.2rem;font-size:0.87rem;line-height:1.9}
.sec-hdr{font-size:1.1rem;font-weight:700;color:#1a5c2a;border-left:4px solid #2e7d32;
         padding-left:10px;margin:1.4rem 0 0.7rem 0}
.det-box{background:#fafffe;border:1px solid #c8e6c9;border-radius:10px;
         padding:1rem 1.2rem;font-size:0.87rem;line-height:1.9}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🥦 PKM Veg Shop — Next Day Sales Forecast</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">RandomForestRegressor · OpenWeatherMap (Chennai + Bangalore) · Traffic · Season · Per-Product</p>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    uploaded = st.file_uploader("📂 Upload PKM_Project_Dataset.xlsx", type=["xlsx"])
    next_date_input = st.date_input("📅 Forecast Date", value=pd.Timestamp("2025-03-22"))
    st.markdown("---")
    run_btn = st.button("🚀 Run Forecast", use_container_width=True, type="primary")
    st.markdown("---")
    st.markdown("**Model:** RandomForestRegressor  \n**Trees:** 30,000  \n**Split:** 80/20 time-ordered  \n**Weather:** OpenWeatherMap  \n**Locations:** Chennai + Bangalore")

if "results" not in st.session_state:
    st.session_state.results = None

if run_btn:
    if uploaded is None:
        st.error("⚠️ Please upload the Excel file first.")
        st.stop()
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
     tmp.write(uploaded.read())
    tmp_path = tmp.name
    with st.spinner("🔄 Running pipeline — ~2 min for 30,000 trees..."):
        try:
            from model import run_pipeline
            res = run_pipeline(tmp_path, str(next_date_input))
            st.session_state.results = res
            st.success("✅ Done!")
        except Exception as e:
            import traceback
            st.error(f"❌ Error: {e}")
            st.code(traceback.format_exc())
            st.stop()
    
        finally:
            os.unlink(tmp_path)

if st.session_state.results is None:
    st.info("👈 Upload your Excel file and click **🚀 Run Forecast**.")
    st.stop()

# ── Unpack ────────────────────────────────────────────────
res       = st.session_state.results
df        = res['per_product_df']
m         = res['metrics']
w         = res['weather']
sales     = res['sales_raw']
next_date = res['next_date']

from model import get_season, traffic_index, FESTIVE_WINDOWS

# ══════════════════════════════════════════════════════════
# SECTION 1 — KPI CARDS
# ══════════════════════════════════════════════════════════
st.markdown('<p class="sec-hdr">📊 Model Summary</p>', unsafe_allow_html=True)
c1,c2,c3,c4,c5,c6 = st.columns(6)
kpis = [
    (f"₹{res['total_pred']:,.0f}", f"Total Forecast · {next_date.strftime('%d %b %Y')}"),
    (f"{m['test_r2']:.4f}",        "Test R²"),
    (f"{m['oob_r2']:.4f}",         "OOB R²"),
    (f"{m['train_r2']:.4f}",       "Train R²"),
    (f"₹{m['rmse']:,.0f}",         "RMSE"),
    (f"₹{m['mae']:,.0f}",          "MAE"),
]
for col,(val,lbl) in zip([c1,c2,c3,c4,c5,c6], kpis):
    col.markdown(f'<div class="kpi-card"><div class="kpi-val">{val}</div>'
                 f'<div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# SECTION 2 — WEATHER + TOP 15 BAR CHART
# ══════════════════════════════════════════════════════════
st.markdown('<p class="sec-hdr">🌡️ Live Conditions & Top Products</p>', unsafe_allow_html=True)
wc1, wc2 = st.columns([1.3, 2.7])

with wc1:
    rain_icon   = "🌧️" if w['is_rainy'] else "☀️"
    season_now  = get_season(next_date.month)
    season_icon = {"Summer":"🔥","SouthWest_Monsoon":"🌧️","NorthEast_Monsoon":"🌦️","Winter":"❄️"}.get(season_now,"🌿")
    festive_now = any(next_date.month==m2 and d1<=next_date.day<=min(d2,31) for m2,d1,d2 in FESTIVE_WINDOWS)
    ch_t  = f"{w['chennai_temp']:.1f}°C"   if w['chennai_temp']   is not None else "—"
    bn_t  = f"{w['bangalore_temp']:.1f}°C" if w['bangalore_temp'] is not None else "—"
    ti    = traffic_index(next_date.dayofweek, w['is_rainy'])
    st.markdown(f"""
    <div class="wx-card">
    <b>🌡️ Weather (Chennai + Bangalore avg)</b><br>
    {rain_icon} <b>Temp:</b> {w['temp']:.1f}°C &nbsp;|&nbsp; 💧 <b>Humidity:</b> {w['humidity']:.0f}%<br>
    🌬️ <b>Wind:</b> {w['wind_speed']:.1f} m/s &nbsp;|&nbsp; 🌧️ <b>Rain:</b> {w['rain_1h']:.1f} mm<br>
    🌡️ <b>Heat Index:</b> {w['heat_index']:.1f}°C<br>
    📍 Chennai: <b>{ch_t}</b> &nbsp;|&nbsp; Bangalore: <b>{bn_t}</b><br>
    ────────────────────────────<br>
    {season_icon} <b>Season:</b> {season_now.replace('_',' ')}<br>
    🚦 <b>Traffic Index:</b> {ti:.2f} / 1.00<br>
    {'🎉 <b>Festive Period!</b>' if festive_now else '📅 Regular Day'}
    </div>
    """, unsafe_allow_html=True)

with wc2:
    top15 = df.head(15).copy()
    fig_bar = px.bar(top15, x='predicted_sale', y='display_name', orientation='h',
                     color='predicted_sale', color_continuous_scale='Greens',
                     title=f"Top 15 Products — {next_date.strftime('%d %b %Y')}",
                     text=top15['predicted_sale'].apply(lambda x: f"₹{x:,.0f}"),
                     labels={'predicted_sale':'Predicted Sale (₹)','display_name':'Product'})
    fig_bar.update_traces(textposition='outside', textfont_size=10)
    fig_bar.update_layout(height=400, coloraxis_showscale=False,
                          yaxis=dict(autorange='reversed', tickfont=dict(size=10)),
                          margin=dict(l=0,r=30,t=45,b=10),
                          plot_bgcolor='#f9f9f9', paper_bgcolor='white')
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════
# SECTION 3 — DAILY HISTORY LINE CHART
# ══════════════════════════════════════════════════════════
st.markdown('<p class="sec-hdr">📈 Daily Sales History + Forecast</p>', unsafe_allow_html=True)

da  = res['daily_actual']; dp = res['daily_pred_all']
dt  = res['dates_train'];  dte= res['dates_test']
boundary = dt.iloc[-1]
dp_train = dp[dp['Invoice Date'] <= boundary]
dp_test  = dp[dp['Invoice Date'] >= dte.iloc[0]]

fig_line = go.Figure()
fig_line.add_trace(go.Scatter(x=da['Invoice Date'], y=da['SaleValue'],
    fill='tozeroy', fillcolor='rgba(70,130,180,0.08)',
    line=dict(width=0), showlegend=False, hoverinfo='skip'))
fig_line.add_trace(go.Scatter(x=da['Invoice Date'], y=da['SaleValue'],
    mode='lines+markers', name='Actual Daily Sale',
    line=dict(color='steelblue', width=2.5), marker=dict(size=6),
    hovertemplate='%{x|%d %b}<br>Actual: ₹%{y:,.0f}<extra></extra>'))
fig_line.add_trace(go.Scatter(x=dp_train['Invoice Date'], y=dp_train['Predicted'],
    mode='lines+markers', name='Train Predicted',
    line=dict(color='#2e7d32', width=2), marker=dict(symbol='square', size=5),
    hovertemplate='%{x|%d %b}<br>Train: ₹%{y:,.0f}<extra></extra>'))
fig_line.add_trace(go.Scatter(x=dp_test['Invoice Date'], y=dp_test['Predicted'],
    mode='lines+markers', name=f"Test Predicted (R²={m['test_r2']:.4f})",
    line=dict(color='#2e7d32', width=2, dash='dash'), marker=dict(symbol='square', size=7),
    hovertemplate='%{x|%d %b}<br>Test: ₹%{y:,.0f}<extra></extra>'))
fig_line.add_vline(x=str(boundary.date()), line_dash='dot', line_color='orange', line_width=1.8,
    annotation_text=f"Train/Test split ({boundary.strftime('%d %b')})",
    annotation_position='top left', annotation_font_color='orange')
fig_line.add_trace(go.Scatter(x=[next_date], y=[res['total_pred']],
    mode='markers+text', name=f"Forecast {next_date.strftime('%d %b')}",
    marker=dict(color='red', size=16, symbol='star'),
    text=[f"  ₹{res['total_pred']:,.0f}"], textposition='middle right',
    textfont=dict(color='red', size=12),
    hovertemplate=f"Forecast: ₹{res['total_pred']:,.0f}<extra></extra>"))
fig_line.update_layout(height=420, xaxis_title="Date", yaxis_title="Total Sale (₹)",
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    yaxis=dict(tickprefix='₹', tickformat=',.0f'),
    plot_bgcolor='#f9f9f9', paper_bgcolor='white', hovermode='x unified')
st.plotly_chart(fig_line, use_container_width=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════
# SECTION 4 — PER-PRODUCT TABLE
# ══════════════════════════════════════════════════════════
st.markdown(f'<p class="sec-hdr">📦 All Products — {next_date.strftime("%d %b %Y")}</p>', unsafe_allow_html=True)

f1, f2, f3, f4 = st.columns([3,2,2,2])
with f1: search = st.text_input("🔍 Search", placeholder="Category or product name...")
with f2:
    cats = ["All"] + sorted(df['category'].dropna().unique().tolist())
    cat_filter = st.selectbox("🗂️ Category", cats)
with f3:
    seas_filter = st.selectbox("🌿 Season", ["All","In-Season Only","Off-Season Only"])
with f4:
    sort_by = st.selectbox("↕️ Sort By", ["predicted_sale","predicted_qty","avg_price","seasonal_lift","soh"])

disp = df.copy()
if search:
    mask = (disp['display_name'].str.contains(search,case=False,na=False) |
            disp['category'].str.contains(search,case=False,na=False) |
            disp['mat_description'].str.contains(search,case=False,na=False))
    disp = disp[mask]
if cat_filter != "All":   disp = disp[disp['category']==cat_filter]
if seas_filter=="In-Season Only":  disp = disp[disp['is_seasonal']==1]
elif seas_filter=="Off-Season Only": disp = disp[disp['is_seasonal']==0]
disp = disp.sort_values(sort_by, ascending=False).reset_index(drop=True)

st.caption(f"Showing **{len(disp)}** of **{len(df)}** products")

show = disp[['material_no','category','mat_description','predicted_sale','predicted_qty',
             'avg_price','last_actual','avg_sale','soh','indent',
             'seasonal_lift','is_seasonal','season','traffic_index','vendor']].copy()
show.columns = ['Mat No','Category','Material Description','Pred Sale (₹)','Pred Qty',
                'Avg Price (₹)','Last Actual (₹)','Avg Sale (₹)','SOH','Indent',
                'Seasonal Lift','In Season','Season','Traffic Index','Vendor']
for c in ['Pred Sale (₹)','Last Actual (₹)','Avg Sale (₹)','Avg Price (₹)']:
    show[c] = show[c].apply(lambda x: f"₹{x:,.2f}")
show['Pred Qty']      = show['Pred Qty'].apply(lambda x: f"{x:,.1f}")
show['SOH']           = show['SOH'].apply(lambda x: f"{x:,.1f}")
show['Indent']        = show['Indent'].apply(lambda x: f"{x:,.1f}")
show['Seasonal Lift'] = show['Seasonal Lift'].apply(lambda x: f"{x:.2f}×")
show['In Season']     = show['In Season'].apply(lambda x: "✅" if x==1 else "—")
show['Traffic Index'] = show['Traffic Index'].apply(lambda x: f"{x:.2f}")

st.dataframe(show, use_container_width=True, height=440)
st.download_button("⬇️ Download CSV",
    data=disp.to_csv(index=False).encode('utf-8'),
    file_name=f"PKM_predictions_{next_date.strftime('%Y%m%d')}.csv",
    mime="text/csv")
st.markdown("---")

# ══════════════════════════════════════════════════════════
# SECTION 5 — PRODUCT DRILL-DOWN
# ══════════════════════════════════════════════════════════
st.markdown('<p class="sec-hdr">🔎 Product Drill-Down</p>', unsafe_allow_html=True)

selected = st.selectbox("Select a product (Category — Material Description)",
                        df['display_name'].tolist(), index=0)

if selected:
    row = df[df['display_name']==selected].iloc[0]
    mat_no = row['material_no']

    d1,d2,d3,d4 = st.columns(4)
    d1.metric("🎯 Predicted Sale",  f"₹{row['predicted_sale']:,.2f}", delta=f"avg ₹{row['avg_sale']:,.0f}")
    d2.metric("📦 Predicted Qty",   f"{row['predicted_qty']:,.1f} units")
    d3.metric("💰 Avg Price/Unit",  f"₹{row['avg_price']:,.2f}")
    d4.metric("📈 Seasonal Lift",   f"{row['seasonal_lift']:.2f}×",
              delta="In Season ✅" if row['is_seasonal']==1 else "Off Season")

    di1, di2 = st.columns(2)
    with di1:
        st.markdown(f"""<div class="det-box">
        <b>📋 Product Info</b><br>
        🔢 <b>Material No:</b> {row['material_no']}<br>
        🗂️ <b>Category:</b> {row['category']}<br>
        📝 <b>Description:</b> {row['mat_description']}<br>
        🏭 <b>Vendor:</b> {row['vendor']}<br>
        📦 <b>SOH:</b> {row['soh']:,.1f} &nbsp;|&nbsp; 📋 <b>Indent:</b> {row['indent']:,.1f}
        </div>""", unsafe_allow_html=True)

    with di2:
        s_icon = {"Summer":"🔥","SouthWest_Monsoon":"🌧️","NorthEast_Monsoon":"🌦️","Winter":"❄️"}.get(row['season'],"🌿")
        st.markdown(f"""<div class="det-box">
        <b>🌍 Forecast Context</b><br>
        {s_icon} <b>Season:</b> {row['season'].replace('_',' ')}<br>
        {'🌱' if row['is_seasonal'] else '💤'} <b>In Season:</b> {'Yes ✅' if row['is_seasonal'] else 'No'}<br>
        📊 <b>Seasonal Lift:</b> {row['seasonal_lift']:.2f}×<br>
        🚦 <b>Traffic Index:</b> {row['traffic_index']:.2f}<br>
        🚚 <b>Delivery Delay:</b> {row['delay_index']:.2f}<br>
        {'🎉 <b>Festive Period!</b>' if row['is_festive_day'] else '📅 Regular Day'}<br>
        🌡️ <b>Temp:</b> {w['temp']:.1f}°C &nbsp;|&nbsp; 💧 <b>Humidity:</b> {w['humidity']:.0f}%
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Product history chart
    prod_hist = (sales[sales['Material No']==mat_no]
                 .groupby('Invoice Date')['SaleValue'].sum().reset_index())

    fig_prod = go.Figure()
    fig_prod.add_trace(go.Scatter(x=prod_hist['Invoice Date'], y=prod_hist['SaleValue'],
        fill='tozeroy', fillcolor='rgba(46,125,50,0.10)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig_prod.add_trace(go.Scatter(x=prod_hist['Invoice Date'], y=prod_hist['SaleValue'],
        mode='lines+markers', name='Actual Sale',
        line=dict(color='#1565c0', width=2.5), marker=dict(size=7),
        hovertemplate='%{x|%d %b}<br>₹%{y:,.0f}<extra></extra>'))
    fig_prod.add_hline(y=row['avg_sale'], line_dash='dot', line_color='gray',
                       annotation_text=f"Avg ₹{row['avg_sale']:,.0f}",
                       annotation_position='bottom right', annotation_font_color='gray')
    fig_prod.add_trace(go.Scatter(x=[next_date], y=[row['predicted_sale']],
        mode='markers+text', name='Forecast',
        marker=dict(color='red', size=16, symbol='star'),
        text=[f"  ₹{row['predicted_sale']:,.0f}"], textposition='middle right',
        textfont=dict(color='red', size=12),
        hovertemplate=f"Forecast: ₹{row['predicted_sale']:,.0f}<extra></extra>"))
    fig_prod.update_layout(
        title=f"<b>{row['category']} — {row['mat_description']}</b>  ·  History & {next_date.strftime('%d %b')} Forecast",
        height=360, xaxis_title="Date", yaxis_title="Sale Value (₹)",
        yaxis=dict(tickprefix='₹', tickformat=',.0f'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        plot_bgcolor='#f9f9f9', paper_bgcolor='white', hovermode='x unified')
    st.plotly_chart(fig_prod, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════
# SECTION 6 — CATEGORY SUMMARY
# ══════════════════════════════════════════════════════════
st.markdown('<p class="sec-hdr">📊 Category-Level Predicted Sales</p>', unsafe_allow_html=True)

cat_sum = (df.groupby('category')
           .agg(total_predicted=('predicted_sale','sum'),
                product_count  =('predicted_sale','count'),
                avg_lift       =('seasonal_lift','mean'))
           .reset_index().sort_values('total_predicted', ascending=False))

fig_cat = px.bar(cat_sum, x='category', y='total_predicted',
    color='avg_lift', color_continuous_scale='RdYlGn',
    text=cat_sum['total_predicted'].apply(lambda x: f"₹{x:,.0f}"),
    title=f"Predicted Sale by Category — {next_date.strftime('%d %b %Y')}",
    labels={'total_predicted':'Total Predicted Sale (₹)','category':'Category','avg_lift':'Avg Seasonal Lift'})
fig_cat.update_traces(textposition='outside', textfont_size=10)
fig_cat.update_layout(height=360, plot_bgcolor='#f9f9f9', paper_bgcolor='white', xaxis_tickangle=-30)
st.plotly_chart(fig_cat, use_container_width=True)

st.markdown("---")
st.markdown("<center><small>PKM Veg Shop · RandomForestRegressor · OpenWeatherMap (Chennai + Bangalore)</small></center>",
            unsafe_allow_html=True)