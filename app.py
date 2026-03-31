import streamlit as st
import pandas as pd
import numpy as np
import hdbscan

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

if "file_name" not in st.session_state:
    st.session_state.file_name = None

def map_columns(df):
    col_map = {}

    for c in df.columns:
        col = c.lower()

        if 'customer' in col:
            col_map['CustomerID'] = c
        elif 'date' in col:
            col_map['InvoiceDate'] = c
        elif 'quantity' in col:
            col_map['Quantity'] = c
        elif 'price' in col or 'amount' in col:
            col_map['UnitPrice'] = c

    return col_map
# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Customer Intelligence System", layout="wide")

st.title("🚀 Customer Intelligence Platform")
st.markdown("### Smart Segmentation • Insights • Decision Support")

# ==============================
# ==============================
# FILE UPLOAD
# ==============================
uploaded_file = st.file_uploader("Upload Dataset", type=["csv","xlsx"])

if uploaded_file is not None:

    # ---------- SESSION TRACK ----------
    if "file_name" not in st.session_state:
        st.session_state.file_name = None

    if st.session_state.file_name != uploaded_file.name:
        st.session_state.clear()
        st.session_state.file_name = uploaded_file.name
        st.rerun()

    # ---------- LOAD DATA ----------
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, encoding='latin1')
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ Dataset Loaded Successfully")
    st.info(f"📁 Current File: {uploaded_file.name}")

    # ==============================
    # 🔥 AUTO COLUMN MAPPING
    # ==============================
    def map_columns(df):
        col_map = {}

        for c in df.columns:
            col = c.lower()

            if 'customer' in col:
                col_map['CustomerID'] = c
            elif 'date' in col:
                col_map['InvoiceDate'] = c
            elif 'quantity' in col:
                col_map['Quantity'] = c
            elif 'price' in col or 'amount' in col:
                col_map['UnitPrice'] = c

        return col_map

    col_map = map_columns(df)

    df = df.rename(columns={
        col_map.get('CustomerID'): 'CustomerID',
        col_map.get('InvoiceDate'): 'InvoiceDate',
        col_map.get('Quantity'): 'Quantity',
        col_map.get('UnitPrice'): 'UnitPrice'
    })

    # ---------- HANDLE MISSING CUSTOMER ----------
    if 'CustomerID' not in df.columns:
        df['CustomerID'] = df.index

    # ---------- FINAL VALIDATION ----------
    required_cols = ['CustomerID','InvoiceDate','Quantity','UnitPrice']

    if not all(col in df.columns for col in required_cols):
        st.error("❌ Unable to detect required columns automatically")
        st.write("Detected columns:", df.columns.tolist())
        st.stop()

    st.success("✅ Columns mapped successfully!")
    st.write("🔍 Detected Mapping:", col_map)

    # 👉 Continue your pipeline BELOW this line

    # ==============================
    # DATA OVERVIEW
    # ==============================
    st.subheader("📊 Dataset Overview")

    c1, c2 = st.columns(2)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])

    with st.expander("Preview Data"):
        st.dataframe(df.head())

    # ==============================
    # CLEANING
    # ==============================
    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['InvoiceDate'])

    if 'InvoiceDate' not in df.columns:
        st.error("❌ Date column is mandatory for analysis")
        st.stop()

# ==============================
# 🔥 ADAPTIVE RFM CREATION
# ==============================

    st.subheader("📊 Feature Engineering (RFM)")

    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    # -------- RECENCY --------
    recency = df.groupby('CustomerID')['InvoiceDate'].apply(
        lambda x: (reference_date - x.max()).days
    )

    # -------- FREQUENCY --------
    if 'InvoiceNo' in df.columns:
        frequency = df.groupby('CustomerID')['InvoiceNo'].nunique()
        st.success("✔ Frequency calculated using InvoiceNo")
    else:
        # fallback → count transactions
        frequency = df.groupby('CustomerID').size()
        st.warning("⚠ InvoiceNo missing → using transaction count as Frequency")

    # -------- MONETARY --------
    if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
        monetary = df.groupby('CustomerID').apply(
            lambda x: (x['Quantity'] * x['UnitPrice']).sum()
        )
        st.success("✔ Monetary calculated using Quantity × UnitPrice")
    else:
        # fallback → use UnitPrice only
        monetary = df.groupby('CustomerID')['UnitPrice'].sum()
        st.warning("⚠ Quantity missing → using UnitPrice sum as Monetary")

    # -------- FINAL RFM --------
    rfm = pd.DataFrame({
        'CustomerID': recency.index,
        'Recency': recency.values,
        'Frequency': frequency.values,
        'Monetary': monetary.values
    })

    rfm.head()
   # ==============================
    # 🔥 TOP PRODUCT (SMART DETECTION)
    # ==============================

    product_col = None

    for col in df.columns:
        if any(word in col.lower() for word in ['description','product','item']):
            product_col = col
            break

    if product_col:
        top_product = df.groupby('CustomerID')[product_col].agg(
            lambda x: x.value_counts().index[0]
        )
        rfm['Top_Product'] = rfm['CustomerID'].map(top_product)
        st.success(f"✔ Top Product from column: {product_col}")
    else:
        rfm['Top_Product'] = "Not Available"
        st.warning("⚠ No product-related column found") # ==============================
    # OUTLIER REMOVAL
    # ==============================
    rfm_filtered = rfm[
        (rfm['Monetary'] < rfm['Monetary'].quantile(0.99)) &
        (rfm['Frequency'] < rfm['Frequency'].quantile(0.99))
    ].copy()

    # ==============================
    # SCALING
    # ==============================
    scaler = StandardScaler()
    scaled = scaler.fit_transform(np.log1p(rfm_filtered[['Recency','Frequency','Monetary']]))

    # ==============================
    # HDBSCAN (Noise Removal)
    # ==============================
    hdb = hdbscan.HDBSCAN(min_cluster_size=50)
    rfm_filtered['Noise'] = hdb.fit_predict(scaled)

    clean_data = rfm_filtered[rfm_filtered['Noise'] != -1]

    # ==============================
    # KMEANS (3 CLUSTERS)
    # ==============================
    kmeans = KMeans(n_clusters=3, random_state=42)

    clean_scaled = scaler.fit_transform(np.log1p(clean_data[['Recency','Frequency','Monetary']]))
    clean_data['Cluster'] = kmeans.fit_predict(clean_scaled)

    # ==============================
    # MAP BACK
    # ==============================
    rfm['Cluster'] = -1
    rfm.loc[clean_data.index, 'Cluster'] = clean_data['Cluster']

    # ==============================
    # ORDER CLUSTERS
    # ==============================
    cluster_means = rfm.groupby('Cluster')['Monetary'].mean().sort_values()
    cluster_map = {old: new for new, old in enumerate(cluster_means.index)}
    rfm['Cluster'] = rfm['Cluster'].map(cluster_map)

    # ==============================
    # PERSONA
    # ==============================
    persona_map = {
        -1: "Noise",
        0: "Low Value",
        1: "General",
        2: "High Value"
    }

    rfm['Persona'] = rfm['Cluster'].map(persona_map)

    # ==============================
    # MARKETING STRATEGY
    # ==============================
    def strategy(p):
        if p == "High Value":
            return "Premium offers, loyalty rewards, exclusive deals"
        elif p == "General":
            return "Upselling, personalized discounts"
        elif p == "Low Value":
            return "Discounts, engagement campaigns"
        else:
            return "Re-engagement campaigns"

    rfm['Marketing_Strategy'] = rfm['Persona'].apply(strategy)

    # ==============================
    # CHURN
    # ==============================
    threshold = rfm['Recency'].quantile(0.75)
    rfm['Churn_Status'] = np.where(rfm['Recency'] > threshold, "At Risk", "Active")
        # ==============================
    # 💡 SMART BUSINESS INSIGHTS
    # ==============================

    st.subheader("💡 Business Insights")

    total_customers = len(rfm)

    high_value = (rfm['Persona'] == "High Value").sum()
    low_value = (rfm['Persona'] == "Low Value").sum()
    general = (rfm['Persona'] == "General").sum()

    churn_risk = (rfm['Churn_Status'] == "At Risk").sum()

    high_pct = round((high_value / total_customers) * 100, 2)
    low_pct = round((low_value / total_customers) * 100, 2)
    churn_pct = round((churn_risk / total_customers) * 100, 2)

    # ==============================
    # 🔥 INSIGHTS OUTPUT
    # ==============================

    st.success(f"💰 {high_pct}% of customers are High Value — major revenue drivers")
    st.warning(f"⚠ {churn_pct}% customers are at risk of churn")
    st.info(f"📉 {low_pct}% customers are Low Value — potential growth segment")

    # ==============================
    # 🔥 TOP SEGMENT ANALYSIS
    # ==============================

    top_segment = rfm.groupby('Persona')['Monetary'].sum().idxmax()

    st.success(f"🏆 {top_segment} segment contributes the highest revenue")

    # ==============================
    # 🔥 ACTIONABLE RECOMMENDATIONS
    # ==============================

    st.markdown("### 🎯 Recommended Actions")

    if high_pct > 30:
        st.write("✔ Focus on loyalty programs for High Value customers")

    if churn_pct > 20:
        st.write("⚠ Immediate retention campaigns required for churn-risk customers")

    if low_pct > 40:
        st.write("📈 Convert Low Value customers through targeted offers")

    if general > high_value:
        st.write("🔄 Upsell General customers to increase revenue")

    # ==============================
    # 🔥 PRODUCT INSIGHT (IF AVAILABLE)
    # ==============================

    if 'Top_Product' in rfm.columns:
        top_products = rfm['Top_Product'].value_counts().head(3)
        st.markdown("### 🛍️ Popular Products")
        st.write(top_products)
    # ==============================
    # DASHBOARD
    # ==============================
    st.subheader("📊 Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", len(rfm))
    c2.metric("High Value", (rfm['Persona']=="High Value").sum())
    c3.metric("At Risk", (rfm['Churn_Status']=="At Risk").sum())

# ==============================
# 📈 VISUAL INSIGHTS
# ==============================

    st.subheader("📊 Customer Segment Distribution")
    st.bar_chart(rfm['Persona'].value_counts())

    st.subheader("💰 Revenue Contribution by Segment")
    st.bar_chart(rfm.groupby('Persona')['Monetary'].sum())

    st.subheader("⚠ Customer Churn Distribution")
    st.bar_chart(rfm['Churn_Status'].value_counts())
    # ==============================
    # OUTPUT
    # ==============================
    st.subheader("📄 Final Output")

    final = rfm[['CustomerID','Recency','Frequency','Monetary',
                 'Cluster','Persona','Marketing_Strategy',
                 'Top_Product','Churn_Status']]

    st.dataframe(final.head())

    # ==============================
    # DOWNLOAD CSV
    # ==============================
    csv = final.to_csv(index=False).encode('utf-8')

    st.download_button(
        "⬇ Download CSV",
        csv,
        "customer_output.csv",
        "text/csv"
    )
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from io import BytesIO

    # ==============================
    # PDF GENERATION (FULL DATA)
    # ==============================
    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(pdf_buffer)
    styles = getSampleStyleSheet()

    elements = []

    # Title
    elements.append(Paragraph("Customer Segmentation Report", styles['Title']))
    elements.append(Spacer(1, 12))

    # -------- SELECT DATA --------
    pdf_data = final.copy()   # ✅ FULL DATASET

    # Convert to list
    data = [pdf_data.columns.tolist()] + pdf_data.values.tolist()

    # -------- SPLIT INTO CHUNKS (FOR MULTI-PAGE) --------
    chunk_size = 30   # rows per page

    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]

        table = Table(chunk, repeatRows=1)

        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),

            ('GRID', (0,0), (-1,-1), 0.5, colors.black),

            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),  # smaller text

            ('ALIGN', (0,0), (-1,-1), 'CENTER'),

            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ]))

        elements.append(table)
        elements.append(PageBreak())  # new page
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib import colors, pagesizes
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from io import BytesIO

    # ==============================
    # PDF GENERATION (FIXED WIDTH & LANDSCAPE)
    # ==============================
    # ==============================
        # OUTPUT (Find this section in your current code)
        # ==============================
    st.subheader("📄 Final Output")

        # This line MUST come before the PDF code
    final = rfm[['CustomerID','Recency','Frequency','Monetary',
                    'Cluster','Persona','Marketing_Strategy',
                    'Top_Product','Churn_Status']]

    st.dataframe(final.head())

        # ==============================
        # 📄 NEW PDF GENERATION CODE (PASTE HERE)
        # ==============================
    from reportlab.lib import pagesizes # Add this import if missing
        
    pdf_buffer = BytesIO()

        # 1. Set to LANDSCAPE for more width
    doc = SimpleDocTemplate(
            pdf_buffer, 
            pagesize=pagesizes.landscape(pagesizes.A4),
            rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30
        )
    styles = getSampleStyleSheet()
    elements = []

        # Title
    elements.append(Paragraph("Customer Segmentation & Business Intelligence Report", styles['Title']))
    elements.append(Spacer(1, 12))

        # -------- SELECT DATA --------
    pdf_data = final.copy()   

        # Convert to list for ReportLab
    data = [pdf_data.columns.tolist()] + pdf_data.values.tolist()

        # 2. CALCULATE COLUMN WIDTHS
    total_width = 780
    col_widths = [
            total_width * 0.08,  # CustomerID
            total_width * 0.07,  # Recency
            total_width * 0.07,  # Frequency
            total_width * 0.08,  # Monetary
            total_width * 0.05,  # Cluster
            total_width * 0.10,  # Persona
            total_width * 0.30,  # Marketing_Strategy
            total_width * 0.15,  # Top_Product
            total_width * 0.10,  # Churn_Status
        ]

        # -------- SPLIT INTO CHUNKS --------
    chunk_size = 22   

    for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            table = Table(chunk, colWidths=col_widths, repeatRows=1)

            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#333333")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 7),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('BACKGROUND', (0,1), (-1,-1), colors.white),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ]))

            elements.append(table)
            elements.append(PageBreak())

        # Build PDF
    doc.build(elements)

        # Download button
    st.download_button(
            "⬇ Download Full PDF Report",
            pdf_buffer.getvalue(),
            "customer_intelligence_report.pdf"
        )