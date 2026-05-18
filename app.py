import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import warnings

# =========================================================
# REMOVE WARNINGS
# =========================================================
warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="FreshTrack AI",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #f5f7fa;
}

/* Section Cards */
.section {
    background-color: white;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

/* KPI Cards */
[data-testid="stMetric"] {
    background-color: white;
    border-left: 4px solid #2563eb;
    padding: 12px;
    border-radius: 12px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

/* Buttons */
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    border: none;
    height: 42px;
    width: 100%;
}

.stButton > button:hover {
    background-color: #1d4ed8;
}

/* Recommendation Cards */
.equal-card {
    height: 155px;

    padding: 14px;
    border-radius: 14px;

    color: #111827;
    font-size: 13px;
    line-height: 1.5;

    box-shadow: 0 2px 8px rgba(0,0,0,0.05);

    display: flex;
    flex-direction: column;

    overflow: hidden;

    transition: all 0.3s ease;
}

.equal-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 14px rgba(0,0,0,0.08);
}

.equal-card h4 {
    margin-bottom: 6px;
    font-size: 15px;
}

.equal-card p {
    margin: 0;
}

.equal-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 18px rgba(0,0,0,0.08);
}

.equal-card h4 {
    margin-bottom: 8px;
    font-size: 16px;
}

.equal-card p {
    margin: 0;
}

.equal-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 18px rgba(0,0,0,0.08);
}

.equal-card h4 {
    margin-bottom: 8px;
    font-size: 17px;
}

.equal-card p {
    margin: 0;
}

.red-card {
    background-color: #fee2e2;
    border-left: 6px solid #ef4444;
}

.blue-card {
    background-color: #dbeafe;
    border-left: 6px solid #2563eb;
}

.orange-card {
    background-color: #ffedd5;
    border-left: 6px solid #f59e0b;
}

.green-card {
    background-color: #dcfce7;
    border-left: 6px solid #10b981;
}

</style>
""", unsafe_allow_html=True)

brand_color = "#2563eb"
colors = px.colors.qualitative.Set2

# =========================================================
# SIDEBAR LOGO
# =========================================================
st.sidebar.image(
    "miracle_logo.png",
    width=250
)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("FreshTrack AI")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "ML Prediction", "Recommendations"]
)

# =========================================================
# DASHBOARD PAGE
# =========================================================
if page == "Dashboard":

    st.title("FreshTrack AI Dashboard")

    file = st.file_uploader(
        "Upload Dataset",
        type=["csv"],
        key="dashboard_upload"
    )

    if file:

        df = pd.read_csv(file)
        df['date'] = pd.to_datetime(df['date'])

        # =================================================
        # FILTERS
        # =================================================
        st.markdown('<div class="section">', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        product = c1.selectbox(
            "Select Product",
            ["All"] + list(df['product_name'].unique())
        )

        location = c2.selectbox(
            "Select Location",
            ["All"] + list(df['location'].unique())
        )

        date_range = c3.date_input(
            "Date Range",
            [df['date'].min(), df['date'].max()]
        )

        if product != "All":
            df = df[df['product_name'] == product]

        if location != "All":
            df = df[df['location'] == location]

        df = df[
            (df['date'] >= pd.to_datetime(date_range[0])) &
            (df['date'] <= pd.to_datetime(date_range[1]))
        ]

        st.markdown('</div>', unsafe_allow_html=True)

        # =================================================
        # KPI METRICS
        # =================================================
        st.markdown('<div class="section">', unsafe_allow_html=True)

        total_ordered = df['quantity_ordered_kg'].sum()
        total_used = df['quantity_used_kg'].sum()
        total_waste = df['quantity_wasted_kg'].sum()
        total_loss = df['loss_amount_inr'].sum()

        waste_pct = (
            (total_waste / total_ordered) * 100
            if total_ordered else 0
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Ordered", f"{total_ordered:,.0f} KG")
        c2.metric("Used", f"{total_used:,.0f} KG")
        c3.metric("Waste %", f"{waste_pct:.2f}%")
        c4.metric("Loss", f"₹ {total_loss:,.0f}")

        st.markdown('</div>', unsafe_allow_html=True)

        # =================================================
        # CHARTS
        # =================================================
        st.markdown('<div class="section">', unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:

            trend = (
                df.groupby('date')['quantity_wasted_kg']
                .sum()
                .reset_index()
            )

            fig = px.line(
                trend,
                x='date',
                y='quantity_wasted_kg',
                title="Waste Trend Over Time",
                color_discrete_sequence=[brand_color]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with c2:

            fig = px.scatter(
                df,
                x='temperature_celsius',
                y='quantity_wasted_kg',
                title="Temperature vs Waste",
                color_discrete_sequence=[brand_color]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Upload dataset to begin.")

# =========================================================
# ML PREDICTION PAGE
# =========================================================
elif page == "ML Prediction":

    st.title("ML Prediction Center")

    demand_model = joblib.load("demand_model.pkl")
    spoil_model = joblib.load("spoilage_model.pkl")
    encoders = joblib.load("encoders.pkl")

    def safe_encode(enc, val):

        return (
            enc.transform([val])[0]
            if val in enc.classes_
            else 0
        )

    st.markdown('<div class="section">', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:

       # =========================================================
# CATEGORY -> PRODUCT MAPPING
# =========================================================

category_product_map = {

    "Dairy": [
        "Milk",
        "Cheese",
        "Butter",
        "Curd",
        "Paneer",
        "Yogurt",
        "Cream"
    ],

    "Frozen": [
        "Frozen Peas",
        "Ice Cream"
    ],

    "Meat": [
        "Chicken",
        "Fish"
    ],

    "Beverages": [
        "Juice"
    ]
}

# =========================================================
# CATEGORY DROPDOWN
# =========================================================

category = st.selectbox(
    "Category",
    sorted(category_product_map.keys())
)

# =========================================================
# PRODUCT DROPDOWN FILTERED BY CATEGORY
# =========================================================

filtered_products = category_product_map[category]

product = st.selectbox(
    "Product",
    filtered_products
)

    with c2:

        quantity = st.number_input(
            "Quantity Ordered (KG)",
            10,
            5000,
            100
        )

        temperature = st.slider(
            "Temperature (°C)",
            0,
            50,
            25
        )

        humidity = st.slider(
            "Humidity (%)",
            10,
            100,
            60
        )

    with c3:

        unit_cost = st.number_input(
            "Unit Cost (₹)",
            1,
            1000,
            50
        )

        storage_capacity = st.number_input(
            "Storage Capacity (KG)",
            100,
            10000,
            2000
        )

        shelf_life = st.number_input(
            "Shelf Life (Days)",
            1,
            60,
            7
        )

    today = pd.Timestamp.today()

    input_data = pd.DataFrame([{

        "warehouse_id": 1,

        "product_id":
            safe_encode(
                encoders['product_name'],
                product
            ),

        "category":
            safe_encode(
                encoders['category'],
                category
            ),

        "supplier_id": 101,

        "quantity_ordered_kg": quantity,

        "unit_cost_inr": unit_cost,

        "shelf_life_days": shelf_life,

        "temperature_celsius": temperature,

        "humidity_percent": humidity,

        "storage_capacity_kg": storage_capacity,

        "day_of_week": today.weekday(),

        "month": today.month
    }])

    input_data = input_data[[
        'warehouse_id',
        'product_id',
        'category',
        'supplier_id',
        'quantity_ordered_kg',
        'unit_cost_inr',
        'shelf_life_days',
        'temperature_celsius',
        'humidity_percent',
        'storage_capacity_kg',
        'day_of_week',
        'month'
    ]]

    if st.button("Predict"):

        demand = demand_model.predict(input_data)[0]
        spoil = spoil_model.predict(input_data)[0]

        st.success(
            f"Predicted Demand: {int(demand)} KG"
        )

        if spoil == 1:
            st.error("⚠️ High Spoilage Risk")
        else:
            st.success("✅ Low Spoilage Risk")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# RECOMMENDATIONS PAGE
# =========================================================
elif page == "Recommendations":

    st.title("AI Recommendation Center")

    file = st.file_uploader(
        "Upload Dataset",
        type=["csv"],
        key="recommendation_upload"
    )

    if file:

        df = pd.read_csv(file)
        df['date'] = pd.to_datetime(df['date'])

        total_ordered = df['quantity_ordered_kg'].sum()
        total_waste = df['quantity_wasted_kg'].sum()
        total_loss = df['loss_amount_inr'].sum()

        waste_pct = (
            (total_waste / total_ordered) * 100
            if total_ordered else 0
        )

        top_product = (
            df.groupby('product_name')['quantity_wasted_kg']
            .sum()
            .idxmax()
        )

        top_location = (
            df.groupby('location')['loss_amount_inr']
            .sum()
            .idxmax()
        )

        avg_temp = df['temperature_celsius'].mean()

        # =================================================
        # KPI METRICS
        # =================================================
        st.markdown('<div class="section">', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Waste %", f"{waste_pct:.2f}%")
        c2.metric("Total Waste", f"{total_waste:,.0f} KG")
        c3.metric("Financial Loss", f"₹ {total_loss:,.0f}")
        c4.metric("Avg Temp", f"{avg_temp:.1f}°C")

        st.markdown('</div>', unsafe_allow_html=True)

        # =================================================
        # STRATEGIC RECOMMENDATIONS
        # =================================================
        st.subheader("Strategic Recommendations")

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.markdown(f"""
            <div class="equal-card red-card">
                <h4>🔴 High Waste Product</h4>
                <p>
                Reduce procurement quantity for
                <b>{top_product}</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with c2:

            st.markdown(f"""
            <div class="equal-card blue-card">
                <h4>🔵 Storage Optimization</h4>
                <p>
                Improve storage conditions in
                <b>{top_location}</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with c3:

            temp_msg = (
                "Deploy cooling optimization."
                if avg_temp > 30
                else
                "Temperature is stable."
            )

            st.markdown(f"""
            <div class="equal-card orange-card">
                <h4>🟠 Temperature Risk</h4>
                <p>{temp_msg}</p>
            </div>
            """, unsafe_allow_html=True)

        with c4:

            st.markdown("""
            <div class="equal-card green-card">
                <h4>🟢 Inventory Rotation</h4>
                <p>
                Prioritize low shelf-life inventory.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # =================================================
        # CHARTS
        # =================================================
        st.markdown('<div class="section">', unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:

            waste_products = (
                df.groupby('product_name')['quantity_wasted_kg']
                .sum()
                .sort_values(ascending=False)
                .head(5)
                .reset_index()
            )

            fig = px.scatter(
                waste_products,
                x='product_name',
                y='quantity_wasted_kg',
                size='quantity_wasted_kg',
                title="Top Waste Products",
                color='quantity_wasted_kg',
                color_continuous_scale='Blues',
                height=400
            )

            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white'
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with c2:

            loss_loc = (
                df.groupby('location')['loss_amount_inr']
                .sum()
                .reset_index()
            )

            fig2 = px.pie(
                loss_loc,
                values='loss_amount_inr',
                names='location',
                title="Loss Distribution by Location",
                hole=0.45,
                color_discrete_sequence=colors
            )

            fig2.update_layout(
                height=400,
                paper_bgcolor='white'
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

        # =================================================
        # ACTION PLAN TABLE
        # =================================================
        st.markdown('<div class="section">', unsafe_allow_html=True)

        st.subheader("Priority Action Plan")

        actions = pd.DataFrame({

            "Priority": [
                "High",
                "Medium",
                "Medium",
                "Low"
            ],

            "Area": [
                "Inventory Planning",
                "Warehouse Cooling",
                "Storage Optimization",
                "Demand Forecasting"
            ],

            "Recommendation": [
                f"Reduce ordering for {top_product}",
                "Monitor warehouse temperature",
                f"Improve storage in {top_location}",
                "Use ML prediction before procurement"
            ]
        })

        st.dataframe(
            actions,
            use_container_width=True,
            hide_index=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Upload dataset to generate recommendations.")
