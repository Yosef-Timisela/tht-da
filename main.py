import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import squarify

# Config
st.set_page_config(page_title="Food Delivery Estimation", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel("df_da_final.xlsx")

    # Convert to numeric
    numeric_cols = ['Preparation_Time_min', 'Delivery_Time_min', 'Actual_Travel_Time', 'Distance_km', 'Courier_Experience_yrs']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=numeric_cols, inplace=True)

    return df

df = load_data()

# Sidebar filters
st.sidebar.title("📋 Filters")

weather_options = st.sidebar.multiselect("Weather", options=df['Weather'].unique(), default=df['Weather'].unique())
traffic_options = st.sidebar.multiselect("Traffic Level", options=df['Traffic_Level'].unique(), default=df['Traffic_Level'].unique())
time_options = st.sidebar.multiselect("Time of Day", options=df['Time_of_Day'].unique(), default=df['Time_of_Day'].unique())
vehicle_options = st.sidebar.multiselect("Vehicle Type", options=df['Vehicle_Type'].unique(), default=df['Vehicle_Type'].unique())

# Distance slider
min_distance = float(df['Distance_km'].min())
max_distance = float(df['Distance_km'].max())

distance_range = st.sidebar.slider(
    "Distance (KM) Range",
    min_value=round(min_distance, 1),
    max_value=round(max_distance, 1),
    value=(round(min_distance, 1), round(max_distance, 1)),
    step=0.1
)

# Apply filters
filtered_df = df[
    (df['Weather'].isin(weather_options)) &
    (df['Traffic_Level'].isin(traffic_options)) &
    (df['Time_of_Day'].isin(time_options)) &
    (df['Vehicle_Type'].isin(vehicle_options)) &
    (df['Distance_km'] >= distance_range[0]) &
    (df['Distance_km'] <= distance_range[1])
]

# KPI Cards
st.title("🚚 Food Delivery Estimation Dashboard")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Number of Orders", f"{len(filtered_df)}")
col2.metric("Avg Prep. Time (Min)", f"{filtered_df['Preparation_Time_min'].mean():.2f}")
col3.metric("Avg Delivery Time (Min)", f"{filtered_df['Delivery_Time_min'].mean():.2f}")
col4.metric("Avg Travel Time (Min)", f"{filtered_df['Actual_Travel_Time'].mean():.2f}")
col5.metric("Avg Distance (KM)", f"{filtered_df['Distance_km'].mean():.2f}")

# Donut Charts Row
st.markdown("## 🍩 Avg Delivery Time Breakdown")

donut1, donut2, donut3 = st.columns(3)

# 1. Donut: Avg Delivery Time by Weather
weather_avg = filtered_df.groupby("Weather")["Delivery_Time_min"].mean().reset_index()
fig1 = go.Figure(data=[go.Pie(labels=weather_avg["Weather"], values=weather_avg["Delivery_Time_min"], hole=.5)])
fig1.update_layout(title="By Weather", margin=dict(t=40, b=0, l=0, r=0))
donut1.plotly_chart(fig1, use_container_width=True)

# 2. Donut: Avg Delivery Time by Traffic Level
traffic_avg = filtered_df.groupby("Traffic_Level")["Delivery_Time_min"].mean().reset_index()
fig2 = go.Figure(data=[go.Pie(labels=traffic_avg["Traffic_Level"], values=traffic_avg["Delivery_Time_min"], hole=.5)])
fig2.update_layout(title="By Traffic Level", margin=dict(t=40, b=0, l=0, r=0))
donut2.plotly_chart(fig2, use_container_width=True)

# 3. Donut: Avg Delivery Time by Vehicle Type
vehicle_avg = filtered_df.groupby("Vehicle_Type")["Delivery_Time_min"].mean().reset_index()
fig3 = go.Figure(data=[go.Pie(labels=vehicle_avg["Vehicle_Type"], values=vehicle_avg["Delivery_Time_min"], hole=.5)])
fig3.update_layout(title="By Vehicle Type", margin=dict(t=40, b=0, l=0, r=0))
donut3.plotly_chart(fig3, use_container_width=True)

# Scatter + Treemap Row
st.markdown("## 📍 Delivery Time Details")

scatter_col, treemap_col = st.columns(2)

# 4. Scatter Plot: Distance vs Delivery Time per Delivery (size = Courier Exp)
with scatter_col:
    fig4, ax4 = plt.subplots(figsize=(8, 6))

    # Create scatterplot
    sns.scatterplot(
        data=filtered_df,
        x="Distance_km",
        y="Delivery_Time_min",
        hue="Vehicle_Type",
        size="Courier_Experience_yrs",
        alpha=0.7,
        ax=ax4,
        sizes=(20, 300),
        legend="brief"
    )

    # Filter out numeric labels from legend
    handles, labels = ax4.get_legend_handles_labels()
    new_handles = []
    new_labels = []

    for h, l in zip(handles, labels):
        # keep only non-numeric labels (i.e., Vehicle Types)
        if not l.replace('.', '', 1).isdigit():
            new_handles.append(h)
            new_labels.append(l)

    ax4.legend(new_handles, new_labels, title="Vehicle Type", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Set axis labels and title
    ax4.set_title("Distance vs Delivery Time per Delivery\n(Size = Courier Exp, Color = Vehicle Type)")
    ax4.set_xlabel("Distance (KM)")
    ax4.set_ylabel("Delivery Time (Min)")

    plt.tight_layout()
    st.pyplot(fig4)

# 5. Treemap
with treemap_col:
    tree_df = filtered_df.groupby(["Time_of_Day", "Traffic_Level"])["Delivery_Time_min"].mean().reset_index()
    tree_df["value"] = tree_df["Delivery_Time_min"]
    tree_df["label"] = tree_df.apply(lambda row: f"{row['Time_of_Day']} - {row['Traffic_Level']}\n{row['value']:.1f} min", axis=1)

    fig5 = plt.figure(figsize=(8, 6))
    squarify.plot(sizes=tree_df["value"], label=tree_df["label"], alpha=.8)
    plt.axis('off')
    plt.title("Delivery Time by Time of Day & Traffic Level")
    st.pyplot(fig5)