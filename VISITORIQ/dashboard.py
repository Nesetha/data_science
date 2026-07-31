import streamlit as st
import sqlite3
import pandas as pd
import os

st.set_page_config(
    page_title="VisitorIQ Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 VisitorIQ Dashboard")
st.markdown("### Real-Time Visitor Analytics & Re-Identification System")


# Database Connection
conn = sqlite3.connect("visitors.db")

df = pd.read_sql_query(
    "SELECT * FROM visitors",
    conn
)

if df.empty:
    st.warning("No visitor data found.")
    st.stop()

# Snapshot Count

snapshot_folder = "data/snapshots"

if os.path.exists(snapshot_folder):
    snapshot_files = [
        f for f in os.listdir(snapshot_folder)
        if f.endswith(".jpg")
    ]
else:
    snapshot_files = []

# Metrics

unique_visitors = df["visitor_id"].nunique()
total_records = len(df)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "👤 Unique Visitors",
        unique_visitors
    )

with col2:
    st.metric(
        "🗄️ Database Records",
        total_records
    )

with col3:
    st.metric(
        "📸 Snapshots Stored",
        len(snapshot_files)
    )

st.divider()

# Visitor Records

st.subheader("📋 Visitor Records")

st.dataframe(
    df,
    use_container_width=True
)

# Recent Visitors

st.subheader("🕒 Recent Visitors")

st.dataframe(
    df.tail(10),
    use_container_width=True
)

st.divider()

# Analytics Charts

col4, col5 = st.columns(2)

with col4:
    st.subheader("🎨 Color Distribution")

    color_counts = df["color"].value_counts()

    st.bar_chart(color_counts)

with col5:
    st.subheader("🏋️ Build Distribution")

    build_counts = df["build"].value_counts()

    st.bar_chart(build_counts)

# Height Statistics


st.divider()

st.subheader("📏 Height Statistics")

st.dataframe(
    df["height"].describe().to_frame()
)

# DOWNLOAD BUTTON

st.download_button(
    label="📥 Download Visitor Data",
    data=df.to_csv(index=False),
    file_name="visitor_records.csv",
    mime="text/csv"
)

# Visitor Snapshot Gallery

st.divider()

st.subheader("📸 Visitor Snapshot Gallery")

if len(snapshot_files) == 0:

    st.info("No visitor snapshots found.")

else:

    cols = st.columns(4)

    for idx, image_file in enumerate(snapshot_files):

        image_path = os.path.join(
            snapshot_folder,
            image_file
        )

        with cols[idx % 4]:

            st.image(
                image_path,
                caption=image_file,
                use_container_width=True
            )