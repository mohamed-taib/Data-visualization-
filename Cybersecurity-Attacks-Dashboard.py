import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset
file_path = 'c:/Users/Tsieb/Desktop/M1 AI/datavis/cybersecurity_attacks.csv'
data = pd.read_csv(file_path)

# User Guidance Overlays using Streamlit Expander (Simulated Overlays)
st.sidebar.title("User Guidance Overlays")

# Step-by-step guidance
with st.sidebar.expander("Getting Started"):
    st.write("Welcome to the Cybersecurity Attacks Dashboard!")
    st.write("1. **Data Cleaning**: Handle missing values and remove duplicates in the sidebar options.")
    st.write("2. **Filters**: Apply filters for attack types, protocols, and severity levels to customize the dataset.")
    st.write("3. **Visualization Sections**: Explore the main visualizations on attack types, protocol usage, and correlations.")

with st.sidebar.expander("Data Exploration"):
    st.write("1. **Summary Statistics**: Review numerical summaries of the dataset.")
    st.write("2. **Custom Visualizations**: Use the customizable visualizations section to create your plots.")

with st.sidebar.expander("Advanced Analysis"):
    st.write("1. **Time Series Analysis**: Analyze trends over time if timestamp data is available.")
    st.write("2. **Clustering**: Discover patterns using KMeans clustering for numerical data")

# Data Cleaning
st.sidebar.header("Data Cleaning Options")
st.sidebar.markdown("Clean the dataset by handling missing values or removing duplicates before analysis.")
missing_value_option = st.sidebar.radio("Handle Missing Values", ["Drop Rows", "Fill with Mean", "Fill with Median", "None"])

# Handle missing values
if missing_value_option == "Drop Rows":
    data = data.dropna()
elif missing_value_option == "Fill with Mean":
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        data[col] = data[col].fillna(data[col].mean())
elif missing_value_option == "Fill with Median":
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        data[col] = data[col].fillna(data[col].median())

# Remove duplicates
remove_duplicates = st.sidebar.checkbox("Remove Duplicates")
if remove_duplicates:
    data = data.drop_duplicates()

# Sidebar filters
st.sidebar.title("Filters")
st.sidebar.markdown("Filter the dataset by attack types, protocols, and severity levels.")
attack_types = st.sidebar.multiselect("Select Attack Types", data['Attack Type'].unique(), default=data['Attack Type'].unique())
protocols = st.sidebar.multiselect("Select Protocols", data['Protocol'].unique(), default=data['Protocol'].unique())
severity_levels = st.sidebar.multiselect("Select Severity Levels", data['Severity Level'].unique(), default=data['Severity Level'].unique())

# Filter data
filtered_data = data[
    (data['Attack Type'].isin(attack_types)) &
    (data['Protocol'].isin(protocols)) &
    (data['Severity Level'].isin(severity_levels))
]

# Main dashboard
st.title("Cybersecurity Attacks Dashboard")
st.markdown("Explore the insights and trends in cybersecurity attacks using the visualizations below.")

# Summary Statistics
st.subheader("Summary Statistics")
st.markdown("Overview of numerical data in the dataset. This provides insights into the range, mean, and variance of numerical attributes.")
st.write(filtered_data.describe())

# Attack Type Distribution
st.subheader("Distribution of Attack Types")
st.markdown("Visualize the frequency of each attack type in the dataset. This helps identify the most common types of attacks.")
attack_count = filtered_data['Attack Type'].value_counts()
fig1 = px.bar(attack_count, x=attack_count.index, y=attack_count.values, labels={'x': 'Attack Type', 'y': 'Count'}, title="Attack Type Distribution")
st.plotly_chart(fig1)

# Protocol Usage
st.subheader("Protocol Usage")
st.markdown("Analyze the distribution of different network protocols used in the attacks. This visualization highlights which protocols are most targeted.")
protocol_count = filtered_data['Protocol'].value_counts()
fig2 = px.bar(protocol_count, x=protocol_count.index, y=protocol_count.values, labels={'x': 'Protocol', 'y': 'Count'}, title="Protocol Usage")
st.plotly_chart(fig2)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
st.markdown("Explore the relationships between numerical features in the dataset. Correlation heatmaps help identify features with strong relationships.")
numerical_data = filtered_data.select_dtypes(include=['float64', 'int64'])
if not numerical_data.empty:
    fig3, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numerical_data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig3)
else:
    st.write("No numerical data available after filtering.")

# Scatter Plot
st.subheader("Scatter Plot: Packet Length vs Anomaly Scores")
st.markdown("Visualize the relationship between packet length and anomaly scores. This can reveal potential patterns or outliers in the data.")
fig4 = px.scatter(filtered_data, x='Packet Length', y='Anomaly Scores', color='Attack Type', title="Packet Length vs Anomaly Scores")
st.plotly_chart(fig4)

# Severity Levels by Attack Types
st.subheader("Severity Levels by Attack Types")
st.markdown("Analyze the distribution of severity levels across different attack types. This shows the risk level associated with each type of attack.")
if not filtered_data.empty:
    fig5 = px.histogram(filtered_data, x="Attack Type", color="Severity Level", barmode="group", title="Severity Levels by Attack Types")
    st.plotly_chart(fig5)
else:
    st.write("No data available after filtering.")

# Box Plot: Packet Length by Severity Level
st.subheader("Packet Length by Severity Level")
st.markdown("Examine the distribution of packet lengths for each severity level. Box plots provide a summary of the data's range and potential outliers.")
if not filtered_data.empty:
    fig6 = px.box(filtered_data, x="Severity Level", y="Packet Length", color="Severity Level", title="Packet Length by Severity Level")
    st.plotly_chart(fig6)
else:
    st.write("No data available after filtering.")

# Time Series Analysis (if Timestamp exists)
st.subheader("Time Series Analysis")
st.markdown("Visualize the trends in attacks over time, if timestamp data is available. This can help identify seasonal patterns or anomalies.")
if 'Timestamp' in filtered_data.columns:
    filtered_data['Timestamp'] = pd.to_datetime(filtered_data['Timestamp'])
    time_data = filtered_data.groupby(pd.Grouper(key='Timestamp', freq='D')).size().reset_index(name='Count')
    fig7 = px.line(time_data, x='Timestamp', y='Count', title="Attack Trends Over Time")
    st.plotly_chart(fig7)

    # Advanced Time Series Analysis
    st.subheader("Seasonal Decomposition of Time Series")
    st.markdown("Analyze trends, seasonality, and residuals in attack data over time. Decomposition helps in understanding components of time series.")
    decomposition = seasonal_decompose(time_data['Count'], model='additive', period=7)
    st.line_chart(decomposition.trend)
    st.line_chart(decomposition.seasonal)
    st.line_chart(decomposition.resid)
else:
    st.write("Timestamp column not available for time series analysis.")

# Clustering Analysis
st.subheader("Clustering Analysis")
st.markdown("Group similar attack patterns using clustering techniques. Clustering helps uncover hidden patterns in the data.")
if not numerical_data.empty:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numerical_data)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    filtered_data['Cluster'] = clusters
    fig8 = px.scatter(filtered_data, x=numerical_data.columns[0], y=numerical_data.columns[1], color='Cluster', title="Cluster Analysis")
    st.plotly_chart(fig8)
else:
    st.write("Not enough numerical data for clustering.")

# Protocol vs. Attack Type Heatmap
st.subheader("Protocol vs. Attack Type Heatmap")
st.markdown("Analyze the relationship between protocols and attack types. This heatmap highlights intersections of key features.")
protocol_attack = pd.crosstab(filtered_data['Protocol'], filtered_data['Attack Type'])
fig9, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(protocol_attack, annot=True, cmap="Blues", fmt="d", ax=ax)
st.pyplot(fig9)

# Severity Levels Over Time
st.subheader("Severity Levels Over Time")
st.markdown("Analyze severity levels over time. This visualization reveals changes in attack severity across periods.")
if 'Timestamp' in filtered_data.columns:
    filtered_data['Timestamp'] = pd.to_datetime(filtered_data['Timestamp'])
    severity_trend = filtered_data.groupby(['Timestamp', 'Severity Level']).size().reset_index(name='Count')
    fig10 = px.line(severity_trend, x='Timestamp', y='Count', color='Severity Level', title="Severity Levels Over Time")
    st.plotly_chart(fig10)

# Payload Data Distribution
st.subheader("Payload Data Distribution")
st.markdown("Visualize the size of payload data. Histogram reveals data distribution patterns.")
if 'Payload Data' in filtered_data.columns:
    payload_lengths = filtered_data['Payload Data'].str.len().dropna()
    fig11, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(payload_lengths, kde=True, bins=30, ax=ax)
    ax.set_title("Payload Data Size Distribution")
    st.pyplot(fig11)
else:
    st.write("No payload data available for visualization.")

# Advanced Clustering Visualization (3D)
st.subheader("3D Clustering Visualization")
st.markdown("Visualize clustering in three dimensions for numerical data. 3D visualizations enhance pattern recognition.")
if numerical_data.shape[1] >= 3:
    fig12 = px.scatter_3d(filtered_data, x=numerical_data.columns[0], y=numerical_data.columns[1],
                          z=numerical_data.columns[2], color='Cluster', title="3D Cluster Analysis")
    st.plotly_chart(fig12)

# Alerts vs. Severity
st.subheader("Alerts/Warnings by Severity Level")
st.markdown("Compare alerts/warnings by severity levels. This identifies the intensity of warnings across severities.")
if 'Alerts/Warnings' in filtered_data.columns:
    alerts_severity = filtered_data.groupby('Severity Level')['Alerts/Warnings'].sum().reset_index()
    fig13 = px.bar(alerts_severity, x='Severity Level', y='Alerts/Warnings', title="Alerts/Warnings by Severity Level")
    st.plotly_chart(fig13)

# Interactive Table for Filtered Data
st.subheader("Filtered Data Table")
st.markdown("View the filtered data in tabular form for further inspection.")
st.dataframe(filtered_data)
