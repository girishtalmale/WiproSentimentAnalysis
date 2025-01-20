# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:57:22 2024

@author: GHRCE
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="College Survey Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #2c3e50;
        font-size: 2.5rem !important;
        padding-bottom: 2rem;
        text-align: center;
        font-weight: bold;
    }
    .stHeader {
        color: #34495e;
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load the datasets
df = pd.read_csv('data/processed_survey.csv')
results = pd.read_csv('data/model_resultsmain.csv')

# Dashboard Title
st.title("College Survey Sentiment Analysis Dashboard")

# Create three columns for key metrics
col1, col2, col3 = st.columns(3)

# Calculate metrics
total_responses = len(df)
happy_count = df['sentiment'].value_counts().get('happy', 0)
unhappy_count = df['sentiment'].value_counts().get('unhappy', 0)
satisfaction_rate = (happy_count / total_responses * 100) if total_responses > 0 else 0

# Display metrics in cards
with col1:
    st.markdown("""
        <div class="metric-card">
            <h3 style='text-align: center; color: #2c3e50;'>Total Responses</h3>
            <h2 style='text-align: center; color: #3498db;'>{}</h2>
        </div>
    """.format(total_responses), unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="metric-card">
            <h3 style='text-align: center; color: #2c3e50;'>Satisfaction Rate</h3>
            <h2 style='text-align: center; color: #27ae60;'>{:.1f}%</h2>
        </div>
    """.format(satisfaction_rate), unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="metric-card">
            <h3 style='text-align: center; color: #2c3e50;'>Response Ratio</h3>
            <h2 style='text-align: center; color: #e74c3c;'>{:.1f}:1</h2>
        </div>
    """.format(happy_count/unhappy_count if unhappy_count > 0 else float('inf')), unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["Overall Sentiment", "Department Analysis", "Facility Analysis", "Model Performance"])

with tab1:
    st.header("Overall Sentiment Distribution")
    
    # Create a more attractive pie chart using plotly
    fig = px.pie(
        values=df['sentiment'].value_counts(),
        names=df['sentiment'].value_counts().index,
        title='Sentiment Distribution',
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.4
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    if unhappy_count > happy_count:
        st.warning("⚠️ Concern: The number of 'Unhappy' responses exceeds the number of 'Happy' responses!")
    elif happy_count > 0 and unhappy_count == 0:
        st.success("😊 Great News: There are only 'Happy' responses with no 'Unhappy' responses!")
    elif happy_count == 0 and unhappy_count > 0:
        st.warning("⚠️ Alert: No 'Happy' responses were recorded, but there are 'Unhappy' responses.")

with tab2:
    st.header("Department-wise Sentiment Analysis")
    
    # Create a more sophisticated department analysis using plotly
    dept_sentiment = df.groupby(['department', 'sentiment']).size().unstack(fill_value=0)
    fig = go.Figure()
    
    for sentiment in dept_sentiment.columns:
        fig.add_trace(go.Bar(
            name=sentiment,
            x=dept_sentiment.index,
            y=dept_sentiment[sentiment],
            text=dept_sentiment[sentiment],
            textposition='auto',
        ))
    
    fig.update_layout(
        barmode='group',
        title='Department-wise Sentiment Distribution',
        xaxis_title='Department',
        yaxis_title='Count',
        legend_title='Sentiment',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    # Group by department and sentiment, then count occurrences
    department_sentiment_counts = df.groupby(['department', 'sentiment']).size().unstack(fill_value=0)
    # Check for departments where 'Unhappy' count exceeds 'Happy' count
    concern_departments = department_sentiment_counts[
        (department_sentiment_counts.get('unhappy', 0) > department_sentiment_counts.get('happy', 0))
    ].index.tolist()

    # Display the list of concerning departments
    if concern_departments:
        st.warning("⚠️ Concern: The following departments have more 'Unhappy' responses than 'Happy':")
        for department in concern_departments:
            st.write(f"- {department}")
    else:
        st.success("😊 Great News: No department has more 'Unhappy' responses than 'Happy'.")

with tab3:
    st.header("Facility-wise Sentiment Analysis")
    
    # Create an interactive facility analysis
    facility_sentiment = df.groupby(['category', 'sentiment']).size().unstack(fill_value=0)
    fig = px.bar(
        facility_sentiment,
        barmode='group',
        title='Facility-wise Sentiment Analysis',
        labels={'value': 'Count', 'category': 'Facility'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    # Group by category (facility) and sentiment, then count occurrences
    facility_sentiment_counts = df.groupby(['category', 'sentiment']).size().unstack(fill_value=0)
    # Check for facilities where 'Unhappy' count exceeds 'Happy' count
    concern_facilities = facility_sentiment_counts[
        (facility_sentiment_counts.get('unhappy', 0) > facility_sentiment_counts.get('happy', 0))
    ].index.tolist()

    # Display the list of concerning facilities, each on a new line
    if concern_facilities:
        st.warning("⚠️ Concern: The following facilities have more 'Unhappy' responses than 'Happy':")
        for facility in concern_facilities:
            st.write(f"- {facility}")
    else:
        st.success("😊 Great News: No facility has more 'Unhappy' responses than 'Happy'.")

with tab4:
    st.header("Model Performance Comparison")
    
    # Create an interactive model performance visualization
    fig = go.Figure()
    metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    colors = px.colors.qualitative.Set3[:len(metrics)]
    
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric,
            x=results['Model'],
            y=results[metric],
            marker_color=colors[i],
            text=results[metric].round(3),
            textposition='auto',
        ))
    
    fig.update_layout(
        barmode='group',
        title='Model Performance Metrics Comparison',
        xaxis_title='Model',
        yaxis_title='Score',
        legend_title='Metrics',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# Add model results table with styling
st.markdown("### Detailed Model Performance Metrics")
st.dataframe(
    results.style.background_gradient(cmap='Blues', subset=['Accuracy', 'F1-Score', 'Precision', 'Recall'])
)