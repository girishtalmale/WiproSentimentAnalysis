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
# tab1, tab2, tab3, tab4 = st.tabs(["Overall Sentiment", "Department Analysis", "Facility Analysis", "Model Performance"])
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overall Sentiment", "Department Analysis", "Facility Analysis", "Model Performance", "Statistical Sentiment Analysis"])
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

    # # Add model results table with styling
    st.markdown("### Detailed Model Performance Metrics")
    st.dataframe(
       results.style.background_gradient(cmap='Blues', subset=['Accuracy', 'F1-Score', 'Precision', 'Recall'])
     )
with tab5:
    st.header("Statistical Sentiment Analysis")
    
    # Compute the mean sentiment score
    df['sentiment_score'] = df['sentiment'].map({'happy': 1, 'neutral': 0, 'unhappy': -1})
    mean_sentiment_score = df['sentiment_score'].mean()

    # Display the mean sentiment score
    st.markdown(f"""
        ### Mean Sentiment Score
        <div class="metric-card">
            <h3 style='text-align: center; color: #2c3e50;'>Mean Sentiment Score</h3>
            <h2 style='text-align: center; color: {"#27ae60" if mean_sentiment_score > 0 else "#e74c3c"};'>{mean_sentiment_score:.2f}</h2>
        </div>
    """, unsafe_allow_html=True)

    # Provide interpretation of the mean sentiment score
    if mean_sentiment_score > 0:
        st.success("😊 Positive Trend: The overall sentiment trend is positive.")
    elif mean_sentiment_score < 0:
        st.warning("⚠️ Negative Trend: The overall sentiment trend is negative.")
    else:
        st.info("⚖️ Neutral Trend: The overall sentiment trend is neutral.")
    # Sentiment Frequency Analysis
    st.subheader("Sentiment Frequency Analysis")
    
    # Compute sentiment frequency counts
    sentiment_freq = df['sentiment'].value_counts()
    st.subheader("Sentiment Frequency Counts")
    st.table(sentiment_freq)

    # Plot sentiment frequency distribution
    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram
    sns.histplot(df['sentiment_score'], bins=5, kde=False, ax=ax[0], color='skyblue')
    ax[0].set_title("Histogram of Sentiment Scores")
    ax[0].set_xlabel("Sentiment Score")
    ax[0].set_ylabel("Frequency")

    # Density Plot
    sns.kdeplot(df['sentiment_score'], ax=ax[1], shade=True, color='orange')
    ax[1].set_title("Density Plot of Sentiment Scores")
    ax[1].set_xlabel("Sentiment Score")
    ax[1].set_ylabel("Density")

    st.pyplot(fig)

    # Cumulative Distribution
    st.subheader("Cumulative Distribution")
    cumulative_df = df['sentiment_score'].value_counts(normalize=True).sort_index().cumsum()
    cumulative_df = cumulative_df.reset_index()
    cumulative_df.columns = ['Sentiment Score', 'Cumulative Percentage']

    # Plot cumulative distribution
    fig_cumulative = px.line(
        cumulative_df, 
        x="Sentiment Score", 
        y="Cumulative Percentage", 
        title="Cumulative Distribution of Sentiment Scores",
        markers=True
    )
    fig_cumulative.update_layout(
        xaxis_title="Sentiment Score",
        yaxis_title="Cumulative Percentage",
        yaxis_tickformat=".0%",
        height=500
    )
    st.plotly_chart(fig_cumulative, use_container_width=True)

    # Interpretation and Insights
    st.info("""
        ### Insights:
        - **Mean Sentiment Score** gives an overall trend (positive, neutral, or negative).
        - **Frequency Analysis** shows the count and distribution of each sentiment.
        - **Histogram and Density Plot** visualize the shape of the data (e.g., normal, skewed).
        - **Cumulative Distribution** highlights the percentage of students below each sentiment score.
    """)
    # Sentiment Variability Analysis
    st.subheader("Sentiment Variability Analysis")
    sentiment_scores = df['sentiment_score']

    # Measures of Dispersion
    sentiment_range = sentiment_scores.max() - sentiment_scores.min()
    sentiment_variance = sentiment_scores.var()
    sentiment_std_dev = sentiment_scores.std()
    q1 = sentiment_scores.quantile(0.25)
    q3 = sentiment_scores.quantile(0.75)
    iqr = q3 - q1

    st.markdown(f"""
        **Range:** {sentiment_range:.2f}  
        **Variance:** {sentiment_variance:.2f}  
        **Standard Deviation:** {sentiment_std_dev:.2f}  
        **Interquartile Range (IQR):** {iqr:.2f}
    """)

    # Boxplot for IQR and Outliers
    st.subheader("Boxplot of Sentiment Scores")
    fig_boxplot = px.box(
        df, 
        y='sentiment_score', 
        title="Boxplot of Sentiment Scores (IQR and Outliers)",
        color_discrete_sequence=['#3498db']
    )
    st.plotly_chart(fig_boxplot, use_container_width=True)

    # Interpretation and Insights
    st.info("""
        ### Insights:
        - **Range:** Indicates the spread between the most positive and negative sentiments.
        - **Variance & Standard Deviation:** Quantify how spread out the sentiment scores are.
        - **IQR & Boxplot:** Highlight central tendencies and potential outliers.
    """)
    # ANOVA Analysis
    st.subheader("ANOVA: Sentiment Differences Across Groups")
    if 'department' in df.columns:
        # Perform ANOVA
        from scipy.stats import f_oneway

        groups = [group['sentiment_score'].values for name, group in df.groupby('department')]
        f_stat, p_value = f_oneway(*groups)

        st.markdown(f"""
            **ANOVA Results**  
            - **F-Statistic:** {f_stat:.2f}  
            - **P-Value:** {p_value:.4f}
        """)

        if p_value < 0.05:
            st.success("😊 Significant differences were found in sentiment scores across departments (p < 0.05).")
        else:
            st.info("⚖️ No significant differences were found in sentiment scores across departments (p ≥ 0.05).")

        # Boxplot for department-wise sentiment comparison
        fig_anova = px.box(
            df,
            x='department',
            y='sentiment_score',
            title="Department-wise Sentiment Score Distribution",
            labels={'department': 'Department', 'sentiment_score': 'Sentiment Score'},
            color='department'
        )
        fig_anova.update_layout(height=500)
        st.plotly_chart(fig_anova, use_container_width=True)
    else:
        st.error("The dataset does not contain a 'department' column. Please include department information to perform ANOVA.")

    # Interpretation and Insights
    st.info("""
        ### Insights:
        - **ANOVA:** Evaluates whether there are statistically significant differences in sentiment scores across groups (e.g., departments).
        - **Boxplot:** Visualizes sentiment score distribution for each group, highlighting differences and variability.
        - **P-Value:** A p-value < 0.05 indicates significant group differences, while p ≥ 0.05 suggests no significant differences.
    """)
    # Sentiment Satisfaction Index (SSI)
    st.subheader("Sentiment Satisfaction Index (SSI)")
    st.markdown("The Sentiment Satisfaction Index (SSI) is a weighted score based on the distribution of 'happy', 'neutral', and 'unhappy' responses.")

    # Custom weights for each sentiment category
    weights = {'happy': st.number_input("Weight for 'Happy':", min_value=0.0, max_value=1.0, value=0.6),
               'neutral': st.number_input("Weight for 'Neutral':", min_value=0.0, max_value=1.0, value=0.3),
               'unhappy': st.number_input("Weight for 'Unhappy':", min_value=0.0, max_value=1.0, value=0.1)}

    # Ensure weights sum to 1
    total_weight = sum(weights.values())
    if total_weight != 1.0:
        st.error("⚠️ The weights do not sum to 1. Please adjust the weights.")

    else:
        # Calculate SSI
        sentiment_counts = df['sentiment'].value_counts()
        ssi = sum(sentiment_counts.get(sentiment, 0) * weight for sentiment, weight in weights.items()) / len(df)

        st.markdown(f"""
            **Sentiment Satisfaction Index (SSI):** {ssi:.2f}
        """)
        if ssi > 0.7:
            st.success("😊 High satisfaction among students.")
        elif 0.4 <= ssi <= 0.7:
            st.warning("⚖️ Moderate satisfaction among students.")
        else:
            st.error("⚠️ Low satisfaction among students.")

        # Visualization of Weighted Sentiments
        st.subheader("SSI Weighted Sentiment Distribution")
        weighted_sentiments = {k: v * sentiment_counts.get(k, 0) for k, v in weights.items()}
        fig_ssi = px.bar(
            x=list(weighted_sentiments.keys()),
            y=list(weighted_sentiments.values()),
            title="Weighted Sentiment Contribution to SSI",
            labels={'x': 'Sentiment', 'y': 'Weighted Contribution'},
            color=list(weighted_sentiments.keys()),
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_ssi.update_layout(height=400)
        st.plotly_chart(fig_ssi, use_container_width=True)

    # Interpretation and Insights
    st.info("""
        ### Insights:
        - **SSI:** Aggregates sentiment data into a single, weighted index to reflect overall satisfaction.
        - **Weight Customization:** Adjust weights to prioritize specific sentiment categories based on institutional goals.
        - **Visualization:** Understand how each sentiment category contributes to the overall index.
    """)