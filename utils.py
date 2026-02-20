import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import json
from datetime import datetime

class TextPreprocessor:
    """Simple text preprocessing for bug reports"""
    
    def __init__(self):
        pass
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess(self, text):
        """Main preprocessing function"""
        return self.clean_text(text)

def create_metrics_display(metrics_dict):
    """Create metrics display for Streamlit"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = metrics_dict.get('accuracy', 0)
        st.metric("Accuracy", f"{accuracy:.2%}")
    
    with col2:
        precision = metrics_dict.get('precision', 0)
        st.metric("Precision", f"{precision:.2%}")
    
    with col3:
        recall = metrics_dict.get('recall', 0)
        st.metric("Recall", f"{recall:.2%}")
    
    with col4:
        f1 = metrics_dict.get('f1', 0)
        st.metric("F1-Score", f"{f1:.2%}")

def plot_confusion_matrix(cm, labels):
    """Plot confusion matrix using plotly"""
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        text_auto=True,
        aspect="auto"
    )
    
    fig.update_layout(
        title="Confusion Matrix",
        width=500,
        height=500,
        xaxis_title="Predicted Label",
        yaxis_title="True Label"
    )
    
    return fig

def plot_roc_curve(fpr, tpr, auc_score, labels):
    """Plot ROC curve"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc_score:.3f})',
        line=dict(color='#2E91E5', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=600,
        height=500,
        showlegend=True
    )
    
    return fig

def plot_feature_importance(feature_names, importance_scores, top_n=20):
    """Plot feature importance"""
    indices = np.argsort(importance_scores)[::-1][:top_n]
    
    fig = go.Figure(data=[
        go.Bar(
            x=importance_scores[indices],
            y=[feature_names[i][:30] + "..." if len(feature_names[i]) > 30 else feature_names[i] for i in indices],
            orientation='h',
            marker=dict(
                color=importance_scores[indices],
                colorscale='Viridis',
                showscale=True
            ),
            text=[f"{importance_scores[i]:.4f}" for i in indices],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=f"Top {top_n} Most Important Features",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=600,
        width=800,
        yaxis=dict(autorange="reversed")
    )
    
    return fig

def create_ensemble_visualization():
    """Create visualization of ensemble model architecture"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Random Forest", "SVM", "Logistic Regression", "XGBoost"),
        specs=[[{'type': 'domain'}, {'type': 'domain'}],
               [{'type': 'domain'}, {'type': 'domain'}]]
    )
    
    models = ['Random Forest', 'SVM', 'Logistic Regression', 'XGBoost']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, (model, color) in enumerate(zip(models, colors)):
        row = i // 2 + 1
        col = i % 2 + 1
        
        fig.add_trace(
            go.Pie(
                labels=['Weight', ''],
                values=[1, 3],
                name=model,
                marker=dict(colors=[color, '#E5ECF6']),
                textinfo='none',
                hole=0.6,
                showlegend=False
            ),
            row=row, col=col
        )
        
        fig.add_annotation(
            text=f"<b>{model}</b>",
            x=(col-1)*0.5 + 0.25,
            y=0.8 - (row-1)*0.4,
            showarrow=False,
            font=dict(size=12),
            xref="paper",
            yref="paper"
        )
    
    fig.update_layout(
        title="Ensemble Model Architecture",
        height=500,
        width=800,
        showlegend=False
    )
    
    return fig

def load_metrics(filepath):
    """Load metrics from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None