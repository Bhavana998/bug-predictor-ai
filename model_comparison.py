"""
Model Comparison and Visualization
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class ModelComparator:
    """Compare multiple models side by side"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def add_model(self, name, model, X_test, y_test):
        """Add model to comparison"""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        self.results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'predictions': y_pred,
            'probabilities': y_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return self.results[name]
    
    def plot_comparison_dashboard(self):
        """Create comprehensive comparison dashboard"""
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Model Accuracy', 'Model Precision', 'Model Recall',
                'F1 Scores', 'Confusion Matrix - RF', 'Confusion Matrix - SVM',
                'Confusion Matrix - LR', 'Confusion Matrix - XGB', 'Ensemble Performance'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}],
                [{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'scatter'}]
            ]
        )
        
        # Metrics comparison
        names = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFE194']
        
        for i, metric in enumerate(metrics[:3]):  # First row
            values = [self.results[name][metric] for name in names]
            fig.add_trace(
                go.Bar(x=names, y=values, name=metric.capitalize(),
                      marker_color=colors[i % len(colors)],
                      text=[f"{v:.2%}" for v in values],
                      textposition='outside'),
                row=1, col=i+1
            )
        
        # Confusion matrices
        model_order = ['Random Forest', 'SVM', 'Logistic Regression', 'XGBoost']
        for i, name in enumerate(model_order):
            if name in self.results:
                cm = self.results[name]['confusion_matrix']
                row = 2 if i < 2 else 3
                col = (i % 2) + 1
                
                fig.add_trace(
                    go.Heatmap(
                        z=cm,
                        x=['Bug', 'Feature'],
                        y=['Bug', 'Feature'],
                        colorscale='Blues',
                        showscale=False,
                        text=cm,
                        texttemplate="%{text}"
                    ),
                    row=row, col=col
                )
        
        # Ensemble vs individual
        if 'Ensemble' in self.results:
            ensemble_acc = self.results['Ensemble']['accuracy']
            individual_acc = [self.results[name]['accuracy'] for name in model_order if name in self.results]
            
            fig.add_trace(
                go.Scatter(
                    x=model_order[:len(individual_acc)],
                    y=individual_acc,
                    mode='lines+markers',
                    name='Individual Models',
                    line=dict(color='gray', width=2, dash='dash'),
                    marker=dict(size=10)
                ),
                row=3, col=3
            )
            
            fig.add_hline(
                y=ensemble_acc,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Ensemble: {ensemble_acc:.2%}",
                row=3, col=3
            )
        
        fig.update_layout(
            height=900,
            showlegend=False,
            title_text="Model Comparison Dashboard"
        )
        
        return fig
    
    def plot_probability_distribution(self, texts, predictions, probabilities, class_names):
        """Show probability distribution for predictions"""
        fig = go.Figure()
        
        for i, text in enumerate(texts[:5]):  # Show first 5
            probs = probabilities[i]
            
            fig.add_trace(go.Bar(
                name=f"Text {i+1}",
                x=class_names,
                y=probs,
                text=[f"{p:.1%}" for p in probs],
                textposition='inside',
                hovertemplate=f"<b>{text[:50]}...</b><br>Probability: %{{y:.1%}}<extra></extra>"
            ))
        
        fig.update_layout(
            title="Probability Distribution for Multiple Predictions",
            xaxis_title="Class",
            yaxis_title="Probability",
            yaxis_range=[0, 1],
            barmode='group',
            height=500
        )
        
        return fig
    
    def plot_improvement_chart(self, before_acc, after_acc, technique="Optimization"):
        """Show improvement after optimization"""
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Before', 'After'],
            y=[before_acc, after_acc],
            marker_color=['#FF6B6B', '#4ECDC4'],
            text=[f"{before_acc:.1%}", f"{after_acc:.1%}"],
            textposition='outside',
            textfont=dict(size=16)
        ))
        
        improvement = (after_acc - before_acc) * 100
        fig.add_annotation(
            x=1, y=after_acc,
            text=f"+{improvement:.1f}% improvement",
            showarrow=True,
            arrowhead=1,
            font=dict(size=14, color="green")
        )
        
        fig.update_layout(
            title=f"Model Improvement with {technique}",
            xaxis_title="Stage",
            yaxis_title="Accuracy",
            yaxis_range=[0, 1],
            yaxis_tickformat='.0%',
            height=400
        )
        
        return fig

def create_feature_importance_plot(feature_names, importance_scores, top_n=20):
    """Create feature importance visualization"""
    # Sort and get top N
    indices = np.argsort(importance_scores)[::-1][:top_n]
    
    fig = go.Figure(data=[
        go.Bar(
            x=importance_scores[indices],
            y=[feature_names[i][:30] + "..." if len(feature_names[i]) > 30 else feature_names[i] 
               for i in indices],
            orientation='h',
            marker=dict(
                color=importance_scores[indices],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance", x=1.05)
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
        yaxis=dict(autorange="reversed"),
        margin=dict(l=200)
    )
    
    return fig

def plot_training_history(history):
    """Plot training accuracy and loss curves"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Model Accuracy', 'Model Loss')
    )
    
    # Accuracy
    fig.add_trace(
        go.Scatter(
            y=history.history['accuracy'],
            name='Train Accuracy',
            mode='lines',
            line=dict(color='#2E91E5', width=2)
        ),
        row=1, col=1
    )
    
    if 'val_accuracy' in history.history:
        fig.add_trace(
            go.Scatter(
                y=history.history['val_accuracy'],
                name='Val Accuracy',
                mode='lines',
                line=dict(color='#FF6B6B', width=2, dash='dash')
            ),
            row=1, col=1
        )
    
    # Loss
    fig.add_trace(
        go.Scatter(
            y=history.history['loss'],
            name='Train Loss',
            mode='lines',
            line=dict(color='#4ECDC4', width=2)
        ),
        row=1, col=2
    )
    
    if 'val_loss' in history.history:
        fig.add_trace(
            go.Scatter(
                y=history.history['val_loss'],
                name='Val Loss',
                mode='lines',
                line=dict(color='#45B7D1', width=2, dash='dash')
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        height=400,
        showlegend=True,
        title_text="Training History"
    )
    
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1, tickformat='.0%')
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    
    return fig