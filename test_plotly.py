import streamlit as st
import plotly.graph_objects as go

st.title("Plotly Test")
st.write("Testing if plotly is installed...")

try:
    fig = go.Figure(data=go.Bar(x=[1,2,3], y=[1,3,2]))
    st.plotly_chart(fig)
    st.success("✅ Plotly is working!")
except Exception as e:
    st.error(f"❌ Error: {e}")