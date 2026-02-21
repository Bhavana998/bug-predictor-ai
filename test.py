import streamlit as st
import plotly.graph_objects as go

st.title("BugSense AI - Test")
st.write("Testing plotly import...")

fig = go.Figure(data=go.Bar(x=[1,2,3], y=[1,3,2]))
st.plotly_chart(fig)

st.success("✅ Plotly is working!")