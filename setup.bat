@echo off
cd C:\Users\bhava\Downloads\bug-prediction-app

echo Creating requirements.txt...
echo streamlit==1.28.0 > requirements.txt
echo pandas==1.5.3 >> requirements.txt
echo numpy==1.24.3 >> requirements.txt
echo scikit-learn==1.3.0 >> requirements.txt
echo joblib==1.3.2 >> requirements.txt
echo plotly==5.15.0 >> requirements.txt

echo Creating runtime.txt...
echo 3.9.18 > runtime.txt

echo Creating .streamlit config...
mkdir .streamlit 2>nul
echo [server] > .streamlit\config.toml
echo maxUploadSize = 10 >> .streamlit\config.toml
echo enableCORS = false >> .streamlit\config.toml
echo enableXsrfProtection = true >> .streamlit\config.toml
echo headless = true >> .streamlit\config.toml
echo. >> .streamlit\config.toml
echo [theme] >> .streamlit\config.toml
echo primaryColor = "#00A3E0" >> .streamlit\config.toml
echo backgroundColor = "#FFFFFF" >> .streamlit\config.toml
echo secondaryBackgroundColor = "#F0F2F6" >> .streamlit\config.toml
echo textColor = "#2C3E50" >> .streamlit\config.toml
echo font = "sans serif" >> .streamlit\config.toml

echo Files created successfully!
pause