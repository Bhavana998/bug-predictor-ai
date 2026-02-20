@echo off
echo ========================================
echo   BUG PREDICTION MODEL - COMPLETE SETUP
echo ========================================
echo.

echo Step 1: Preparing balanced dataset...
python prepare_dataset.py

echo.
echo Step 2: Training model...
python train_model_final.py

echo.
echo Step 3: Starting Streamlit app...
echo The app will open at http://localhost:8501
start http://localhost:8501
streamlit run app.py

pause