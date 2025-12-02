# MachinLearningProject: Hong Kong Daily Minimum Temperature Forecasting with RNN & LSTM
This is a deep learning project that forecasts Hong Kong's daily minimum grass temperature in 2025 using Simple RNN and LSTM models, trained on historical data (1947–2024) from the Hong Kong Observatory (HKO). \
The dataset used in this training are from the HK government's open data libary provided by HKO:\
https://data.gov.hk/en-data/dataset/hk-hko-rss-daily-grass-min-temp \
Daily Grass Minimum Temperature All Year - Hong Kong Observatory (daily_HKO_GMT_ALL.csv) \
Daily Grass Minimum Temperature Current Year - Hong Kong Observatory (daily_HKO_GMT_2025.csv) \

Uses a 60-day sliding window to predict the next day’s temperature. Includes hyperparameter tuning (Keras Tuner), early stopping, bootstrap confidence intervals, and comprehensive error analysis.
## Contents
- requirements.txt for local environments
- an Jupyter/Colab notebook file
- the trained Keras Tuner trials & best models (in /model)
- a Standalone Python script
## Requirements
numpy 2.1.2 \
pandas 2.2.3 \
matplotlib 3.9.2 \
scipy 1.14.1 \
scikit-learn 1.5.2 \
tensorflow 2.17.0 \
keras-tuner 1.4.7 \
meteostat 1.6.8 \
Install via bash: \
pip install -r requirements.txt
## Running steps
For local environments: 
1. install the requirements if you haven't
2. if you want to train your own model, delete/ rename the model folder 
3. run the .py script

For colab / juypter environments: 
1. open the .ipynb file
2. Run all cells sequentially
3. Plots and results will appear as output
## Extra notes
- You can enable the stop_early code in both createRNNModel and creatLSTMModel function to reduce the time needed for model training (could make result less accurate)