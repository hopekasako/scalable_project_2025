import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime, timedelta
import time
import os
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class EnhancedStockForecastingPipeline:
    def __init__(self):
        try:
            # Get the absolute path to the model file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, 'models/linear_regression_model.pkl')
            
            # Load the pre-trained model
            try:
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found at: {model_path}")
                    
                model_data = joblib.load(model_path)
                self.model = model_data['model']
                self.features = model_data['features']
                logging.info(f"Model loaded successfully from: {model_path}")
                logging.info(f"Features used: {self.features}")
            except Exception as e:
                logging.error(f"Error loading model: {str(e)}")
                raise
            
            # Initialize cache
            self._data_cache = {}
            self._last_fetch_time = None
            self._cache_duration = timedelta(minutes=1)  # Cache data for 1 minute
            
        except Exception as e:
            logging.error(f"Initialization error: {str(e)}")
            raise

    def calculate_ema(self, data, span):
        """Calculate EMA using pandas"""
        return data.ewm(span=span, adjust=False).mean()

    def fetch_real_time_data(self):
        """Fetch real-time Netflix stock data with caching"""
        try:
            current_time = datetime.now()
            
            # Check if we have cached data that's still valid
            if (self._last_fetch_time and 
                current_time - self._last_fetch_time < self._cache_duration and 
                'data' in self._data_cache):
                logging.info("Using cached data")
                return self._data_cache['data']
            
            stock = yf.Ticker("NFLX")
            data = stock.history(period="1d", interval="1m")
            data = data.reset_index()
            
            # Ensure column names match expected format
            data.columns = data.columns.str.title()
            if 'Datetime' not in data.columns and 'Date' in data.columns:
                data = data.rename(columns={'Date': 'Datetime'})
            
            # Cache the data
            self._data_cache['data'] = data
            self._last_fetch_time = current_time
            
            logging.info(f"Fetched {len(data)} records of real-time data")
            return data
            
        except Exception as e:
            logging.error(f"Error fetching data: {str(e)}")
            return None

    def calculate_technical_indicators(self, df):
        """Calculate all technical indicators using pandas"""
        try:
            # Calculate returns
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Price Range calculations
            df['Price_Range'] = df['High'] - df['Low']
            df['Price_Range_Pct'] = df['Price_Range'] / df['Open']
            
            # Moving Averages
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            
            # EMAs
            df['EMA_5'] = self.calculate_ema(df['Close'], 5)
            df['EMA_20'] = self.calculate_ema(df['Close'], 20)
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Fill NaN values with 0
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating indicators: {str(e)}")
            return None

    def make_predictions(self, df):
        """Make predictions using the loaded model"""
        try:
            # Extract features for prediction
            X = df[self.features]
            
            # Make predictions
            predictions = self.model.predict(X)
            
            # Add predictions to DataFrame
            df['Predicted_Price'] = predictions
            df['Prediction_Time'] = datetime.now()
            
            # Calculate prediction errors
            df['Prediction_Error'] = (df['Predicted_Price'] - df['Close']).abs()
            df['Prediction_Error_Pct'] = (df['Prediction_Error'] / df['Close'] * 100)
            
            return df
            
        except Exception as e:
            logging.error(f"Error making predictions: {str(e)}")
            return None

    def run_pipeline(self):
        """Run the enhanced pipeline"""
        try:
            logging.info("Starting real-time forecasting pipeline...")
            
            while True:
                # Fetch and process data
                df = self.fetch_real_time_data()
                if df is not None:
                    logging.info("Calculating technical indicators...")
                    df = self.calculate_technical_indicators(df)
                    if df is not None:
                        logging.info("Making predictions...")
                        predictions_df = self.make_predictions(df)
                        if predictions_df is not None:
                            # Display latest prediction
                            latest = predictions_df.iloc[-1]
                            print("\nLatest Prediction:")
                            print(f"Time: {latest['Datetime']}")
                            print(f"Current Price: ${latest['Close']:.2f}")
                            print(f"Predicted Price: ${latest['Predicted_Price']:.2f}")
                            print(f"Prediction Error: {latest['Prediction_Error_Pct']:.2f}%")
                            print("-" * 50)
                
                # Wait before next update
                time.sleep(60)
                
        except KeyboardInterrupt:
            logging.info("Shutting down pipeline gracefully...")
        except Exception as e:
            logging.error(f"Pipeline error: {str(e)}")

if __name__ == "__main__":
    try:
        pipeline = EnhancedStockForecastingPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.error(f"Failed to start pipeline: {str(e)}")