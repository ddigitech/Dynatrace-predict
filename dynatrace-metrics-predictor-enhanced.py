
import os
import logging
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pmdarima import auto_arima
from typing import Dict, Any, Optional

class EnhancedDynatraceMetricsPredictor:
    def __init__(self, dynatrace_url: str, api_token: str):
        """
        Initialize the Enhanced Dynatrace Metrics Predictor with comprehensive logging and error handling.
        
        :param dynatrace_url: Base URL for Dynatrace environment
        :param api_token: API token for authentication
        """
        # Configure logging with more detailed configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('dynatrace_metrics_predictor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.dynatrace_url = dynatrace_url
        self.headers = {
            'Authorization': f'Api-Token {api_token}',
            'Content-Type': 'application/json'
        }
        
        # Enhanced error tracking
        self.errors = []
    
    def fetch_metrics(self, metric_key: str, from_time: str, to_time: str) -> Optional[pd.DataFrame]:
        """
        Enhanced metrics retrieval with comprehensive error handling and logging.
        
        :param metric_key: The specific metric to retrieve
        :param from_time: Start timestamp
        :param to_time: End timestamp
        :return: DataFrame with metrics data or None
        """
        self.logger.info(f"Attempting to fetch metrics for {metric_key}")
        
        try:
            endpoint = f"{self.dynatrace_url}/api/v2/metrics/query"
            
            params = {
                'metricSelector': metric_key,
                'from': from_time,
                'to': to_time,
                'resolution': '1h'
            }
            
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            
            metric_data = response.json()
            
            if not metric_data or 'result' not in metric_data:
                raise ValueError("No metric data retrieved")
            
            df = pd.DataFrame(metric_data['result'][0]['data'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            self.logger.info(f"Successfully retrieved {len(df)} metric data points")
            return df
        
        except requests.exceptions.RequestException as req_error:
            error_msg = f"Network error fetching metrics: {req_error}"
            self.logger.error(error_msg)
            self.errors.append(error_msg)
        
        except ValueError as val_error:
            error_msg = f"Data validation error: {val_error}"
            self.logger.error(error_msg)
            self.errors.append(error_msg)
        
        except Exception as e:
            error_msg = f"Unexpected error in metric retrieval: {e}"
            self.logger.error(error_msg)
            self.errors.append(error_msg)
        
        return None
    
    def advanced_prediction(self, metrics_df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Advanced prediction using multiple models and comprehensive evaluation.
        
        :param metrics_df: DataFrame with metric data
        :param target_column: Column to predict
        :return: Dictionary of prediction results
        """
        self.logger.info("Starting advanced prediction")
        
        # Prepare features
        X = metrics_df[['hour_of_day', 'day_of_week']]
        y = metrics_df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Multiple prediction models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                results[name] = {
                    'MSE': mean_squared_error(y_test, y_pred),
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'R2': r2_score(y_test, y_pred)
                }
                
                self.logger.info(f"Model {name} evaluation complete")
            
            except Exception as e:
                error_msg = f"Error in {name} model: {e}"
                self.logger.error(error_msg)
                self.errors.append(error_msg)
        
        # ARIMA for time series specific prediction
        try:
            arima_model = auto_arima(y, seasonal=True, m=24)
            arima_forecast = arima_model.predict(n_periods=7*24)
            results['ARIMA'] = {
                'forecast': arima_forecast.tolist()
            }
        except Exception as e:
            error_msg = f"ARIMA model error: {e}"
            self.logger.error(error_msg)
            self.errors.append(error_msg)
        
        return results
    
    def visualize_predictions(self, metrics_df: pd.DataFrame, predictions: Dict[str, Any], target_column: str):
        """
        Create comprehensive visualization of historical data and predictions.
        
        :param metrics_df: Original metrics DataFrame
        :param predictions: Prediction results dictionary
        :param target_column: Column being predicted
        """
        plt.figure(figsize=(15, 8))
        
        # Plot historical data
        plt.plot(metrics_df['timestamp'], metrics_df[target_column], 
                 label='Historical Data', color='blue')
        
        # Plot model predictions
        for model_name, model_results in predictions.items():
            if 'forecast' in model_results:
                forecast_timestamps = pd.date_range(
                    start=metrics_df['timestamp'].max(), 
                    periods=len(model_results['forecast'])+1, 
                    freq='H'
                )[1:]
                
                plt.plot(forecast_timestamps, model_results['forecast'], 
                         label=f'{model_name} Forecast', linestyle='--')
        
        plt.title(f'Metrics Prediction for {target_column}')
        plt.xlabel('Timestamp')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save visualization
        plt.savefig('dynatrace_metrics_prediction.png')
        self.logger.info("Prediction visualization saved")
    
    def generate_comprehensive_report(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report.
        
        :param predictions: Prediction results dictionary
        :return: Detailed report dictionary
        """
        report = {
            'errors': self.errors,
            'model_performance': {}
        }
        
        for model_name, results in predictions.items():
            if 'MSE' in results:
                report['model_performance'][model_name] = {
                    'Mean Squared Error': results['MSE'],
                    'Mean Absolute Error': results['MAE'],
                    'R-squared': results['R2']
                }
        
        return report

def main():
    # Configuration
    DYNATRACE_URL = os.getenv('DYNATRACE_URL', 'https://cfv77596.live.dynatrace.com')
    API_TOKEN = os.getenv('DYNATRACE_API_TOKEN', 'dt0c01.J76V6X34XUKDKVJ2AV7MFYFB.IMUGHAX7LEVH2MAPRR43LGJZ6BWI3ALPLVRVPZYKYYB6S2Y24CIVEKW3HNWO7Z3U')
    METRIC_KEY = 'builtin:service.requestCount.total'
    
    predictor = EnhancedDynatraceMetricsPredictor(DYNATRACE_URL, API_TOKEN)
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
    # Retrieve metrics
    metrics = predictor.fetch_metrics(
        METRIC_KEY, 
        start_time.isoformat(), 
        end_time.isoformat()
    )
    
    if metrics is not None:
        # Prepare data
        metrics['hour_of_day'] = metrics['timestamp'].dt.hour
        metrics['day_of_week'] = metrics['timestamp'].dt.dayofweek
        
        # Advanced prediction
        predictions = predictor.advanced_prediction(metrics, 'value')
        
        # Visualize predictions
        predictor.visualize_predictions(metrics, predictions, 'value')
        
        # Generate report
        report = predictor.generate_comprehensive_report(predictions)
        
        print("Comprehensive Prediction Report:")
        print(report)

if __name__ == "__main__":
    main()