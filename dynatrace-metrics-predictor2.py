import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class DynatraceMetricsPredictor:
    def __init__(self, dynatrace_url, api_token):
        """
        Initialize the Dynatrace Metrics Predictor with connection details.
        
        :param dynatrace_url: Base URL for Dynatrace environment
        :param api_token: API token for authentication
        """
        self.dynatrace_url = dynatrace_url
        self.headers = {
            'Authorization': f'Api-Token {api_token}',
            'Content-Type': 'application/json'
        }
    
    def fetch_metrics(self, metric_key, from_time, to_time):
        """
        Fetch metrics from Dynatrace for a specific time range.
        
        :param metric_key: The specific metric to retrieve
        :param from_time: Start timestamp
        :param to_time: End timestamp
        :return: DataFrame with metrics data
        """
        # Construct the API endpoint for metrics querying
        endpoint = f"{self.dynatrace_url}/api/v2/metrics/query"
        
        params = {
            'metricSelector': metric_key,
            'from': from_time,
            'to': to_time,
            'resolution': '1h'  # Hourly resolution
        }
        
        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            
            metric_data = response.json()
            
            # Transform metrics into a pandas DataFrame
            df = pd.DataFrame(metric_data['result'][0]['data'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching metrics: {e}")
            return None
    
    def prepare_prediction_data(self, metrics_df):
        """
        Prepare data for prediction by creating time-based features.
        
        :param metrics_df: DataFrame with metric data
        :return: Prepared DataFrame for prediction
        """
        metrics_df['hour_of_day'] = metrics_df['timestamp'].dt.hour
        metrics_df['day_of_week'] = metrics_df['timestamp'].dt.dayofweek
        
        return metrics_df
    
    def predict_metrics(self, metrics_df, target_column, prediction_days=7):
        """
        Predict metrics for the next 7 days using polynomial regression.
        
        :param metrics_df: Prepared DataFrame with metrics
        :param target_column: Column to predict
        :param prediction_days: Number of days to predict
        :return: DataFrame with predictions
        """
        # Prepare features
        X = metrics_df[['hour_of_day', 'day_of_week']]
        y = metrics_df[target_column]
        
        # Use polynomial features for more complex prediction
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        
        # Train polynomial regression model
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Generate prediction timestamps
        last_timestamp = metrics_df['timestamp'].max()
        prediction_timestamps = [last_timestamp + timedelta(hours=h) 
                                 for h in range(1, prediction_days * 24 + 1)]
        
        # Create prediction DataFrame
        prediction_df = pd.DataFrame({
            'timestamp': prediction_timestamps,
            'hour_of_day': [ts.hour for ts in prediction_timestamps],
            'day_of_week': [ts.dayofweek for ts in prediction_timestamps]
        })
        
        # Transform prediction features
        X_pred_poly = poly.transform(prediction_df[['hour_of_day', 'day_of_week']])
        
        # Predict values
        prediction_df[target_column] = model.predict(X_pred_poly)
        
        return prediction_df
    
    def generate_metrics_report(self, predictions):
        """
        Generate a summary report of predicted metrics.
        
        :param predictions: DataFrame with metric predictions
        :return: Dictionary with summary statistics
        """
        report = {
            'mean_prediction': predictions[predictions.columns[-1]].mean(),
            'median_prediction': predictions[predictions.columns[-1]].median(),
            'min_prediction': predictions[predictions.columns[-1]].min(),
            'max_prediction': predictions[predictions.columns[-1]].max(),
            'prediction_variance': predictions[predictions.columns[-1]].var()
        }
        
        return report

def main():
    # Configuration - replace with your actual Dynatrace details
    DYNATRACE_URL = os.getenv('DYNATRACE_URL', 'https://your-tenant.dynatrace.com')
    API_TOKEN = os.getenv('DYNATRACE_API_TOKEN', 'your-api-token')
    
    # Specify the metric you want to analyze
    METRIC_KEY = 'builtin:service.requestCount.total'
    
    # Time range for historical data
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)  # Last 30 days
    
    # Initialize predictor
    predictor = DynatraceMetricsPredictor(DYNATRACE_URL, API_TOKEN)
    
    # Fetch metrics
    metrics = predictor.fetch_metrics(
        METRIC_KEY, 
        start_time.isoformat(), 
        end_time.isoformat()
    )
    
    if metrics is not None:
        # Prepare metrics for prediction
        prepared_metrics = predictor.prepare_prediction_data(metrics)
        
        # Predict next 7 days
        predictions = predictor.predict_metrics(
            prepared_metrics, 
            target_column='value', 
            prediction_days=7
        )
        
        # Generate report
        report = predictor.generate_metrics_report(predictions)
        
        # Save predictions and report
        predictions.to_csv('dynatrace_metric_predictions.csv', index=False)
        
        print("Metrics Prediction Report:")
        for key, value in report.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()
