import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class DynatraceMetricsPredictor:
    def __init__(self, tenant_url, api_token):
        """
        Initialize the Dynatrace Metrics Predictor
        
        :param tenant_url: Your Dynatrace tenant URL
        :param api_token: Your Dynatrace API token
        """
        self.tenant_url = tenant_url
        self.api_token = api_token
        self.headers = {
            'Authorization': f'Api-Token {self.api_token}',
            'Content-Type': 'application/json'
        }
    
    def fetch_metric(self, metric_key, from_time, to_time):
        """
        Fetch a specific metric from Dynatrace
        
        :param metric_key: The specific metric to retrieve
        :param from_time: Start timestamp
        :param to_time: End timestamp
        :return: Pandas DataFrame with metric data
        """
        # Construct the API endpoint
        endpoint = f"{self.tenant_url}/api/v2/metrics/query"
        
        # Prepare query parameters
        params = {
            'metricSelector': metric_key,
            'from': from_time,
            'to': to_time,
            'resolution': 'PT1H'  # 1-hour resolution
        }
        
        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            metric_data = []
            for result in data.get('result', []):
                for datapoint in result.get('data', []):
                    metric_data.append({
                        'timestamp': datetime.fromtimestamp(datapoint[0]/1000),
                        'value': datapoint[1]
                    })
            
            return pd.DataFrame(metric_data)
        
        except requests.RequestException as e:
            print(f"Error fetching metric: {e}")
            return pd.DataFrame()
    
    def predict_metric(self, metric_df, prediction_days=7):
        """
        Predict future metric values using polynomial regression
        
        :param metric_df: DataFrame with timestamp and value columns
        :param prediction_days: Number of days to predict
        :return: DataFrame with predicted values
        """
        # Prepare data for regression
        metric_df = metric_df.sort_values('timestamp')
        metric_df['days_since_start'] = (metric_df['timestamp'] - metric_df['timestamp'].min()).dt.total_seconds() / (24 * 3600)
        
        # Prepare features and target
        X = metric_df['days_since_start'].values.reshape(-1, 1)
        y = metric_df['value'].values
        
        # Polynomial features
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)
        
        # Fit regression model
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Generate future predictions
        last_date = metric_df['timestamp'].max()
        future_days = np.array(range(1, prediction_days + 1)).reshape(-1, 1)
        future_days_total = future_days + metric_df['days_since_start'].max()
        
        # Transform future days
        future_days_poly = poly.transform(future_days_total)
        
        # Predict values
        predicted_values = model.predict(future_days_poly)
        
        # Create prediction DataFrame
        prediction_df = pd.DataFrame({
            'timestamp': [last_date + timedelta(days=x) for x in range(1, prediction_days + 1)],
            'predicted_value': predicted_values
        })
        
        return prediction_df
    
    def analyze_metrics(self, metric_key, days_back=30):
        """
        Comprehensive metric analysis
        
        :param metric_key: Metric to analyze
        :param days_back: Number of historical days to analyze
        :return: Dictionary with analysis results
        """
        # Calculate timestamps
        to_time = int(datetime.now().timestamp() * 1000)
        from_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
        
        # Fetch historical metric
        historical_data = self.fetch_metric(metric_key, from_time, to_time)
        
        if historical_data.empty:
            return {"error": "No metric data retrieved"}
        
        # Predict future values
        predictions = self.predict_metric(historical_data)
        
        # Perform basic statistical analysis
        analysis = {
            "historical_stats": {
                "mean": historical_data['value'].mean(),
                "median": historical_data['value'].median(),
                "std_dev": historical_data['value'].std(),
                "min": historical_data['value'].min(),
                "max": historical_data['value'].max()
            },
            "predictions": predictions.to_dict(orient='records'),
            "trend": "increasing" if predictions['predicted_value'].iloc[-1] > predictions['predicted_value'].iloc[0] else "decreasing"
        }
        
        return analysis

# Example usage
def main():
    # Replace with your actual Dynatrace tenant URL and API token
    tenant_url = 'https://your-tenant.dynatrace.com'
    api_token = 'your-api-token'
    
    # Initialize the predictor
    predictor = DynatraceMetricsPredictor(tenant_url, api_token)
    
    # Example metric key - replace with your specific Dynatrace metric
    metric_key = 'custom:your.specific.metric'
    
    # Analyze the metric
    result = predictor.analyze_metrics(metric_key)
    
    # Print results
    print("Metric Analysis:")
    print(result)

if __name__ == "__main__":
    main()
