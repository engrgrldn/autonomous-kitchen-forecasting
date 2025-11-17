"""
AI-Augmented Demand Forecasting with LLM Explanations
======================================================
Enhances traditional ML forecasts with natural language explanations
using Large Language Models (Claude/GPT).

Author: Geraldine Castillo
Date: November 2025
"""

import anthropic
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json
from datetime import datetime, timedelta


class ForecastExplainer:
    """
    Provides natural language explanations for demand forecasts using LLMs.
    """
    
    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize the forecast explainer.
        
        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env variable)
            model: Model to use for explanations
        """
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model
        
    def explain_forecast(
        self, 
        prediction: float,
        historical_avg: float,
        features: Dict[str, float],
        context: Dict[str, any] = None
    ) -> Dict[str, str]:
        """
        Generate natural language explanation for a forecast.
        
        Args:
            prediction: Forecasted demand value
            historical_avg: Historical average for comparison
            features: Dictionary of feature values used in prediction
            context: Additional context (location, date, etc.)
            
        Returns:
            Dictionary with explanation, key_factors, and recommendations
        """
        
        # Calculate change
        pct_change = ((prediction - historical_avg) / historical_avg) * 100
        
        # Prepare feature summary
        feature_text = self._format_features(features)
        
        # Build prompt
        prompt = f"""Analyze this demand forecast for an autonomous kitchen:

FORECAST DETAILS:
- Predicted demand: {prediction:.0f} orders
- Historical average: {historical_avg:.0f} orders
- Change: {pct_change:+.1f}%

KEY FEATURES:
{feature_text}

CONTEXT:
{json.dumps(context, indent=2) if context else 'None provided'}

Provide a concise analysis with:
1. Main explanation (2-3 sentences): Why is demand at this level?
2. Key factors (3-5 bullet points): What's driving this forecast?
3. Business recommendations (2-3 actions): What should kitchen operators do?

Format as JSON:
{{
    "explanation": "...",
    "key_factors": ["factor 1", "factor 2", ...],
    "recommendations": ["action 1", "action 2", ...],
    "confidence": "high/medium/low"
}}
"""
        
        # Get LLM explanation
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = message.content[0].text
            # Parse JSON response
            result = json.loads(response_text)
            
            # Add metadata
            result['prediction'] = prediction
            result['change_pct'] = pct_change
            result['timestamp'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "explanation": f"Predicted {prediction:.0f} orders ({pct_change:+.1f}% vs average)",
                "key_factors": ["Unable to generate detailed explanation"],
                "recommendations": ["Review forecast manually"],
                "confidence": "low"
            }
    
    def _format_features(self, features: Dict[str, float]) -> str:
        """Format features for LLM prompt."""
        lines = []
        for key, value in features.items():
            # Format based on feature type
            if 'lag' in key.lower():
                lines.append(f"- {key}: {value:.0f} orders")
            elif 'day' in key.lower():
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                if 0 <= value < 7:
                    lines.append(f"- Day of week: {days[int(value)]}")
            elif 'promotion' in key.lower() or 'holiday' in key.lower():
                lines.append(f"- {key}: {'Yes' if value == 1 else 'No'}")
            else:
                lines.append(f"- {key}: {value:.2f}")
        
        return '\n'.join(lines)
    
    def detect_anomalies(
        self, 
        predictions: pd.Series,
        actual: pd.Series = None,
        threshold: float = 2.0
    ) -> List[Dict]:
        """
        Detect anomalies in forecasts and explain them.
        
        Args:
            predictions: Series of predictions
            actual: Series of actual values (optional)
            threshold: Standard deviations for anomaly detection
            
        Returns:
            List of anomaly explanations
        """
        anomalies = []
        
        # Statistical anomaly detection
        mean = predictions.mean()
        std = predictions.std()
        
        for idx, pred in predictions.items():
            z_score = abs((pred - mean) / std)
            
            if z_score > threshold:
                anomaly = {
                    'date': idx,
                    'value': pred,
                    'z_score': z_score,
                    'severity': 'high' if z_score > 3 else 'medium'
                }
                
                # Get LLM explanation for anomaly
                prompt = f"""Explain this demand anomaly in an autonomous kitchen:

- Date: {idx}
- Predicted demand: {pred:.0f} orders
- Normal range: {mean:.0f} ± {std:.0f} orders
- Z-score: {z_score:.2f}

In 1-2 sentences, explain what might cause this unusual demand level.
"""
                
                try:
                    message = self.client.messages.create(
                        model=self.model,
                        max_tokens=150,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    anomaly['explanation'] = message.content[0].text
                except:
                    anomaly['explanation'] = f"Demand {z_score:.1f}σ from normal"
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def generate_weekly_summary(
        self,
        forecast_df: pd.DataFrame,
        features_df: pd.DataFrame
    ) -> str:
        """
        Generate executive summary of weekly forecast.
        
        Args:
            forecast_df: DataFrame with predictions
            features_df: DataFrame with feature values
            
        Returns:
            Natural language summary
        """
        
        # Calculate summary statistics
        total_demand = forecast_df['prediction'].sum()
        daily_avg = forecast_df['prediction'].mean()
        peak_day = forecast_df.loc[forecast_df['prediction'].idxmax()]
        low_day = forecast_df.loc[forecast_df['prediction'].idxmin()]
        
        # Identify trends
        trend = 'increasing' if forecast_df['prediction'].iloc[-1] > forecast_df['prediction'].iloc[0] else 'decreasing'
        
        prompt = f"""Generate an executive summary for this week's demand forecast:

WEEKLY STATISTICS:
- Total expected demand: {total_demand:.0f} orders
- Daily average: {daily_avg:.0f} orders
- Peak day: {peak_day.name} with {peak_day['prediction']:.0f} orders
- Lowest day: {low_day.name} with {low_day['prediction']:.0f} orders
- Trend: {trend}

Write a professional 3-paragraph summary covering:
1. Overview and key numbers
2. Notable patterns or concerns
3. Strategic recommendations

Keep it concise and actionable for kitchen managers.
"""
        
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            return f"Weekly forecast: {total_demand:.0f} total orders, {daily_avg:.0f} avg/day. Error generating details: {str(e)}"


class AutomatedInsights:
    """
    Automatically generate business insights from forecast data.
    """
    
    def __init__(self, explainer: ForecastExplainer):
        self.explainer = explainer
    
    def analyze_forecast_vs_capacity(
        self,
        forecast: pd.Series,
        capacity: float
    ) -> Dict:
        """
        Analyze if forecasted demand exceeds kitchen capacity.
        
        Args:
            forecast: Series of demand predictions
            capacity: Maximum daily capacity
            
        Returns:
            Analysis with recommendations
        """
        overcapacity_days = forecast[forecast > capacity]
        utilization = (forecast / capacity * 100).mean()
        
        if len(overcapacity_days) > 0:
            prompt = f"""Analyze this capacity issue for an autonomous kitchen:

SITUATION:
- Kitchen capacity: {capacity:.0f} orders/day
- Average utilization: {utilization:.1f}%
- Days exceeding capacity: {len(overcapacity_days)}
- Overcapacity dates: {list(overcapacity_days.index)}
- Excess demand: {overcapacity_days.values}

Provide:
1. Risk assessment (2 sentences)
2. Three immediate actions to take
3. Long-term recommendation

Format as JSON with keys: risk, actions, long_term
"""
            
            try:
                message = self.explainer.client.messages.create(
                    model=self.explainer.model,
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}]
                )
                result = json.loads(message.content[0].text)
                result['overcapacity_days'] = len(overcapacity_days)
                result['utilization'] = utilization
                return result
            except:
                return {
                    'risk': 'Capacity exceeded',
                    'actions': ['Increase capacity', 'Adjust menu', 'Limit orders'],
                    'long_term': 'Review capacity planning',
                    'overcapacity_days': len(overcapacity_days),
                    'utilization': utilization
                }
        else:
            return {
                'risk': 'No capacity issues',
                'utilization': utilization,
                'message': 'Forecast within capacity limits'
            }
    
    def compare_scenarios(
        self,
        base_forecast: pd.Series,
        scenario_forecast: pd.Series,
        scenario_name: str
    ) -> Dict:
        """
        Compare two forecast scenarios and explain differences.
        
        Args:
            base_forecast: Baseline predictions
            scenario_forecast: Alternative scenario predictions
            scenario_name: Name of the scenario
            
        Returns:
            Comparison analysis
        """
        diff = scenario_forecast - base_forecast
        pct_change = (diff / base_forecast * 100).mean()
        
        prompt = f"""Compare these two demand forecast scenarios:

SCENARIO: {scenario_name}

DIFFERENCES:
- Average change: {pct_change:+.1f}%
- Total impact: {diff.sum():+.0f} orders
- Days with biggest change: {diff.nlargest(3).to_dict()}

Explain:
1. What this scenario means (1-2 sentences)
2. Business impact (2-3 points)
3. Should we prepare for this scenario? (Yes/No with reason)

Format as JSON with keys: explanation, impact, recommendation
"""
        
        try:
            message = self.explainer.client.messages.create(
                model=self.explainer.model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            result = json.loads(message.content[0].text)
            result['avg_change_pct'] = pct_change
            result['total_impact'] = diff.sum()
            return result
        except:
            return {
                'explanation': f'{scenario_name}: {pct_change:+.1f}% change',
                'impact': [f'Total impact: {diff.sum():+.0f} orders'],
                'recommendation': 'Review scenario manually'
            }


# Example usage
if __name__ == "__main__":
    # Initialize explainer
    explainer = ForecastExplainer()
    
    # Example forecast explanation
    result = explainer.explain_forecast(
        prediction=450,
        historical_avg=380,
        features={
            'day_of_week': 5,  # Friday
            'lag_7': 420,
            'rolling_avg_7': 390,
            'is_promotion': 1,
            'is_holiday': 0,
            'temperature': 22.5
        },
        context={
            'location': 'Kitchen_A',
            'date': '2025-11-22'
        }
    )
    
    print("Forecast Explanation:")
    print(json.dumps(result, indent=2))
    
    # Example anomaly detection
    sample_predictions = pd.Series([
        380, 390, 385, 395, 800, 375, 385  # Day 5 is anomalous
    ], index=pd.date_range('2025-11-17', periods=7))
    
    anomalies = explainer.detect_anomalies(sample_predictions)
    print("\nDetected Anomalies:")
    print(json.dumps(anomalies, indent=2, default=str))
