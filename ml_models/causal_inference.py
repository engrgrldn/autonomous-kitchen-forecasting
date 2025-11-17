"""
Causal Inference for Demand Forecasting
=======================================
Implements causal analysis methods to understand true causal relationships
in demand patterns, beyond simple correlations.

Methods included:
- Causal Impact Analysis
- Difference-in-Differences (DiD)
- Propensity Score Matching
- Granger Causality
- Instrumental Variables

Author: Geraldine Castillo
Date: November 2025
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class CausalImpactAnalyzer:
    """
    Analyze causal impact of interventions (promotions, menu changes, etc.)
    on demand using counterfactual analysis.
    """
    
    def __init__(self):
        self.model = None
        self.pre_period_data = None
        self.post_period_data = None
        
    def analyze_impact(
        self,
        data: pd.Series,
        pre_period: Tuple[str, str],
        post_period: Tuple[str, str],
        covariates: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Analyze causal impact of an intervention.
        
        Args:
            data: Time series of target variable (demand)
            pre_period: (start, end) dates before intervention
            post_period: (start, end) dates after intervention
            covariates: Optional control variables
            
        Returns:
            Dictionary with causal impact results
        """
        # Split data
        pre_data = data.loc[pre_period[0]:pre_period[1]]
        post_data = data.loc[post_period[0]:post_period[1]]
        
        # Build counterfactual model using pre-intervention data
        if covariates is not None:
            pre_covariates = covariates.loc[pre_period[0]:pre_period[1]]
            post_covariates = covariates.loc[post_period[0]:post_period[1]]
            
            # Train regression model
            self.model = LinearRegression()
            self.model.fit(pre_covariates, pre_data)
            
            # Predict counterfactual (what would have happened without intervention)
            counterfactual = self.model.predict(post_covariates)
        else:
            # Simple moving average as counterfactual
            window = min(len(pre_data), 7)
            counterfactual = np.full(len(post_data), pre_data.rolling(window).mean().iloc[-1])
        
        # Calculate impact
        actual = post_data.values
        causal_effect = actual - counterfactual
        
        # Statistical significance
        relative_effect = (causal_effect / counterfactual * 100).mean()
        cumulative_effect = causal_effect.sum()
        
        # Confidence intervals (bootstrap)
        confidence_intervals = self._bootstrap_confidence_intervals(
            pre_data.values, post_data.values, counterfactual
        )
        
        # Probability of causal effect
        p_value = self._calculate_p_value(causal_effect)
        
        return {
            'average_causal_effect': causal_effect.mean(),
            'cumulative_effect': cumulative_effect,
            'relative_effect_pct': relative_effect,
            'confidence_interval_95': confidence_intervals,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'actual': actual,
            'counterfactual': counterfactual,
            'causal_effect': causal_effect,
            'interpretation': self._interpret_results(
                relative_effect, p_value, cumulative_effect
            )
        }
    
    def _bootstrap_confidence_intervals(
        self,
        pre_data: np.ndarray,
        post_data: np.ndarray,
        counterfactual: np.ndarray,
        n_iterations: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence intervals using bootstrap."""
        effects = []
        
        for _ in range(n_iterations):
            # Resample
            sample_indices = np.random.choice(len(post_data), len(post_data), replace=True)
            sample_actual = post_data[sample_indices]
            sample_counterfactual = counterfactual[sample_indices]
            
            # Calculate effect
            effect = (sample_actual - sample_counterfactual).mean()
            effects.append(effect)
        
        # Calculate percentiles
        alpha = (1 - confidence) / 2
        lower = np.percentile(effects, alpha * 100)
        upper = np.percentile(effects, (1 - alpha) * 100)
        
        return (lower, upper)
    
    def _calculate_p_value(self, causal_effect: np.ndarray) -> float:
        """Calculate p-value for causal effect."""
        # One-sample t-test against zero effect
        t_stat, p_value = stats.ttest_1samp(causal_effect, 0)
        return p_value
    
    def _interpret_results(
        self,
        relative_effect: float,
        p_value: float,
        cumulative_effect: float
    ) -> str:
        """Generate human-readable interpretation."""
        if p_value >= 0.05:
            return f"No significant causal effect detected (p={p_value:.3f}). The intervention likely did not impact demand."
        
        direction = "increased" if relative_effect > 0 else "decreased"
        magnitude = abs(relative_effect)
        
        if magnitude < 5:
            strength = "small"
        elif magnitude < 15:
            strength = "moderate"
        else:
            strength = "large"
        
        return (f"The intervention caused a {strength} {direction} in demand "
                f"({relative_effect:+.1f}%, p={p_value:.3f}). "
                f"Total impact: {cumulative_effect:+.0f} orders.")
    
    def plot_impact(self, results: Dict, intervention_date: str):
        """Visualize causal impact."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Actual vs Counterfactual
        x = range(len(results['actual']))
        ax1.plot(x, results['actual'], 'b-', label='Actual', linewidth=2)
        ax1.plot(x, results['counterfactual'], 'r--', label='Counterfactual', linewidth=2)
        ax1.fill_between(x, results['actual'], results['counterfactual'], 
                         alpha=0.3, color='green' if results['actual'].mean() > results['counterfactual'].mean() else 'red')
        ax1.axvline(x=0, color='black', linestyle=':', label='Intervention')
        ax1.set_title('Actual vs Counterfactual Demand')
        ax1.set_ylabel('Demand (orders)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Causal Effect
        ax2.bar(x, results['causal_effect'], color='green' if results['relative_effect_pct'] > 0 else 'red', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_title(f"Causal Effect: {results['relative_effect_pct']:+.1f}% (p={results['p_value']:.3f})")
        ax2.set_xlabel('Days After Intervention')
        ax2.set_ylabel('Effect (orders)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class DifferenceInDifferences:
    """
    Difference-in-Differences (DiD) analysis for causal inference.
    Compares treatment group with control group before/after intervention.
    """
    
    def __init__(self):
        self.results = None
        
    def analyze(
        self,
        treatment_pre: pd.Series,
        treatment_post: pd.Series,
        control_pre: pd.Series,
        control_post: pd.Series
    ) -> Dict:
        """
        Perform DiD analysis.
        
        Args:
            treatment_pre: Treatment group before intervention
            treatment_post: Treatment group after intervention
            control_pre: Control group before intervention
            control_post: Control group after intervention
            
        Returns:
            DiD results with causal estimate
        """
        # Calculate differences
        treatment_diff = treatment_post.mean() - treatment_pre.mean()
        control_diff = control_post.mean() - control_pre.mean()
        
        # DiD estimator
        did_estimate = treatment_diff - control_diff
        
        # Standard error and t-statistic
        se = self._calculate_standard_error(
            treatment_pre, treatment_post, control_pre, control_post
        )
        t_stat = did_estimate / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(treatment_post) + len(control_post) - 2))
        
        # Confidence interval
        ci_lower = did_estimate - 1.96 * se
        ci_upper = did_estimate + 1.96 * se
        
        return {
            'did_estimate': did_estimate,
            'treatment_effect': treatment_diff,
            'control_trend': control_diff,
            'standard_error': se,
            't_statistic': t_stat,
            'p_value': p_value,
            'confidence_interval': (ci_lower, ci_upper),
            'is_significant': p_value < 0.05,
            'interpretation': self._interpret_did(did_estimate, p_value)
        }
    
    def _calculate_standard_error(
        self,
        treatment_pre: pd.Series,
        treatment_post: pd.Series,
        control_pre: pd.Series,
        control_post: pd.Series
    ) -> float:
        """Calculate standard error for DiD estimator."""
        # Pooled variance
        var_treatment = treatment_post.var() / len(treatment_post) + treatment_pre.var() / len(treatment_pre)
        var_control = control_post.var() / len(control_post) + control_pre.var() / len(control_pre)
        
        return np.sqrt(var_treatment + var_control)
    
    def _interpret_did(self, estimate: float, p_value: float) -> str:
        """Interpret DiD results."""
        if p_value >= 0.05:
            return f"No significant causal effect (p={p_value:.3f}). The intervention did not have a measurable impact compared to the control group."
        
        direction = "increase" if estimate > 0 else "decrease"
        return f"The intervention caused a significant {direction} of {abs(estimate):.1f} orders (p={p_value:.3f}), controlling for time trends."


class PropensityScoreMatching:
    """
    Propensity Score Matching for causal inference.
    Matches treated and untreated units with similar characteristics.
    """
    
    def __init__(self):
        self.propensity_model = None
        
    def estimate_treatment_effect(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariate_cols: List[str],
        caliper: float = 0.1
    ) -> Dict:
        """
        Estimate average treatment effect using PSM.
        
        Args:
            data: DataFrame with treatment, outcome, and covariates
            treatment_col: Name of binary treatment indicator
            outcome_col: Name of outcome variable
            covariate_cols: List of covariate column names
            caliper: Maximum allowable distance for matching
            
        Returns:
            Treatment effect estimate
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import NearestNeighbors
        
        # Estimate propensity scores
        X = data[covariate_cols]
        treatment = data[treatment_col]
        
        self.propensity_model = LogisticRegression()
        self.propensity_model.fit(X, treatment)
        
        propensity_scores = self.propensity_model.predict_proba(X)[:, 1]
        data['propensity_score'] = propensity_scores
        
        # Separate treated and control
        treated = data[data[treatment_col] == 1]
        control = data[data[treatment_col] == 0]
        
        # Match treated units to control units
        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(control[['propensity_score']].values)
        
        matches = []
        for idx, row in treated.iterrows():
            ps = row['propensity_score']
            distances, indices = nn.kneighbors([[ps]])
            
            # Check if within caliper
            if distances[0][0] <= caliper:
                control_idx = control.iloc[indices[0][0]].name
                matches.append({
                    'treated_idx': idx,
                    'control_idx': control_idx,
                    'treated_outcome': row[outcome_col],
                    'control_outcome': control.loc[control_idx, outcome_col],
                    'difference': row[outcome_col] - control.loc[control_idx, outcome_col]
                })
        
        if len(matches) == 0:
            return {
                'error': 'No matches found within caliper',
                'ate': None
            }
        
        matches_df = pd.DataFrame(matches)
        
        # Average treatment effect
        ate = matches_df['difference'].mean()
        se = matches_df['difference'].std() / np.sqrt(len(matches_df))
        t_stat = ate / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(matches_df) - 1))
        
        return {
            'average_treatment_effect': ate,
            'num_matches': len(matches),
            'standard_error': se,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'matches': matches_df,
            'interpretation': f"Average treatment effect: {ate:+.1f} orders (p={p_value:.3f}), based on {len(matches)} matched pairs."
        }


class GrangerCausality:
    """
    Test for Granger causality between time series.
    Determines if one time series can predict another.
    """
    
    def __init__(self):
        pass
        
    def test_causality(
        self,
        cause_series: pd.Series,
        effect_series: pd.Series,
        max_lag: int = 7
    ) -> Dict:
        """
        Test if cause_series Granger-causes effect_series.
        
        Args:
            cause_series: Potential causal variable
            effect_series: Effect variable
            max_lag: Maximum lag to test
            
        Returns:
            Granger causality test results
        """
        results = {}
        
        for lag in range(1, max_lag + 1):
            # Create lagged dataframe
            data = pd.DataFrame({
                'effect': effect_series,
                'cause': cause_series
            })
            
            # Create lag features
            for i in range(1, lag + 1):
                data[f'effect_lag_{i}'] = data['effect'].shift(i)
                data[f'cause_lag_{i}'] = data['cause'].shift(i)
            
            data = data.dropna()
            
            # Restricted model: effect ~ effect_lags
            restricted_features = [f'effect_lag_{i}' for i in range(1, lag + 1)]
            X_restricted = data[restricted_features]
            y = data['effect']
            
            model_restricted = LinearRegression()
            model_restricted.fit(X_restricted, y)
            rss_restricted = np.sum((y - model_restricted.predict(X_restricted)) ** 2)
            
            # Unrestricted model: effect ~ effect_lags + cause_lags
            unrestricted_features = restricted_features + [f'cause_lag_{i}' for i in range(1, lag + 1)]
            X_unrestricted = data[unrestricted_features]
            
            model_unrestricted = LinearRegression()
            model_unrestricted.fit(X_unrestricted, y)
            rss_unrestricted = np.sum((y - model_unrestricted.predict(X_unrestricted)) ** 2)
            
            # F-test
            n = len(data)
            k = lag
            f_stat = ((rss_restricted - rss_unrestricted) / k) / (rss_unrestricted / (n - 2*k - 1))
            p_value = 1 - stats.f.cdf(f_stat, k, n - 2*k - 1)
            
            results[f'lag_{lag}'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'is_significant': p_value < 0.05
            }
        
        # Overall result
        min_p_value = min([r['p_value'] for r in results.values()])
        best_lag = [k for k, v in results.items() if v['p_value'] == min_p_value][0]
        
        return {
            'granger_causes': min_p_value < 0.05,
            'best_lag': best_lag,
            'min_p_value': min_p_value,
            'all_lags': results,
            'interpretation': self._interpret_granger(
                cause_series.name, effect_series.name, min_p_value < 0.05, best_lag
            )
        }
    
    def _interpret_granger(
        self,
        cause_name: str,
        effect_name: str,
        granger_causes: bool,
        best_lag: str
    ) -> str:
        """Interpret Granger causality results."""
        if granger_causes:
            return f"{cause_name} Granger-causes {effect_name} (best at {best_lag}). Past values of {cause_name} help predict {effect_name}."
        else:
            return f"{cause_name} does not Granger-cause {effect_name}. Past values of {cause_name} do not improve predictions of {effect_name}."


# Example usage and utilities
class CausalAnalysisPipeline:
    """
    Complete pipeline for causal analysis in demand forecasting.
    """
    
    def __init__(self):
        self.causal_impact = CausalImpactAnalyzer()
        self.did = DifferenceInDifferences()
        self.psm = PropensityScoreMatching()
        self.granger = GrangerCausality()
        
    def analyze_promotion_impact(
        self,
        demand: pd.Series,
        promotion_start: str,
        promotion_end: str,
        pre_period_days: int = 14,
        covariates: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Comprehensive analysis of promotion impact on demand.
        
        Args:
            demand: Time series of demand
            promotion_start: Start date of promotion
            promotion_end: End date of promotion
            pre_period_days: Days before promotion to use as baseline
            covariates: Control variables
            
        Returns:
            Complete causal analysis
        """
        pre_start = (pd.to_datetime(promotion_start) - timedelta(days=pre_period_days)).strftime('%Y-%m-%d')
        
        # Causal impact analysis
        impact_results = self.causal_impact.analyze_impact(
            demand,
            pre_period=(pre_start, promotion_start),
            post_period=(promotion_start, promotion_end),
            covariates=covariates
        )
        
        return {
            'method': 'Causal Impact Analysis',
            'promotion_period': (promotion_start, promotion_end),
            'results': impact_results,
            'summary': impact_results['interpretation']
        }


if __name__ == "__main__":
    print("Causal Inference Module for Demand Forecasting")
    print("=" * 50)
    print("\nAvailable methods:")
    print("1. Causal Impact Analysis - Counterfactual forecasting")
    print("2. Difference-in-Differences - Compare treatment vs control")
    print("3. Propensity Score Matching - Match similar units")
    print("4. Granger Causality - Test predictive relationships")
    print("\nUse cases:")
    print("- Measure true impact of promotions")
    print("- Understand what drives demand changes")
    print("- Separate correlation from causation")
    print("- Optimize marketing interventions")
