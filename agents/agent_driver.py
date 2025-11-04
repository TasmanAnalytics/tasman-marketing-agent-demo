"""Driver Analysis Agent for identifying outcome drivers.

Identifies what drives outcomes like conversion, revenue, or retention using:
- Logistic regression for binary outcomes (CVR)
- Linear regression for continuous outcomes (AOV, revenue)
- Lasso/Ridge regularization for high-dimensional data
- Permutation feature importance
- Cross-validation for model evaluation
"""

import logging
from typing import Dict, Any, List, Optional
import re

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, r2_score

from agents.base_analysis_agent import BaseAnalysisAgent
from core.analysis_utils import CategoricalEncoder, standardize_features


logger = logging.getLogger(__name__)


class DriverAnalysisAgent(BaseAnalysisAgent):
    """Agent for driver analysis and feature importance.

    Handles questions like:
    - "What drives conversion on mobile?"
    - "What are the key drivers of revenue?"
    - "Which features predict high-value customers?"
    """

    # SQL templates for different units of analysis
    SQL_TEMPLATES = {
        'session_level': """
            SELECT
                s.session_id,
                s.device,
                c.channel,
                c.campaign_name,
                s.pages_viewed,
                s.converted_flag::INT AS target
            FROM fact_sessions s
            LEFT JOIN dim_campaigns c ON s.campaign_id = c.campaign_id
            WHERE s.date >= CURRENT_DATE - INTERVAL '{window_days}' DAY
            {filters}
        """,

        'order_level': """
            SELECT
                o.order_id,
                s.device,
                c.channel,
                cu.region,
                cu.segment,
                o.revenue,
                o.margin,
                o.quantity
            FROM fact_orders o
            LEFT JOIN fact_sessions s ON o.session_id = s.session_id
            LEFT JOIN dim_campaigns c ON s.campaign_id = c.campaign_id
            LEFT JOIN dim_customers cu ON o.customer_id = cu.customer_id
            WHERE o.order_timestamp >= NOW() - INTERVAL '{window_days}' DAY
            {filters}
        """,

        'customer_level': """
            WITH cust_agg AS (
                SELECT
                    o.customer_id,
                    COUNT(DISTINCT o.order_id) AS total_orders,
                    SUM(o.revenue) AS total_revenue,
                    SUM(o.margin) AS total_margin,
                    AVG(o.revenue) AS avg_order_value,
                    MAX(o.order_timestamp) AS last_order_date,
                    MIN(o.order_timestamp) AS first_order_date
                FROM fact_orders o
                WHERE o.order_timestamp >= NOW() - INTERVAL '{window_days}' DAY
                GROUP BY o.customer_id
            )
            SELECT
                ca.*,
                cu.region,
                cu.segment,
                cu.first_visit_date,
                DATEDIFF('day', cu.first_visit_date, ca.first_order_date) AS days_to_first_order
            FROM cust_agg ca
            LEFT JOIN dim_customers cu ON ca.customer_id = cu.customer_id
            {filters}
        """
    }

    # Target definitions
    TARGET_TYPES = {
        'converted_flag': 'binary',
        'conversion': 'binary',
        'cvr': 'binary',
        'revenue': 'continuous',
        'margin': 'continuous',
        'aov': 'continuous',
        'total_revenue': 'continuous',
        'total_orders': 'continuous'
    }

    def __init__(self, db_connector, config=None, llm_client=None):
        super().__init__(
            db_connector=db_connector,
            config=config,
            llm_client=llm_client,
            agent_name="DriverAnalysisAgent"
        )

    def _get_max_rows(self) -> int:
        """Get max rows for driver analysis from config."""
        return self.config.get('row_caps', {}).get('modelling_max_rows', 250000)

    def plan(self, question: str, role: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate driver analysis plan from question.

        Args:
            question: User question (e.g., "What drives conversion on mobile?")
            role: User role
            context: Additional context

        Returns:
            Plan dict with keys:
                - target: str (outcome variable)
                - target_type: str (binary | continuous)
                - unit: str (session | order | customer)
                - features: List[str] (predictor variables)
                - filters: Dict[str, Any] (where clauses)
                - window_days: int
                - model_type: str (logistic | linear | lasso | ridge)
                - cv_folds: int
                - sql: str
        """
        logger.info(f"Generating driver analysis plan for: {question}")

        # Extract target variable
        target, target_type = self._extract_target(question)

        # Extract unit of analysis
        unit = self._extract_unit(question, target)

        # Extract filters
        filters = self._extract_filters(question)

        # Extract window
        window_days = self._extract_window(question)

        # Determine features based on unit
        features = self._get_features_for_unit(unit)

        # Determine model type
        model_type = self._determine_model_type(target_type)

        # Get config defaults
        cv_folds = self.config['stats']['cv_folds']
        max_features = self.config['driver'].get('max_features', 50)

        # Build SQL query
        sql = self._build_sql(unit, filters, window_days)

        plan = {
            'version': '1.0',
            'target': target,
            'target_type': target_type,
            'unit': unit,
            'features': features,
            'filters': filters,
            'window_days': window_days,
            'model_type': model_type,
            'cv_folds': cv_folds,
            'max_features': max_features,
            'sql': sql,
            'seed': self.config['defaults']['random_seed'],
            'parameters': {
                'target': target,
                'model_type': model_type,
                'cv_folds': cv_folds
            }
        }

        # Validate plan
        errors = self.validate_plan(plan)
        if errors:
            logger.error(f"Plan validation errors: {errors}")
            raise ValueError(f"Invalid plan: {errors}")

        logger.info(f"Generated plan: target={target}, unit={unit}, model={model_type}")

        return plan

    def _extract_target(self, question: str) -> tuple[str, str]:
        """Extract target variable from question."""
        question_lower = question.lower()

        # Order matters - check more specific patterns first
        target_patterns = [
            ('total_revenue', r'\b(total revenue|lifetime value|ltv)\b', 'continuous'),
            ('total_orders', r'\b(total orders|order count|frequency)\b', 'continuous'),
            ('converted_flag', r'\b(conversion|convert|cvr)\b', 'binary'),
            ('revenue', r'\b(revenue|sales)\b', 'continuous'),
            ('margin', r'\b(margin|profit)\b', 'continuous'),
        ]

        for target, pattern, target_type in target_patterns:
            if re.search(pattern, question_lower):
                return target, target_type

        # Default to conversion
        return 'converted_flag', 'binary'

    def _extract_unit(self, question: str, target: str) -> str:
        """Extract unit of analysis from question."""
        question_lower = question.lower()

        # Explicit unit mentions
        if 'customer' in question_lower or 'user' in question_lower:
            return 'customer'
        elif 'order' in question_lower:
            return 'order'
        elif 'session' in question_lower or 'visit' in question_lower:
            return 'session'

        # Infer from target
        if target in ['converted_flag', 'conversion', 'cvr']:
            return 'session'
        elif target in ['total_revenue', 'total_orders']:
            return 'customer'
        else:
            return 'order'

    def _extract_filters(self, question: str) -> Dict[str, Any]:
        """Extract filters from question."""
        filters = {}
        question_lower = question.lower()

        # Device filter
        devices = ['mobile', 'desktop', 'tablet']
        for device in devices:
            if device in question_lower:
                filters['device'] = device

        # Channel filter
        channels = ['Google', 'Facebook', 'LinkedIn', 'TikTok', 'Instagram']
        for channel in channels:
            if channel.lower() in question_lower:
                filters['channel'] = channel

        # Region filter
        regions = ['US', 'EU', 'APAC', 'LATAM']
        for region in regions:
            if region.lower() in question_lower:
                filters['region'] = region

        return filters

    def _extract_window(self, question: str) -> int:
        """Extract time window from question."""
        question_lower = question.lower()

        # Look for explicit day mentions
        day_match = re.search(r'(\d+)\s*days?', question_lower)
        if day_match:
            return int(day_match.group(1))

        # Look for week/month mentions
        week_match = re.search(r'(\d+)\s*weeks?', question_lower)
        if week_match:
            return int(week_match.group(1)) * 7

        month_match = re.search(r'(\d+)\s*months?', question_lower)
        if month_match:
            return int(month_match.group(1)) * 30

        # Default from config
        return self.config['windows'].get('driver_days', 180)

    def _get_features_for_unit(self, unit: str) -> List[str]:
        """Get feature list based on unit of analysis."""
        if unit == 'session':
            return ['device', 'channel', 'campaign_name', 'pages_viewed']
        elif unit == 'order':
            return ['device', 'channel', 'region', 'segment', 'quantity']
        else:  # customer
            return ['region', 'segment', 'total_orders', 'avg_order_value', 'days_to_first_order']

    def _determine_model_type(self, target_type: str) -> str:
        """Determine model type based on target."""
        if target_type == 'binary':
            return 'logistic'
        else:
            return 'linear'

    def _build_sql(self, unit: str, filters: Dict[str, Any], window_days: int) -> str:
        """Build SQL query for driver analysis."""
        template = self.SQL_TEMPLATES.get(f'{unit}_level', self.SQL_TEMPLATES['session_level'])

        # Build filter clauses
        filter_clauses = []
        if filters:
            for col, val in filters.items():
                if col in ['device', 'channel']:
                    filter_clauses.append(f"AND {col} = '{val}'")
                elif col == 'region':
                    filter_clauses.append(f"AND cu.region = '{val}'")

        filter_str = '\n            '.join(filter_clauses) if filter_clauses else ''

        return template.format(window_days=window_days, filters=filter_str)

    def run(self, df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute driver analysis on data.

        Args:
            df: DataFrame from pull()
            plan: Plan dict from plan()

        Returns:
            Results dict with:
                - model_metrics: Dict (auc/r2, cv scores)
                - drivers: List[Dict] (feature, importance, rank)
                - coefficients: Dict (feature -> coefficient)
                - diagnostics: Dict (sample size, feature stats)
        """
        logger.info(f"Running driver analysis: {plan['model_type']}")

        target = plan['target']
        target_type = plan['target_type']
        features = plan['features']
        model_type = plan['model_type']
        cv_folds = plan['cv_folds']

        # Validate target exists
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found in data. Available: {list(df.columns)}")

        # Extract target
        y = df[target].copy()

        # Filter to available features
        available_features = [f for f in features if f in df.columns]
        logger.info(f"Using {len(available_features)}/{len(features)} available features")

        # Separate categorical and numeric features
        categorical_features = df[available_features].select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_features = df[available_features].select_dtypes(include=[np.number]).columns.tolist()

        # Encode categorical features
        df_encoded = df[available_features].copy()
        if categorical_features:
            encoder = CategoricalEncoder(
                method='onehot',
                max_categories=self.config['driver']['categorical_encoding']['max_categories'],
                min_frequency=self.config['driver']['categorical_encoding']['min_category_frequency']
            )
            df_encoded = encoder.fit_transform(df_encoded, categorical_features)

        # Handle missing values
        df_encoded = df_encoded.fillna(df_encoded.median())

        # Prepare feature matrix
        X = df_encoded

        # Limit features if needed
        if len(X.columns) > plan['max_features']:
            logger.warning(f"Too many features ({len(X.columns)}). Limiting to {plan['max_features']}.")
            # Keep top numeric features by correlation with target
            correlations = X[numeric_features].corrwith(y).abs().sort_values(ascending=False) if numeric_features else pd.Series()
            top_features = correlations.head(plan['max_features']).index.tolist()
            X = X[top_features]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=plan['seed']
        )

        # Select and train model
        if model_type == 'logistic':
            model = LogisticRegression(max_iter=1000, random_state=plan['seed'])
            scoring = 'roc_auc'
        elif model_type == 'lasso':
            model = Lasso(alpha=self.config['driver']['regularization_alpha'], random_state=plan['seed'])
            scoring = 'r2'
        elif model_type == 'ridge':
            model = Ridge(alpha=self.config['driver']['regularization_alpha'], random_state=plan['seed'])
            scoring = 'r2'
        else:  # linear
            model = LinearRegression()
            scoring = 'r2'

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring)

        # Fit final model
        model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test)
        if model_type == 'logistic':
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            test_score = roc_auc_score(y_test, y_pred_proba)
            metric_name = 'auc'
        else:
            test_score = r2_score(y_test, y_pred)
            metric_name = 'r2'

        # Feature importances via permutation
        perm_importance = permutation_importance(
            model, X_test, y_test,
            n_repeats=10,
            random_state=plan['seed'],
            scoring=scoring
        )

        # Build driver table
        drivers = []
        for i, col in enumerate(X.columns):
            importance = perm_importance.importances_mean[i]
            importance_std = perm_importance.importances_std[i]

            drivers.append({
                'feature': col,
                'importance': float(importance),
                'importance_std': float(importance_std),
                'rank': 0  # Will be set after sorting
            })

        # Sort by importance and assign ranks
        drivers = sorted(drivers, key=lambda x: abs(x['importance']), reverse=True)
        for rank, driver in enumerate(drivers, 1):
            driver['rank'] = rank

        # Extract coefficients
        coefficients = {}
        if hasattr(model, 'coef_'):
            coef = model.coef_
            if model_type == 'logistic':
                coef = coef[0]  # Unwrap for logistic regression
            for i, col in enumerate(X.columns):
                coefficients[col] = float(coef[i])

        return {
            'model_type': model_type,
            'model_metrics': {
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                f'test_{metric_name}': float(test_score),
                'metric': metric_name
            },
            'drivers': drivers[:20],  # Top 20
            'coefficients': coefficients,
            'diagnostics': {
                'n_samples': len(df),
                'n_features': len(X.columns),
                'n_categorical': len(categorical_features),
                'n_numeric': len(numeric_features),
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
        }

    def report(self, results: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report with insights from driver analysis.

        Args:
            results: Results dict from run()
            plan: Original plan dict

        Returns:
            Report dict with summary, insights, caveats, next_actions
        """
        model_type = results['model_type']
        metrics = results['model_metrics']
        drivers = results['drivers']

        metric_name = metrics['metric']
        test_score = metrics[f'test_{metric_name}']
        cv_mean = metrics['cv_mean']

        # Build summary
        summary = (
            f"Analyzed {results['diagnostics']['n_samples']} {plan['unit']}s "
            f"with {results['diagnostics']['n_features']} features. "
            f"Model: {model_type}, "
            f"CV {metric_name.upper()}: {cv_mean:.3f}, "
            f"Test {metric_name.upper()}: {test_score:.3f}."
        )

        # Build insights
        insights = []

        # Model performance
        if metric_name == 'auc':
            if test_score >= 0.8:
                insights.append(f"Strong predictive performance (AUC={test_score:.3f}).")
            elif test_score >= 0.7:
                insights.append(f"Moderate predictive performance (AUC={test_score:.3f}).")
            else:
                insights.append(f"Limited predictive performance (AUC={test_score:.3f}). Consider feature engineering.")
        else:
            if test_score >= 0.5:
                insights.append(f"Model explains {test_score:.1%} of variance (R²={test_score:.3f}).")
            else:
                insights.append(f"Model explains limited variance (R²={test_score:.3f}). Consider additional features.")

        # Top drivers
        if drivers:
            top_3 = drivers[:3]
            driver_names = [d['feature'] for d in top_3]
            insights.append(f"Top drivers: {', '.join(driver_names)}.")

            # Feature-specific insights
            if results.get('coefficients'):
                for driver in top_3:
                    coef = results['coefficients'].get(driver['feature'])
                    if coef:
                        direction = "increases" if coef > 0 else "decreases"
                        insights.append(
                            f"{driver['feature']} {direction} {plan['target']} "
                            f"(coefficient: {coef:+.3f})."
                        )

        # Caveats
        caveats = []

        # Sample size
        if results['diagnostics']['n_samples'] < 1000:
            caveats.append("Small sample size may limit model reliability.")

        # Model performance
        if test_score < 0.65 and metric_name == 'auc':
            caveats.append("Low AUC suggests weak predictive signal. Results may not be actionable.")

        # Overfitting check
        if cv_mean - test_score > 0.1:
            caveats.append("Model may be overfitting (CV score >> test score).")

        # Feature count
        if results['diagnostics']['n_features'] < 3:
            caveats.append("Limited features may miss important drivers.")

        # Next actions
        next_actions = []

        if test_score >= 0.7:
            next_actions.append("Investigate top drivers in detail with hypothesis tests.")
            next_actions.append("Segment analysis by top driver to find subgroups.")
        else:
            next_actions.append("Add more features (time-based, interaction terms).")
            next_actions.append("Check for data quality issues or missing values.")

        next_actions.append("Validate findings with A/B test if possible.")

        return {
            'summary': summary,
            'insights': insights,
            'caveats': caveats,
            'next_actions': next_actions,
            'raw_results': results
        }
