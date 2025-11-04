"""Segmentation Agent for customer/campaign clustering.

Creates actionable segments using:
- K-means clustering with optimal k selection
- RFM (Recency, Frequency, Monetary) segmentation
- DBSCAN for anomaly detection
- Silhouette scoring for cluster quality
"""

import logging
from typing import Dict, Any, List, Optional
import re

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from agents.base_analysis_agent import BaseAnalysisAgent
from core.analysis_utils import winsorize, detect_outliers_iqr


logger = logging.getLogger(__name__)


class SegmentationAgent(BaseAnalysisAgent):
    """Agent for customer and campaign segmentation.

    Handles questions like:
    - "Segment customers by value"
    - "Find customer segments for targeting"
    - "Group campaigns by performance"
    """

    # SQL templates for different units
    SQL_TEMPLATES = {
        'customer': """
            WITH cust_orders AS (
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
                co.*,
                cu.region,
                cu.segment AS original_segment,
                DATEDIFF('day', co.last_order_date, NOW()) AS recency_days,
                DATEDIFF('day', co.first_order_date, co.last_order_date) AS customer_age_days
            FROM cust_orders co
            LEFT JOIN dim_customers cu ON co.customer_id = cu.customer_id
        """,

        'campaign': """
            WITH camp_metrics AS (
                SELECT
                    f.campaign_id,
                    c.campaign_name,
                    c.channel,
                    SUM(f.spend) AS total_spend,
                    SUM(f.impressions) AS total_impressions,
                    SUM(f.clicks) AS total_clicks,
                    CASE WHEN SUM(f.impressions) > 0
                         THEN SUM(f.clicks)::DOUBLE / SUM(f.impressions)
                         ELSE 0 END AS ctr
                FROM fact_ad_spend f
                JOIN dim_campaigns c ON f.campaign_id = c.campaign_id
                WHERE f.date >= CURRENT_DATE - INTERVAL '{window_days}' DAY
                GROUP BY f.campaign_id, c.campaign_name, c.channel
            ),
            camp_conversions AS (
                SELECT
                    s.campaign_id,
                    COUNT(*) AS sessions,
                    SUM(CASE WHEN s.converted_flag THEN 1 ELSE 0 END) AS conversions,
                    CASE WHEN COUNT(*) > 0
                         THEN SUM(CASE WHEN s.converted_flag THEN 1 ELSE 0 END)::DOUBLE / COUNT(*)
                         ELSE 0 END AS cvr
                FROM fact_sessions s
                WHERE s.date >= CURRENT_DATE - INTERVAL '{window_days}' DAY
                GROUP BY s.campaign_id
            )
            SELECT
                cm.*,
                COALESCE(cc.sessions, 0) AS sessions,
                COALESCE(cc.conversions, 0) AS conversions,
                COALESCE(cc.cvr, 0) AS cvr
            FROM camp_metrics cm
            LEFT JOIN camp_conversions cc ON cm.campaign_id = cc.campaign_id
        """
    }

    def __init__(self, db_connector, config=None, llm_client=None):
        super().__init__(
            db_connector=db_connector,
            config=config,
            llm_client=llm_client,
            agent_name="SegmentationAgent"
        )

    def _get_max_rows(self) -> int:
        """Get max rows for segmentation from config."""
        return self.config.get('row_caps', {}).get('segmentation_max_rows', 200000)

    def plan(self, question: str, role: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate segmentation plan from question.

        Args:
            question: User question (e.g., "Segment customers by value")
            role: User role
            context: Additional context

        Returns:
            Plan dict with keys:
                - method: str (kmeans | rfm | dbscan)
                - unit: str (customer | campaign)
                - features: List[str] (clustering features)
                - k: Optional[int] (number of clusters for k-means)
                - k_range: List[int] (range for optimal k search)
                - standardize: bool
                - outlier_handling: str (winsorize | clip | remove)
                - sql: str
        """
        logger.info(f"Generating segmentation plan for: {question}")

        # Extract method
        method = self._extract_method(question)

        # Extract unit
        unit = self._extract_unit(question)

        # Extract features
        features = self._get_features_for_unit(unit, method)

        # Extract k if specified
        k = self._extract_k(question)

        # Get config defaults
        kmeans_config = self.config['segmentation']['kmeans']
        k_range = [kmeans_config['min_k'], kmeans_config['max_k']]

        standardize = self.config['segmentation'].get('standardize', True)
        outlier_handling = self.config['segmentation'].get('outlier_handling', 'winsorize')
        window_days = self._extract_window(question)

        # Build SQL query
        sql = self._build_sql(unit, window_days)

        plan = {
            'version': '1.0',
            'method': method,
            'unit': unit,
            'features': features,
            'k': k,
            'k_range': k_range,
            'standardize': standardize,
            'outlier_handling': outlier_handling,
            'window_days': window_days,
            'sql': sql,
            'seed': self.config['defaults']['random_seed'],
            'parameters': {
                'method': method,
                'unit': unit,
                'k': k
            }
        }

        # Validate plan
        errors = self.validate_plan(plan)
        if errors:
            logger.error(f"Plan validation errors: {errors}")
            raise ValueError(f"Invalid plan: {errors}")

        logger.info(f"Generated plan: method={method}, unit={unit}, k={k}")

        return plan

    def _extract_method(self, question: str) -> str:
        """Extract segmentation method from question."""
        question_lower = question.lower()

        if 'rfm' in question_lower or 'recency' in question_lower or 'frequency' in question_lower:
            return 'rfm'
        elif 'anomal' in question_lower or 'outlier' in question_lower:
            return 'dbscan'
        else:
            return 'kmeans'

    def _extract_unit(self, question: str) -> str:
        """Extract unit of analysis from question."""
        question_lower = question.lower()

        if 'campaign' in question_lower:
            return 'campaign'
        else:
            return 'customer'

    def _extract_k(self, question: str) -> Optional[int]:
        """Extract number of clusters from question."""
        # Look for patterns like "3 segments", "5 customer segments", "k=4"
        patterns = [
            r'(\d+)\s+\w*\s*(segment|cluster|group)',  # Handles "5 customer segments"
            r'k\s*=\s*(\d+)',
            r'into\s+(\d+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, question.lower())
            if match:
                return int(match.group(1))

        return None  # Auto-select optimal k

    def _extract_window(self, question: str) -> int:
        """Extract time window from question."""
        question_lower = question.lower()

        day_match = re.search(r'(\d+)\s*days?', question_lower)
        if day_match:
            return int(day_match.group(1))

        return self.config['windows'].get('segmentation_days', 180)

    def _get_features_for_unit(self, unit: str, method: str) -> List[str]:
        """Get feature list based on unit and method."""
        if unit == 'customer':
            if method == 'rfm':
                return ['recency_days', 'total_orders', 'total_revenue']
            else:
                return ['total_orders', 'total_revenue', 'avg_order_value', 'customer_age_days']
        else:  # campaign
            return ['total_spend', 'ctr', 'cvr', 'conversions']

    def _build_sql(self, unit: str, window_days: int) -> str:
        """Build SQL query for segmentation."""
        template = self.SQL_TEMPLATES.get(unit, self.SQL_TEMPLATES['customer'])
        return template.format(window_days=window_days)

    def run(self, df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute segmentation on data.

        Args:
            df: DataFrame from pull()
            plan: Plan dict from plan()

        Returns:
            Results dict with:
                - method: str
                - k: int (actual number of segments)
                - segments: List[Dict] (segment profiles)
                - silhouette_score: float (for kmeans)
                - diagnostics: Dict
        """
        logger.info(f"Running segmentation: {plan['method']}")

        method = plan['method']
        features = plan['features']

        # Filter to available features
        available_features = [f for f in features if f in df.columns]
        if not available_features:
            raise ValueError(f"No features available from {features}. Columns: {list(df.columns)}")

        logger.info(f"Using {len(available_features)}/{len(features)} available features")

        # Prepare feature matrix
        X = df[available_features].copy()

        # Handle outliers
        if plan.get('outlier_handling') == 'winsorize':
            for col in available_features:
                limits = self.config['segmentation'].get('winsorize_limits', [0.01, 0.99])
                X[col] = winsorize(X[col], limits=tuple(limits))

        # Handle missing values
        X = X.fillna(X.median())

        # Standardize if needed
        if plan.get('standardize', True):
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X

        # Run segmentation method
        if method == 'kmeans':
            result = self._run_kmeans(df, X_scaled, plan)
        elif method == 'rfm':
            result = self._run_rfm(df, X, plan)
        elif method == 'dbscan':
            result = self._run_dbscan(df, X_scaled, plan)
        else:
            raise ValueError(f"Unsupported method: {method}")

        return result

    def _run_kmeans(self, df: pd.DataFrame, X_scaled: pd.DataFrame,
                   plan: Dict[str, Any]) -> Dict[str, Any]:
        """Run K-means clustering."""
        k = plan.get('k')
        k_range = plan.get('k_range', [2, 10])

        # Find optimal k if not specified
        if k is None:
            silhouette_scores = []
            k_values = range(k_range[0], k_range[1] + 1)

            for k_test in k_values:
                kmeans = KMeans(n_clusters=k_test, random_state=plan['seed'], n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, labels)
                silhouette_scores.append(score)

            best_idx = np.argmax(silhouette_scores)
            k = k_values[best_idx]
            logger.info(f"Optimal k selected: {k} (silhouette={silhouette_scores[best_idx]:.3f})")

        # Fit with selected k
        kmeans = KMeans(n_clusters=k, random_state=plan['seed'], n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        silhouette = silhouette_score(X_scaled, labels)

        # Profile segments
        df_with_segments = df.copy()
        df_with_segments['segment'] = labels

        segments = []
        for seg_id in range(k):
            seg_df = df_with_segments[df_with_segments['segment'] == seg_id]

            # Calculate profile statistics
            profile = {
                'segment_id': int(seg_id),
                'size': len(seg_df),
                'share': float(len(seg_df) / len(df)),
                'profile': {}
            }

            # Add feature statistics
            for col in plan['features']:
                if col in df.columns:
                    profile['profile'][col] = {
                        'mean': float(seg_df[col].mean()),
                        'median': float(seg_df[col].median()),
                        'std': float(seg_df[col].std())
                    }

            segments.append(profile)

        # Sort segments by size
        segments = sorted(segments, key=lambda x: x['size'], reverse=True)

        return {
            'method': 'kmeans',
            'k': int(k),
            'silhouette_score': float(silhouette),
            'segments': segments,
            'diagnostics': {
                'n_samples': len(df),
                'n_features': len(plan['features']),
                'k_range_tested': k_range if plan.get('k') is None else None
            }
        }

    def _run_rfm(self, df: pd.DataFrame, X: pd.DataFrame,
                plan: Dict[str, Any]) -> Dict[str, Any]:
        """Run RFM segmentation."""
        # RFM requires specific columns
        if 'recency_days' not in df.columns or 'total_orders' not in df.columns or 'total_revenue' not in df.columns:
            raise ValueError("RFM requires recency_days, total_orders, and total_revenue columns")

        df_rfm = df.copy()

        # Create RFM scores (1-5, lower recency is better)
        df_rfm['R_score'] = pd.qcut(df_rfm['recency_days'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
        df_rfm['F_score'] = pd.qcut(df_rfm['total_orders'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        df_rfm['M_score'] = pd.qcut(df_rfm['total_revenue'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')

        # Create RFM segment label
        df_rfm['RFM_segment'] = (
            df_rfm['R_score'].astype(str) +
            df_rfm['F_score'].astype(str) +
            df_rfm['M_score'].astype(str)
        )

        # Profile segments
        segment_profiles = df_rfm.groupby('RFM_segment').agg({
            'customer_id': 'count',
            'total_orders': 'mean',
            'total_revenue': 'mean',
            'recency_days': 'mean'
        }).reset_index()

        segment_profiles.columns = ['segment', 'size', 'avg_orders', 'avg_revenue', 'avg_recency']

        # Sort by value (revenue)
        segment_profiles = segment_profiles.sort_values('avg_revenue', ascending=False)

        # Convert to segment list
        segments = []
        for _, row in segment_profiles.head(20).iterrows():  # Top 20 segments
            segments.append({
                'segment_id': row['segment'],
                'size': int(row['size']),
                'share': float(row['size'] / len(df)),
                'profile': {
                    'avg_orders': float(row['avg_orders']),
                    'avg_revenue': float(row['avg_revenue']),
                    'avg_recency': float(row['avg_recency'])
                }
            })

        return {
            'method': 'rfm',
            'k': len(segment_profiles),
            'segments': segments,
            'diagnostics': {
                'n_samples': len(df),
                'n_segments': len(segment_profiles),
                'top_segments_shown': min(20, len(segment_profiles))
            }
        }

    def _run_dbscan(self, df: pd.DataFrame, X_scaled: pd.DataFrame,
                   plan: Dict[str, Any]) -> Dict[str, Any]:
        """Run DBSCAN for anomaly detection."""
        eps = self.config['segmentation']['dbscan'].get('eps', 0.5)
        min_samples = self.config['segmentation']['dbscan'].get('min_samples', 10)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)

        # Separate core, border, and noise points
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        # Profile each cluster
        df_with_labels = df.copy()
        df_with_labels['cluster'] = labels

        segments = []
        for cluster_id in set(labels):
            if cluster_id == -1:
                label = 'Outliers'
            else:
                label = f'Cluster_{cluster_id}'

            cluster_df = df_with_labels[df_with_labels['cluster'] == cluster_id]

            profile = {
                'segment_id': label,
                'size': len(cluster_df),
                'share': float(len(cluster_df) / len(df)),
                'profile': {}
            }

            # Add feature statistics
            for col in plan['features']:
                if col in df.columns:
                    profile['profile'][col] = {
                        'mean': float(cluster_df[col].mean()),
                        'median': float(cluster_df[col].median())
                    }

            segments.append(profile)

        # Sort by size
        segments = sorted(segments, key=lambda x: x['size'], reverse=True)

        return {
            'method': 'dbscan',
            'k': n_clusters,
            'n_noise': n_noise,
            'segments': segments,
            'diagnostics': {
                'n_samples': len(df),
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'eps': eps,
                'min_samples': min_samples
            }
        }

    def report(self, results: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report with insights from segmentation.

        Args:
            results: Results dict from run()
            plan: Original plan dict

        Returns:
            Report dict with summary, insights, caveats, next_actions
        """
        method = results['method']
        k = results['k']
        segments = results['segments']

        # Build summary
        if method == 'rfm':
            summary = f"Created {k} RFM segments from {results['diagnostics']['n_samples']} {plan['unit']}s."
        elif method == 'dbscan':
            summary = (
                f"Found {k} clusters using DBSCAN. "
                f"{results.get('n_noise', 0)} outliers detected "
                f"({results.get('n_noise', 0) / results['diagnostics']['n_samples']:.1%})."
            )
        else:  # kmeans
            summary = (
                f"Created {k} segments from {results['diagnostics']['n_samples']} {plan['unit']}s. "
                f"Silhouette score: {results.get('silhouette_score', 0):.3f}."
            )

        # Build insights
        insights = []

        # Segment quality
        if method == 'kmeans':
            silhouette = results.get('silhouette_score', 0)
            if silhouette >= 0.5:
                insights.append("High-quality segmentation (silhouette ≥ 0.5). Segments are well-separated.")
            elif silhouette >= 0.3:
                insights.append("Moderate segmentation quality (silhouette ≥ 0.3). Some overlap between segments.")
            else:
                insights.append("Low segmentation quality (silhouette < 0.3). Consider fewer segments or different features.")

        # Largest segments
        if segments:
            top_3 = segments[:3]
            segment_labels = [f"{s['segment_id']} ({s['share']:.1%})" for s in top_3]
            insights.append(f"Top 3 segments by size: {', '.join(segment_labels)}.")

            # Segment-specific insights
            if method == 'rfm' and segments:
                best_segment = segments[0]
                insights.append(
                    f"Best segment: {best_segment['segment_id']} - "
                    f"Avg revenue: ${best_segment['profile']['avg_revenue']:.2f}, "
                    f"Avg orders: {best_segment['profile']['avg_orders']:.1f}."
                )

        # Caveats
        caveats = []

        if results['diagnostics']['n_samples'] < 500:
            caveats.append("Small sample size may result in unstable segments.")

        if method == 'kmeans' and results.get('silhouette_score', 0) < 0.3:
            caveats.append("Low silhouette score suggests segments may not be meaningful.")

        if method == 'dbscan' and results.get('n_noise', 0) / results['diagnostics']['n_samples'] > 0.2:
            caveats.append("High outlier rate (>20%) suggests eps parameter may be too small.")

        # Next actions
        next_actions = []

        next_actions.append("Profile segments in detail to understand characteristics.")
        next_actions.append("Design targeted campaigns for high-value segments.")

        if method == 'kmeans':
            next_actions.append("Validate segment stability over time.")

        if method == 'rfm':
            next_actions.append("Create personalized re-engagement campaigns for low recency segments.")

        return {
            'summary': summary,
            'insights': insights,
            'caveats': caveats,
            'next_actions': next_actions,
            'raw_results': results
        }
