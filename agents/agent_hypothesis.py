"""Hypothesis Testing Agent for statistical comparisons.

Tests targeted assertions like "LinkedIn CTR > Facebook CTR" using:
- Two-sample z-test for proportions (CTR, CVR)
- Welch's t-test for continuous metrics (AOV, margin)
- Chi-squared test for >2 groups
- ANOVA for continuous metrics with >2 groups
- Benjamini-Hochberg FDR correction for multiple comparisons
"""

import logging
from typing import Dict, Any, List, Optional
import re

import pandas as pd
import numpy as np

from agents.base_analysis_agent import BaseAnalysisAgent
from core.analysis_utils import (
    proportion_test,
    welch_ttest,
    benjamini_hochberg,
    bootstrap_ci
)


logger = logging.getLogger(__name__)


class HypothesisTestingAgent(BaseAnalysisAgent):
    """Agent for hypothesis testing and statistical comparisons.

    Handles questions like:
    - "Is LinkedIn CTR better than Facebook CTR?"
    - "Compare conversion rates across all channels"
    - "Is mobile AOV different from desktop AOV?"
    """

    # SQL templates for different metrics
    SQL_TEMPLATES = {
        'ctr_by_dimension': """
            SELECT {dimension},
                   SUM(clicks) AS clicks,
                   SUM(impressions) AS impressions,
                   CASE WHEN SUM(impressions) = 0 THEN NULL
                        ELSE SUM(clicks)::DOUBLE / SUM(impressions) END AS ctr
            FROM fact_ad_spend f
            JOIN dim_campaigns c ON f.campaign_id = c.campaign_id
            WHERE f.date >= CURRENT_DATE - INTERVAL '{window_days}' DAY
            GROUP BY 1
            ORDER BY 1
        """,

        'cvr_by_dimension': """
            SELECT {dimension},
                   COUNT(*) AS sessions,
                   SUM(CASE WHEN converted_flag THEN 1 ELSE 0 END) AS conversions,
                   CASE WHEN COUNT(*) = 0 THEN NULL
                        ELSE SUM(CASE WHEN converted_flag THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) END AS cvr
            FROM fact_sessions s
            LEFT JOIN dim_campaigns c ON s.campaign_id = c.campaign_id
            WHERE s.date >= CURRENT_DATE - INTERVAL '{window_days}' DAY
            GROUP BY 1
            ORDER BY 1
        """,

        'aov_by_dimension': """
            SELECT {dimension},
                   COUNT(DISTINCT o.order_id) AS orders,
                   AVG(o.revenue) AS aov,
                   STDDEV(o.revenue) AS aov_std
            FROM fact_orders o
            LEFT JOIN fact_sessions s ON o.session_id = s.session_id
            LEFT JOIN dim_campaigns c ON s.campaign_id = c.campaign_id
            WHERE o.order_timestamp >= NOW() - INTERVAL '{window_days}' DAY
            GROUP BY 1
            ORDER BY 1
        """,

        'margin_by_dimension': """
            SELECT {dimension},
                   COUNT(DISTINCT o.order_id) AS orders,
                   AVG(o.margin) AS avg_margin,
                   STDDEV(o.margin) AS margin_std
            FROM fact_orders o
            LEFT JOIN fact_sessions s ON o.session_id = s.session_id
            LEFT JOIN dim_campaigns c ON s.campaign_id = c.campaign_id
            WHERE o.order_timestamp >= NOW() - INTERVAL '{window_days}' DAY
            GROUP BY 1
            ORDER BY 1
        """,

        # Row-level query for distribution-based tests
        'continuous_row_level': """
            SELECT {dimension},
                   o.{metric} AS value
            FROM fact_orders o
            LEFT JOIN fact_sessions s ON o.session_id = s.session_id
            LEFT JOIN dim_campaigns c ON s.campaign_id = c.campaign_id
            WHERE o.order_timestamp >= NOW() - INTERVAL '{window_days}' DAY
            ORDER BY 1
        """
    }

    # Metric definitions
    METRIC_TYPES = {
        'ctr': 'proportion',
        'cvr': 'proportion',
        'conversion_rate': 'proportion',
        'aov': 'continuous',
        'revenue': 'continuous',
        'margin': 'continuous'
    }

    def __init__(self, db_connector, config=None, llm_client=None):
        super().__init__(
            db_connector=db_connector,
            config=config,
            llm_client=llm_client,
            agent_name="HypothesisTestingAgent"
        )

    def _get_max_rows(self) -> int:
        """Get max rows for hypothesis testing from config."""
        return self.config.get('row_caps', {}).get('hypothesis_max_rows', 100000)

    def plan(self, question: str, role: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate hypothesis test plan from question.

        Args:
            question: User question (e.g., "Is LinkedIn CTR better than Facebook CTR?")
            role: User role
            context: Additional context (schema, previous results)

        Returns:
            Plan dict with keys:
                - metric: str (ctr, cvr, aov, margin)
                - groups: List[str] (group values to compare)
                - dimension: str (channel, device, region)
                - window_days: int
                - test_type: str (auto, z_test, t_test, chi_squared, anova)
                - alternative: str (two_sided, greater, less)
                - alpha: float
                - corrections: bool
                - sql: str
        """
        logger.info(f"Generating hypothesis test plan for: {question}")

        # Extract metric
        metric = self._extract_metric(question)

        # Extract dimension
        dimension = self._extract_dimension(question)

        # Extract groups
        groups = self._extract_groups(question, dimension)

        # Extract window
        window_days = self._extract_window(question)

        # Extract directionality
        alternative = self._extract_alternative(question)

        # Determine test type
        metric_type = self.METRIC_TYPES.get(metric, 'continuous')
        test_type = self._determine_test_type(metric_type, len(groups))

        # Get config defaults
        alpha = self.config['stats']['alpha']
        corrections = self.config['hypothesis'].get('multiple_comparison_correction', True)

        # Build SQL query
        sql = self._build_sql(metric, dimension, window_days, metric_type, len(groups))

        plan = {
            'version': '1.0',
            'metric': metric,
            'metric_type': metric_type,
            'groups': groups,
            'dimension': dimension,
            'window_days': window_days,
            'test_type': test_type,
            'alternative': alternative,
            'alpha': alpha,
            'corrections': corrections and len(groups) > 2,
            'sql': sql,
            'seed': self.config['defaults']['random_seed'],
            'parameters': {
                'metric': metric,
                'test_type': test_type,
                'alpha': alpha
            }
        }

        # Validate plan
        errors = self.validate_plan(plan)
        if errors:
            logger.error(f"Plan validation errors: {errors}")
            raise ValueError(f"Invalid plan: {errors}")

        logger.info(f"Generated plan: metric={metric}, groups={groups}, test_type={test_type}")

        return plan

    def _extract_metric(self, question: str) -> str:
        """Extract metric from question."""
        question_lower = question.lower()

        metric_patterns = {
            'ctr': r'\b(ctr|click[\s-]?through|click rate)\b',
            'cvr': r'\b(cvr|conversion rate|conversion)\b',
            'aov': r'\b(aov|average order value|order value)\b',
            'margin': r'\b(margin|profit)\b',
            'revenue': r'\b(revenue|sales)\b'
        }

        for metric, pattern in metric_patterns.items():
            if re.search(pattern, question_lower):
                return metric

        # Default to CVR if mentions "conversion"
        if 'convert' in question_lower:
            return 'cvr'

        # Default to CTR
        return 'ctr'

    def _extract_dimension(self, question: str) -> str:
        """Extract dimension from question."""
        question_lower = question.lower()

        dimension_patterns = {
            'c.channel': r'\b(channel|platform|source)\b',
            's.device': r'\b(device|mobile|desktop|tablet)\b',
            'cu.region': r'\b(region|country|geo|geography)\b',
            'c.campaign_name': r'\b(campaign)\b'
        }

        for dimension, pattern in dimension_patterns.items():
            if re.search(pattern, question_lower):
                return dimension

        # Default to channel
        return 'c.channel'

    def _extract_groups(self, question: str, dimension: str) -> List[str]:
        """Extract group values from question."""
        # Common values
        channels = ['Google', 'Facebook', 'LinkedIn', 'TikTok', 'Instagram', 'Twitter']
        devices = ['mobile', 'desktop', 'tablet']
        regions = ['US', 'EU', 'APAC', 'LATAM']

        groups_found = []

        if 'channel' in dimension.lower():
            for channel in channels:
                if channel.lower() in question.lower():
                    groups_found.append(channel)
        elif 'device' in dimension.lower():
            for device in devices:
                if device.lower() in question.lower():
                    groups_found.append(device)
        elif 'region' in dimension.lower():
            for region in regions:
                if region.lower() in question.lower():
                    groups_found.append(region)

        # If no specific groups mentioned, return empty list (will compare all groups)
        if not groups_found:
            logger.warning(f"No specific groups found in question. Will compare all groups in {dimension}.")

        return groups_found

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

        # Look for relative time
        if 'last week' in question_lower:
            return 7
        elif 'last month' in question_lower:
            return 30
        elif 'last quarter' in question_lower:
            return 90

        # Default from config
        return self.config['windows'].get('hypothesis_days', 90)

    def _extract_alternative(self, question: str) -> str:
        """Extract directionality from question."""
        question_lower = question.lower()

        # Look for directional keywords
        if any(word in question_lower for word in ['better', 'higher', 'greater', 'more', 'increase']):
            return 'greater'
        elif any(word in question_lower for word in ['worse', 'lower', 'less', 'fewer', 'decrease']):
            return 'less'
        else:
            return 'two-sided'

    def _determine_test_type(self, metric_type: str, n_groups: int) -> str:
        """Determine appropriate statistical test."""
        if metric_type == 'proportion':
            if n_groups == 2:
                return 'z_test'
            else:
                return 'chi_squared'
        else:  # continuous
            if n_groups == 2:
                return 't_test'
            else:
                return 'anova'

    def _build_sql(self, metric: str, dimension: str, window_days: int,
                   metric_type: str, n_groups: int) -> str:
        """Build SQL query for hypothesis test."""
        # For continuous metrics with >2 groups, need row-level data
        if metric_type == 'continuous' and n_groups != 2:
            # For ANOVA, we need row-level values
            template = self.SQL_TEMPLATES.get('continuous_row_level', '')
            return template.format(
                dimension=dimension,
                metric=metric,
                window_days=window_days
            )

        # Otherwise use aggregated templates
        if metric == 'ctr':
            template = self.SQL_TEMPLATES['ctr_by_dimension']
        elif metric in ['cvr', 'conversion_rate']:
            template = self.SQL_TEMPLATES['cvr_by_dimension']
        elif metric == 'aov':
            template = self.SQL_TEMPLATES['aov_by_dimension']
        elif metric in ['margin', 'revenue']:
            template = self.SQL_TEMPLATES['margin_by_dimension']
        else:
            template = self.SQL_TEMPLATES['ctr_by_dimension']  # fallback

        return template.format(
            dimension=dimension,
            window_days=window_days
        )

    def run(self, df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute hypothesis test on data.

        Args:
            df: DataFrame from pull()
            plan: Plan dict from plan()

        Returns:
            Results dict with:
                - test_type: str
                - p_value: float
                - effect: Dict (abs, rel)
                - confidence_interval: Tuple[float, float]
                - group_stats: List[Dict]
                - pairwise: Optional[Dict] (for >2 groups)
        """
        logger.info(f"Running hypothesis test: {plan['test_type']}")

        metric = plan['metric']
        metric_type = plan['metric_type']
        groups = plan['groups']
        dimension = plan['dimension']
        test_type = plan['test_type']

        # If no specific groups, use all groups in data
        if not groups:
            groups = df[dimension.split('.')[-1] if '.' in dimension else dimension].unique().tolist()
            logger.info(f"No specific groups in plan. Using all groups from data: {groups}")

        # Filter to requested groups
        dim_col = dimension.split('.')[-1] if '.' in dimension else dimension
        df_filtered = df[df[dim_col].isin(groups)] if groups else df

        # Run appropriate test
        if test_type == 'z_test':
            result = self._run_proportion_test(df_filtered, plan, dim_col)
        elif test_type == 't_test':
            result = self._run_t_test(df_filtered, plan, dim_col)
        elif test_type == 'chi_squared':
            result = self._run_chi_squared(df_filtered, plan, dim_col)
        elif test_type == 'anova':
            result = self._run_anova(df_filtered, plan, dim_col)
        else:
            raise ValueError(f"Unsupported test type: {test_type}")

        return result

    def _run_proportion_test(self, df: pd.DataFrame, plan: Dict[str, Any],
                            dim_col: str) -> Dict[str, Any]:
        """Run two-sample proportion test."""
        groups = plan['groups']
        metric = plan['metric']

        # Extract group data
        group1_row = df[df[dim_col] == groups[0]].iloc[0]
        group2_row = df[df[dim_col] == groups[1]].iloc[0]

        if metric in ['ctr', 'conversion_rate']:
            successes1 = int(group1_row['clicks'] if 'clicks' in group1_row else group1_row['conversions'])
            trials1 = int(group1_row['impressions'] if 'impressions' in group1_row else group1_row['sessions'])
            successes2 = int(group2_row['clicks'] if 'clicks' in group2_row else group2_row['conversions'])
            trials2 = int(group2_row['impressions'] if 'impressions' in group2_row else group2_row['sessions'])
        else:  # cvr
            successes1 = int(group1_row['conversions'])
            trials1 = int(group1_row['sessions'])
            successes2 = int(group2_row['conversions'])
            trials2 = int(group2_row['sessions'])

        # Run test
        test_result = proportion_test(
            successes1, trials1,
            successes2, trials2,
            alternative=plan.get('alternative', 'two-sided')
        )

        # Bootstrap CI for difference
        p1 = successes1 / trials1 if trials1 > 0 else 0
        p2 = successes2 / trials2 if trials2 > 0 else 0

        return {
            'test_type': 'z_test',
            'p_value': test_result['p_value'],
            'z_stat': test_result['z_stat'],
            'effect': {
                'abs': test_result['diff'],
                'rel': test_result['relative_lift']
            },
            'group_stats': [
                {
                    'group': groups[0],
                    'rate': p1,
                    'successes': successes1,
                    'trials': trials1
                },
                {
                    'group': groups[1],
                    'rate': p2,
                    'successes': successes2,
                    'trials': trials2
                }
            ],
            'confidence_level': plan.get('alpha', 0.05)
        }

    def _run_t_test(self, df: pd.DataFrame, plan: Dict[str, Any],
                   dim_col: str) -> Dict[str, Any]:
        """Run Welch's t-test for continuous metrics."""
        groups = plan['groups']
        metric = plan['metric']

        # Get group data
        group1_data = df[df[dim_col] == groups[0]][metric].dropna().values
        group2_data = df[df[dim_col] == groups[1]][metric].dropna().values

        # Run test
        test_result = welch_ttest(
            group1_data,
            group2_data,
            alternative=plan.get('alternative', 'two-sided')
        )

        # Calculate confidence interval for difference
        from core.analysis_utils import compute_confidence_interval
        ci_lower, ci_upper = compute_confidence_interval(group1_data - np.mean(group1_data) + np.mean(group2_data))

        # Calculate standard deviations
        std1 = np.std(group1_data, ddof=1)
        std2 = np.std(group2_data, ddof=1)

        return {
            'test_type': 't_test',
            'p_value': test_result['p_value'],
            't_stat': test_result['t_stat'],
            'effect': {
                'abs': test_result['diff'],
                'rel': test_result['diff'] / test_result['mean2'] if test_result['mean2'] != 0 else 0
            },
            'confidence_interval': (ci_lower, ci_upper),
            'group_stats': [
                {
                    'group': groups[0],
                    'mean': test_result['mean1'],
                    'std': std1,
                    'n': len(group1_data)
                },
                {
                    'group': groups[1],
                    'mean': test_result['mean2'],
                    'std': std2,
                    'n': len(group2_data)
                }
            ],
            'equal_variance': bool(test_result.get('equal_variance', True)),
            'confidence_level': plan.get('alpha', 0.05)
        }

    def _run_chi_squared(self, df: pd.DataFrame, plan: Dict[str, Any],
                        dim_col: str) -> Dict[str, Any]:
        """Run chi-squared test for proportions with >2 groups."""
        from scipy.stats import chi2_contingency

        metric = plan['metric']
        groups = plan['groups']

        # Build contingency table
        if metric in ['ctr', 'conversion_rate']:
            successes = df['clicks'] if 'clicks' in df.columns else df['conversions']
            trials = df['impressions'] if 'impressions' in df.columns else df['sessions']
        else:  # cvr
            successes = df['conversions']
            trials = df['sessions']

        failures = trials - successes

        contingency_table = pd.DataFrame({
            'successes': successes.values,
            'failures': failures.values
        }, index=df[dim_col].values).T

        # Run chi-squared
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        # Pairwise comparisons with FDR correction
        pairwise_pvalues = []
        comparisons = []

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                g1_idx = df[df[dim_col] == groups[i]].index[0]
                g2_idx = df[df[dim_col] == groups[j]].index[0]

                s1, t1 = successes.iloc[g1_idx], trials.iloc[g1_idx]
                s2, t2 = successes.iloc[g2_idx], trials.iloc[g2_idx]

                pair_result = proportion_test(int(s1), int(t1), int(s2), int(t2))
                pairwise_pvalues.append(pair_result['p_value'])
                comparisons.append((groups[i], groups[j]))

        # Apply BH-FDR correction
        if plan.get('corrections', False) and len(pairwise_pvalues) > 0:
            rejected, adjusted_pvalues = benjamini_hochberg(
                pairwise_pvalues,
                alpha=plan.get('alpha', 0.05)
            )
        else:
            rejected = [p < plan.get('alpha', 0.05) for p in pairwise_pvalues]
            adjusted_pvalues = pairwise_pvalues

        # Build group stats
        group_stats = []
        for group in groups:
            group_row = df[df[dim_col] == group].iloc[0]
            s = successes.loc[group_row.name]
            t = trials.loc[group_row.name]
            group_stats.append({
                'group': group,
                'rate': s / t if t > 0 else 0,
                'successes': int(s),
                'trials': int(t)
            })

        return {
            'test_type': 'chi_squared',
            'p_value': float(p_value),
            'chi2_stat': float(chi2),
            'dof': int(dof),
            'group_stats': group_stats,
            'pairwise': {
                'comparisons': [f"{c[0]} vs {c[1]}" for c in comparisons],
                'p_values': pairwise_pvalues,
                'adjusted_p_values': adjusted_pvalues,
                'significant': rejected
            } if plan.get('corrections', False) else None,
            'confidence_level': plan.get('alpha', 0.05)
        }

    def _run_anova(self, df: pd.DataFrame, plan: Dict[str, Any],
                  dim_col: str) -> Dict[str, Any]:
        """Run one-way ANOVA for continuous metrics with >2 groups."""
        from scipy.stats import f_oneway

        metric = plan['metric']
        groups = plan['groups']

        # Get data for each group
        group_data = [df[df[dim_col] == g][metric].dropna().values for g in groups]

        # Run ANOVA
        f_stat, p_value = f_oneway(*group_data)

        # Pairwise t-tests with FDR correction
        pairwise_pvalues = []
        comparisons = []

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                pair_result = welch_ttest(group_data[i], group_data[j])
                pairwise_pvalues.append(pair_result['p_value'])
                comparisons.append((groups[i], groups[j]))

        # Apply BH-FDR correction
        if plan.get('corrections', False) and len(pairwise_pvalues) > 0:
            rejected, adjusted_pvalues = benjamini_hochberg(
                pairwise_pvalues,
                alpha=plan.get('alpha', 0.05)
            )
        else:
            rejected = [p < plan.get('alpha', 0.05) for p in pairwise_pvalues]
            adjusted_pvalues = pairwise_pvalues

        # Build group stats
        group_stats = []
        for i, group in enumerate(groups):
            group_stats.append({
                'group': group,
                'mean': float(np.mean(group_data[i])),
                'std': float(np.std(group_data[i])),
                'n': len(group_data[i])
            })

        return {
            'test_type': 'anova',
            'p_value': float(p_value),
            'f_stat': float(f_stat),
            'group_stats': group_stats,
            'pairwise': {
                'comparisons': [f"{c[0]} vs {c[1]}" for c in comparisons],
                'p_values': pairwise_pvalues,
                'adjusted_p_values': adjusted_pvalues,
                'significant': rejected
            } if plan.get('corrections', False) else None,
            'confidence_level': plan.get('alpha', 0.05)
        }

    def report(self, results: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report with insights from hypothesis test.

        Args:
            results: Results dict from run()
            plan: Original plan dict

        Returns:
            Report dict with summary, insights, caveats, next_actions
        """
        test_type = results['test_type']
        p_value = results['p_value']
        alpha = plan.get('alpha', 0.05)

        is_significant = p_value < alpha

        # Build summary
        if test_type in ['z_test', 't_test']:
            g1 = results['group_stats'][0]
            g2 = results['group_stats'][1]

            if 'rate' in g1:
                summary = (
                    f"{g1['group']}: {g1['rate']:.1%} ({g1['successes']}/{g1['trials']}), "
                    f"{g2['group']}: {g2['rate']:.1%} ({g2['successes']}/{g2['trials']}). "
                )
            else:
                summary = (
                    f"{g1['group']}: {g1['mean']:.2f} (n={g1['n']}), "
                    f"{g2['group']}: {g2['mean']:.2f} (n={g2['n']}). "
                )

            if is_significant:
                summary += f"Difference is statistically significant (p={p_value:.4f})."
            else:
                summary += f"Difference is not statistically significant (p={p_value:.4f})."

        else:  # chi_squared or anova
            summary = (
                f"Tested {len(results['group_stats'])} groups. "
                f"Overall test: {'significant' if is_significant else 'not significant'} "
                f"(p={p_value:.4f})."
            )

        # Build insights
        insights = []

        if is_significant:
            if test_type in ['z_test', 't_test']:
                effect_abs = results['effect']['abs']
                effect_rel = results['effect']['rel']
                insights.append(
                    f"The difference is {effect_abs:.3f} in absolute terms "
                    f"({effect_rel:+.1%} relative lift)."
                )
            elif 'pairwise' in results and results['pairwise']:
                sig_pairs = [
                    results['pairwise']['comparisons'][i]
                    for i, is_sig in enumerate(results['pairwise']['significant'])
                    if is_sig
                ]
                if sig_pairs:
                    insights.append(
                        f"Significant pairwise differences found: {', '.join(sig_pairs[:3])}."
                    )

        else:
            insights.append("No statistically significant difference detected.")

        # Caveats
        caveats = []

        # Sample size check
        if test_type in ['z_test', 't_test']:
            for group_stat in results['group_stats']:
                n = group_stat.get('n', group_stat.get('trials', 0))
                if n < self.config['stats']['min_sample_size']:
                    caveats.append(
                        f"Group '{group_stat['group']}' has small sample size (n={n}). "
                        f"Results may be unreliable."
                    )

        # Multiple testing
        if results.get('pairwise'):
            n_comparisons = len(results['pairwise']['comparisons'])
            caveats.append(
                f"Multiple comparisons ({n_comparisons} pairs) increase false positive risk. "
                f"FDR correction {'was' if plan.get('corrections') else 'was not'} applied."
            )

        # Next actions
        next_actions = []

        if is_significant:
            next_actions.append("Investigate drivers of the observed difference.")
            next_actions.append("Consider running this test on different time windows to confirm stability.")
        else:
            next_actions.append("Consider increasing sample size or extending time window.")
            next_actions.append("Explore segmentation to find subgroups with differences.")

        return {
            'summary': summary,
            'insights': insights,
            'caveats': caveats,
            'next_actions': next_actions,
            'raw_results': results
        }
