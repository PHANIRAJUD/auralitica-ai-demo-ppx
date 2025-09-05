import logging
from typing import Dict, List, Any, Union
import numpy as np
import pandas as pd
import json
import base64
import io
from datetime import datetime

# Visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.utils import PlotlyJSONEncoder
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

logger = logging.getLogger(__name__)

class VisualizationAgent:
    """
    Visualization Agent that creates graphs, dashboards, and alerts.
    Fourth stage of the AuralyTica Tel AI pipeline.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'generate_charts': True,
            'create_wordcloud': True,
            'dashboard_alerts': True,
            'chart_style': 'plotly',  # 'plotly' or 'matplotlib'
            'color_scheme': 'default',
            'figure_size': (12, 8),
            'dpi': 100
        }

        # Color schemes
        self.color_schemes = {
            'default': {
                'positive': '#2E8B57',    # Sea Green
                'negative': '#DC143C',    # Crimson  
                'neutral': '#4682B4',     # Steel Blue
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'accent': '#2ca02c'
            },
            'professional': {
                'positive': '#28A745',
                'negative': '#DC3545', 
                'neutral': '#6C757D',
                'primary': '#007BFF',
                'secondary': '#6F42C1',
                'accent': '#20C997'
            }
        }

        # Set visualization style
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'available') and 'seaborn-v0_8' in plt.style.available else 'default')

        # Stats tracking
        self.stats = {
            'charts_generated': 0,
            'wordclouds_created': 0,
            'alerts_generated': 0,
            'dashboard_components': 0
        }

        logger.info("VisualizationAgent initialized")

    def create_visualizations(self, preprocessed_data: Dict[str, Any], 
                            sentiment_results: Dict[str, Any], 
                            insights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main visualization creation function.

        Args:
            preprocessed_data: Output from PreprocessingAgent
            sentiment_results: Output from SentimentAgent  
            insights: Output from InsightAgent

        Returns:
            Dictionary containing all visualizations and dashboard components
        """
        try:
            logger.info("Creating visualizations and dashboard components")

            # Reset stats
            self._reset_stats()

            visualizations = {
                'charts': {},
                'wordclouds': {},
                'dashboard_data': {},
                'alerts': [],
                'interactive_components': {},
                'metadata': {}
            }

            # Generate charts
            if self.config.get('generate_charts', True):
                visualizations['charts'] = self._generate_charts(
                    preprocessed_data, sentiment_results, insights
                )
                self.stats['charts_generated'] = len(visualizations['charts'])

            # Create word clouds
            if self.config.get('create_wordcloud', True):
                visualizations['wordclouds'] = self._create_wordclouds(
                    preprocessed_data, insights
                )
                self.stats['wordclouds_created'] = len(visualizations['wordclouds'])

            # Generate dashboard data
            visualizations['dashboard_data'] = self._create_dashboard_data(
                preprocessed_data, sentiment_results, insights
            )

            # Create interactive components
            visualizations['interactive_components'] = self._create_interactive_components(
                sentiment_results, insights
            )

            # Generate alerts
            if self.config.get('dashboard_alerts', True):
                visualizations['alerts'] = self._generate_alerts(insights)
                self.stats['alerts_generated'] = len(visualizations['alerts'])

            # Add metadata
            visualizations['metadata'] = {
                'generation_timestamp': datetime.now().isoformat(),
                'stats': self.stats.copy(),
                'config_used': self.config.copy(),
                'available_libraries': {
                    'matplotlib': MATPLOTLIB_AVAILABLE,
                    'plotly': PLOTLY_AVAILABLE,
                    'wordcloud': WORDCLOUD_AVAILABLE
                }
            }

            self.stats['dashboard_components'] = (
                len(visualizations['charts']) + 
                len(visualizations['wordclouds']) + 
                len(visualizations['interactive_components'])
            )

            logger.info(f"Visualization generation completed. Created {self.stats['charts_generated']} charts, "
                       f"{self.stats['wordclouds_created']} wordclouds, {self.stats['alerts_generated']} alerts")

            return visualizations

        except Exception as e:
            logger.error(f"Error in visualization generation: {str(e)}")
            raise

    def _generate_charts(self, preprocessed_data: Dict[str, Any], 
                        sentiment_results: Dict[str, Any], 
                        insights: Dict[str, Any]) -> Dict[str, Any]:
        """Generate various charts for the dashboard"""
        charts = {}

        # Sentiment Distribution Chart
        if sentiment_results.get('summary'):
            charts['sentiment_distribution'] = self._create_sentiment_distribution_chart(
                sentiment_results['summary']
            )

        # Theme Analysis Chart
        themes_data = insights.get('themes', {}).get('theme_analysis', {})
        if themes_data:
            charts['theme_analysis'] = self._create_theme_analysis_chart(themes_data)

        # Emotion Distribution Chart
        emotions_data = insights.get('emotions', {}).get('emotion_distribution', {})
        if emotions_data:
            charts['emotion_distribution'] = self._create_emotion_distribution_chart(emotions_data)

        # Intensity Analysis Chart
        intensity_data = insights.get('intensity_analysis', {})
        if intensity_data.get('intensity_distribution'):
            charts['intensity_analysis'] = self._create_intensity_analysis_chart(intensity_data)

        # Confidence vs Sentiment Scatter Plot
        if sentiment_results.get('results'):
            charts['confidence_vs_sentiment'] = self._create_confidence_scatter_chart(
                sentiment_results['results']
            )

        # Time series (if temporal data available)
        # This would require timestamp data in the input

        return charts

    def _create_sentiment_distribution_chart(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Create sentiment distribution pie/bar chart"""
        sentiment_dist = summary.get('sentiment_distribution', {})

        if not sentiment_dist:
            return {}

        colors = self._get_colors()

        if PLOTLY_AVAILABLE and self.config.get('chart_style') == 'plotly':
            # Create Plotly pie chart
            fig = go.Figure(data=[go.Pie(
                labels=list(sentiment_dist.keys()),
                values=list(sentiment_dist.values()),
                marker_colors=[colors.get(sentiment, colors['primary']) for sentiment in sentiment_dist.keys()],
                textinfo='label+percent',
                textfont_size=14,
                hole=0.3
            )])

            fig.update_layout(
                title={
                    'text': 'Sentiment Distribution',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                font_size=12,
                showlegend=True,
                width=600,
                height=400
            )

            return {
                'type': 'plotly',
                'data': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder)),
                'title': 'Sentiment Distribution',
                'description': f"Distribution of sentiment across {summary.get('total_analyzed', 0)} feedback entries"
            }

        elif MATPLOTLIB_AVAILABLE:
            # Create matplotlib pie chart
            fig, ax = plt.subplots(figsize=(8, 6))

            labels = list(sentiment_dist.keys())
            values = list(sentiment_dist.values())
            chart_colors = [colors.get(sentiment, colors['primary']) for sentiment in labels]

            wedges, texts, autotexts = ax.pie(values, labels=labels, colors=chart_colors, 
                                            autopct='%1.1f%%', startangle=90)

            ax.set_title('Sentiment Distribution', fontsize=16, fontweight='bold')

            # Convert to base64 for web display
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=self.config.get('dpi', 100), 
                       bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close(fig)

            return {
                'type': 'matplotlib',
                'data': img_base64,
                'title': 'Sentiment Distribution',
                'description': f"Distribution of sentiment across {summary.get('total_analyzed', 0)} feedback entries"
            }

        return {}

    def _create_theme_analysis_chart(self, themes_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create theme analysis bar chart"""
        if not themes_data:
            return {}

        # Prepare data
        themes = list(themes_data.keys())[:10]  # Top 10 themes
        frequencies = [themes_data[theme]['frequency'] for theme in themes]
        dominant_sentiments = [themes_data[theme]['dominant_sentiment'] for theme in themes]

        colors = self._get_colors()
        bar_colors = [colors.get(sentiment, colors['primary']) for sentiment in dominant_sentiments]

        if PLOTLY_AVAILABLE and self.config.get('chart_style') == 'plotly':
            # Create Plotly bar chart
            fig = go.Figure(data=[go.Bar(
                x=themes,
                y=frequencies,
                marker_color=bar_colors,
                text=dominant_sentiments,
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Frequency: %{y}<br>Dominant Sentiment: %{text}<extra></extra>'
            )])

            fig.update_layout(
                title={
                    'text': 'Top Themes by Frequency',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title='Themes',
                yaxis_title='Frequency',
                xaxis_tickangle=-45,
                width=800,
                height=500,
                margin=dict(b=100)
            )

            return {
                'type': 'plotly',
                'data': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder)),
                'title': 'Theme Analysis',
                'description': f"Top {len(themes)} themes identified in customer feedback"
            }

        elif MATPLOTLIB_AVAILABLE:
            # Create matplotlib bar chart
            fig, ax = plt.subplots(figsize=(12, 6))

            bars = ax.bar(range(len(themes)), frequencies, color=bar_colors, alpha=0.8)
            ax.set_xlabel('Themes')
            ax.set_ylabel('Frequency')
            ax.set_title('Top Themes by Frequency', fontsize=16, fontweight='bold')
            ax.set_xticks(range(len(themes)))
            ax.set_xticklabels(themes, rotation=45, ha='right')

            # Add value labels on bars
            for i, (bar, sentiment) in enumerate(zip(bars, dominant_sentiments)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{sentiment}\n{int(height)}',
                       ha='center', va='bottom', fontsize=10)

            plt.tight_layout()

            # Convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=self.config.get('dpi', 100), 
                       bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close(fig)

            return {
                'type': 'matplotlib',
                'data': img_base64,
                'title': 'Theme Analysis',
                'description': f"Top {len(themes)} themes identified in customer feedback"
            }

        return {}

    def _create_emotion_distribution_chart(self, emotions_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create emotion distribution chart"""
        if not emotions_data:
            return {}

        emotions = list(emotions_data.keys())
        percentages = [emotions_data[emotion]['percentage'] * 100 for emotion in emotions]

        if PLOTLY_AVAILABLE and self.config.get('chart_style') == 'plotly':
            # Create Plotly horizontal bar chart
            fig = go.Figure(data=[go.Bar(
                x=percentages,
                y=emotions,
                orientation='h',
                marker_color=px.colors.qualitative.Set3[:len(emotions)],
                hovertemplate='<b>%{y}</b><br>Percentage: %{x:.1f}%<extra></extra>'
            )])

            fig.update_layout(
                title={
                    'text': 'Emotion Distribution',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title='Percentage (%)',
                yaxis_title='Emotions',
                width=600,
                height=400
            )

            return {
                'type': 'plotly',
                'data': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder)),
                'title': 'Emotion Distribution',
                'description': f"Distribution of {len(emotions)} emotions detected in feedback"
            }

        elif MATPLOTLIB_AVAILABLE:
            # Create matplotlib horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, 6))

            y_pos = np.arange(len(emotions))
            bars = ax.barh(y_pos, percentages, color=plt.cm.Set3(np.linspace(0, 1, len(emotions))))

            ax.set_yticks(y_pos)
            ax.set_yticklabels(emotions)
            ax.set_xlabel('Percentage (%)')
            ax.set_title('Emotion Distribution', fontsize=16, fontweight='bold')

            # Add value labels
            for i, (bar, pct) in enumerate(zip(bars, percentages)):
                width = bar.get_width()
                ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                       f'{pct:.1f}%', ha='left', va='center')

            plt.tight_layout()

            # Convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=self.config.get('dpi', 100), 
                       bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close(fig)

            return {
                'type': 'matplotlib',
                'data': img_base64,
                'title': 'Emotion Distribution',
                'description': f"Distribution of {len(emotions)} emotions detected in feedback"
            }

        return {}

    def _create_intensity_analysis_chart(self, intensity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create intensity analysis chart"""
        intensity_dist = intensity_data.get('intensity_distribution', {})

        if not intensity_dist:
            return {}

        levels = list(intensity_dist.keys())
        values = [intensity_dist[level] * 100 for level in levels]  # Convert to percentage

        # Color mapping for intensity levels
        intensity_colors = {
            'very_high': '#8B0000',  # Dark Red
            'high': '#DC143C',       # Crimson
            'medium': '#FF8C00',     # Dark Orange  
            'low': '#32CD32'         # Lime Green
        }

        chart_colors = [intensity_colors.get(level, '#4682B4') for level in levels]

        if PLOTLY_AVAILABLE and self.config.get('chart_style') == 'plotly':
            # Create Plotly donut chart
            fig = go.Figure(data=[go.Pie(
                labels=levels,
                values=values,
                hole=0.4,
                marker_colors=chart_colors,
                textinfo='label+percent'
            )])

            fig.update_layout(
                title={
                    'text': 'Sentiment Intensity Distribution',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                width=500,
                height=400
            )

            return {
                'type': 'plotly',
                'data': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder)),
                'title': 'Intensity Analysis',
                'description': f"Distribution of sentiment intensity levels"
            }

        elif MATPLOTLIB_AVAILABLE:
            # Create matplotlib donut chart
            fig, ax = plt.subplots(figsize=(8, 6))

            wedges, texts, autotexts = ax.pie(values, labels=levels, colors=chart_colors,
                                            autopct='%1.1f%%', pctdistance=0.85)

            # Add center circle for donut effect
            centre_circle = plt.Circle((0,0), 0.60, fc='white')
            ax.add_artist(centre_circle)

            ax.set_title('Sentiment Intensity Distribution', fontsize=16, fontweight='bold')

            # Convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=self.config.get('dpi', 100), 
                       bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close(fig)

            return {
                'type': 'matplotlib',
                'data': img_base64,
                'title': 'Intensity Analysis',
                'description': f"Distribution of sentiment intensity levels"
            }

        return {}

    def _create_confidence_scatter_chart(self, sentiment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create confidence vs sentiment scatter plot"""
        if not sentiment_results:
            return {}

        # Prepare data
        sentiments = []
        confidences = []
        colors_list = []

        colors = self._get_colors()

        for result in sentiment_results:
            sentiment = result.get('final_sentiment', 'neutral')
            confidence = result.get('confidence', 0.0)

            sentiments.append(sentiment)
            confidences.append(confidence)
            colors_list.append(colors.get(sentiment, colors['primary']))

        if PLOTLY_AVAILABLE and self.config.get('chart_style') == 'plotly':
            # Create Plotly scatter plot
            fig = go.Figure()

            for sentiment in set(sentiments):
                sentiment_confidences = [conf for sent, conf in zip(sentiments, confidences) if sent == sentiment]
                sentiment_indices = [i for i, sent in enumerate(sentiments) if sent == sentiment]

                fig.add_trace(go.Scatter(
                    x=sentiment_indices,
                    y=sentiment_confidences,
                    mode='markers',
                    name=sentiment.title(),
                    marker=dict(
                        color=colors.get(sentiment, colors['primary']),
                        size=8,
                        opacity=0.7
                    ),
                    hovertemplate=f'<b>{sentiment.title()}</b><br>Index: %{{x}}<br>Confidence: %{{y:.2f}}<extra></extra>'
                ))

            fig.update_layout(
                title={
                    'text': 'Sentiment Confidence Distribution',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title='Feedback Entry Index',
                yaxis_title='Confidence Score',
                width=800,
                height=500
            )

            return {
                'type': 'plotly',
                'data': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder)),
                'title': 'Confidence vs Sentiment',
                'description': f"Confidence scores for {len(sentiment_results)} sentiment predictions"
            }

        return {}

    def _create_wordclouds(self, preprocessed_data: Dict[str, Any], insights: Dict[str, Any]) -> Dict[str, Any]:
        """Create word clouds from text data"""
        if not WORDCLOUD_AVAILABLE:
            logger.warning("WordCloud library not available")
            return {}

        wordclouds = {}
        processed_entries = preprocessed_data.get('processed_entries', [])

        if not processed_entries:
            return {}

        # Overall word cloud
        all_text = ' '.join([entry['cleaned_text'] for entry in processed_entries])
        if all_text.strip():
            wordclouds['overall'] = self._generate_wordcloud(
                all_text, 'Overall Feedback Word Cloud'
            )

        # Sentiment-specific word clouds
        sentiment_texts = {'positive': [], 'negative': [], 'neutral': []}

        # This requires sentiment data to be linked back to processed entries
        # For now, create a general word cloud

        return wordclouds

    def _generate_wordcloud(self, text: str, title: str) -> Dict[str, Any]:
        """Generate a single word cloud"""
        try:
            wordcloud = WordCloud(
                width=800, 
                height=400,
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate(text)

            if MATPLOTLIB_AVAILABLE:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title(title, fontsize=16, fontweight='bold')

                # Convert to base64
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=self.config.get('dpi', 100), 
                           bbox_inches='tight')
                img_buffer.seek(0)
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                plt.close(fig)

                return {
                    'type': 'matplotlib',
                    'data': img_base64,
                    'title': title,
                    'description': f"Word cloud showing most frequent terms"
                }

        except Exception as e:
            logger.warning(f"Error generating word cloud: {e}")

        return {}

    def _create_dashboard_data(self, preprocessed_data: Dict[str, Any], 
                             sentiment_results: Dict[str, Any], 
                             insights: Dict[str, Any]) -> Dict[str, Any]:
        """Create data for dashboard widgets"""
        dashboard_data = {}

        # Key metrics
        summary = sentiment_results.get('summary', {})
        dashboard_data['key_metrics'] = {
            'total_feedback': summary.get('total_analyzed', 0),
            'overall_sentiment': summary.get('overall_sentiment', 'neutral'),
            'average_confidence': round(summary.get('average_confidence', 0.0), 2),
            'sentiment_distribution': summary.get('sentiment_distribution', {}),
            'high_confidence_ratio': round(summary.get('high_confidence_ratio', 0.0), 2)
        }

        # Top themes
        themes = insights.get('themes', {}).get('theme_analysis', {})
        dashboard_data['top_themes'] = [
            {
                'name': theme.replace('auto_', '').replace('_', ' ').title(),
                'frequency': data['frequency'],
                'sentiment': data['dominant_sentiment']
            }
            for theme, data in list(themes.items())[:5]
        ]

        # Emotion summary
        emotions = insights.get('emotions', {})
        dashboard_data['emotion_summary'] = {
            'dominant_emotion': emotions.get('dominant_emotion', 'none'),
            'total_emotions_detected': emotions.get('total_emotions_detected', 0),
            'emotion_distribution': emotions.get('emotion_distribution', {})
        }

        # Alerts count
        dashboard_data['alerts_count'] = len(insights.get('actionable_insights', []))

        return dashboard_data

    def _create_interactive_components(self, sentiment_results: Dict[str, Any], 
                                     insights: Dict[str, Any]) -> Dict[str, Any]:
        """Create interactive dashboard components"""
        components = {}

        # Sentiment gauge
        summary = sentiment_results.get('summary', {})
        sentiment_dist = summary.get('sentiment_distribution', {})

        if sentiment_dist:
            positive_ratio = sentiment_dist.get('positive', 0)
            negative_ratio = sentiment_dist.get('negative', 0)

            # Calculate sentiment score (-1 to 1)
            sentiment_score = positive_ratio - negative_ratio

            if PLOTLY_AVAILABLE:
                # Create sentiment gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = sentiment_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Sentiment Score"},
                    delta = {'reference': 0},
                    gauge = {
                        'axis': {'range': [None, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [-1, -0.5], 'color': "red"},
                            {'range': [-0.5, 0], 'color': "orange"},
                            {'range': [0, 0.5], 'color': "lightgreen"},
                            {'range': [0.5, 1], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.9
                        }
                    }
                ))

                components['sentiment_gauge'] = {
                    'type': 'plotly',
                    'data': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder)),
                    'title': 'Overall Sentiment Score',
                    'description': f"Composite sentiment score ranging from -1 (very negative) to +1 (very positive)"
                }

        return components

    def _generate_alerts(self, insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate dashboard alerts based on insights"""
        alerts = []

        # Process actionable insights into alerts
        actionable_insights = insights.get('actionable_insights', [])

        for insight in actionable_insights:
            alert_level = self._map_priority_to_level(insight.get('priority', 'medium'))

            alert = {
                'id': f"alert_{len(alerts) + 1}",
                'level': alert_level,
                'title': insight.get('issue', 'Action Required'),
                'message': insight.get('recommendation', 'Please review the feedback'),
                'type': insight.get('type', 'general'),
                'timestamp': datetime.now().isoformat(),
                'data': {
                    'affected_percentage': insight.get('affected_percentage'),
                    'affected_count': insight.get('affected_count'),
                    'theme': insight.get('theme'),
                    'emotion': insight.get('emotion')
                }
            }

            alerts.append(alert)

        # Additional system-generated alerts
        # High negative sentiment alert
        emotions = insights.get('emotions', {}).get('emotion_distribution', {})
        if 'anger' in emotions and emotions['anger']['percentage'] > 0.15:
            alerts.append({
                'id': f"alert_{len(alerts) + 1}",
                'level': 'warning',
                'title': 'High Anger Levels Detected',
                'message': f"Anger detected in {emotions['anger']['percentage']:.1%} of feedback",
                'type': 'emotion_alert',
                'timestamp': datetime.now().isoformat()
            })

        return alerts

    def _map_priority_to_level(self, priority: str) -> str:
        """Map priority to alert level"""
        mapping = {
            'critical': 'error',
            'high': 'warning', 
            'medium': 'info',
            'low': 'info'
        }
        return mapping.get(priority, 'info')

    def _get_colors(self) -> Dict[str, str]:
        """Get color scheme for charts"""
        scheme = self.config.get('color_scheme', 'default')
        return self.color_schemes.get(scheme, self.color_schemes['default'])

    def _reset_stats(self):
        """Reset processing statistics"""
        for key in self.stats:
            self.stats[key] = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get visualization statistics"""
        return self.stats.copy()

    def reset(self):
        """Reset agent to initial state"""
        self._reset_stats()
        logger.info("VisualizationAgent reset")
