import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Union
import logging
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Data loader utility for the AuralyTica Tel AI system.
    Handles loading, processing, and generating sample data.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def load_csv_feedback(self, file_path: str) -> pd.DataFrame:
        """Load customer feedback from CSV file"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            raise

    def load_json_feedback(self, file_path: str) -> List[Dict[str, Any]]:
        """Load customer feedback from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} entries from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            raise

    def create_sample_feedback_data(self, num_entries: int = 100) -> pd.DataFrame:
        """Create sample feedback data for testing"""

        # Sample feedback templates
        positive_feedback = [
            "Great product! Really satisfied with the quality and service.",
            "Excellent customer support. They resolved my issue quickly.",
            "Love this product! Highly recommend to everyone.",
            "Amazing experience. Fast delivery and great packaging.",
            "Outstanding service. The team was very helpful and professional.",
            "Perfect! Exactly what I needed. Will buy again.",
            "Fantastic quality. Worth every penny spent.",
            "Superb customer service. Very responsive and friendly.",
            "Brilliant product design. Works flawlessly.",
            "Exceeded my expectations. Truly impressed with the service."
        ]

        negative_feedback = [
            "Terrible experience. Product didn't work as expected.",
            "Poor customer service. No one responded to my complaints.",
            "Product quality is awful. Broke after one use.",
            "Delivery was extremely slow and packaging was damaged.",
            "Overpriced for what you get. Not worth the money.",
            "Website is confusing and difficult to navigate.",
            "Product description was misleading. Got something different.",
            "Customer support was rude and unhelpful.",
            "Had to return the product. Complete waste of time.",
            "Disappointed with the purchase. Would not recommend."
        ]

        neutral_feedback = [
            "The product is okay. Nothing special but does the job.",
            "Average service. Could be better but not terrible.",
            "Product arrived on time. Standard packaging.",
            "It's an okay product for the price point.",
            "Service was acceptable. No major issues.",
            "The product works fine. No complaints or praise.",
            "Delivery was as expected. Product is decent.",
            "Standard experience. Nothing to complain about.",
            "Product is functional. Meets basic requirements.",
            "Normal customer service experience. Average response time."
        ]

        # Source types
        source_types = ['email', 'chat', 'phone', 'survey', 'review', 'social_media']

        # Generate sample data
        data = []

        for i in range(num_entries):
            # Randomly choose sentiment distribution (slightly more positive)
            rand = random.random()
            if rand < 0.4:  # 40% positive
                feedback_text = random.choice(positive_feedback)
                true_sentiment = 'positive'
            elif rand < 0.75:  # 35% neutral
                feedback_text = random.choice(neutral_feedback)
                true_sentiment = 'neutral'
            else:  # 25% negative
                feedback_text = random.choice(negative_feedback)
                true_sentiment = 'negative'

            # Add some variation to the text
            variations = [
                f"I think {feedback_text.lower()}",
                f"In my opinion, {feedback_text.lower()}",
                f"Overall, {feedback_text.lower()}",
                feedback_text,
                f"{feedback_text} Thanks!",
                f"Just wanted to say, {feedback_text.lower()}"
            ]

            varied_text = random.choice(variations)

            data.append({
                'id': f'feedback_{i+1:04d}',
                'feedback': varied_text,
                'source_type': random.choice(source_types),
                'timestamp': (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
                'customer_id': f'customer_{random.randint(1000, 9999)}',
                'true_sentiment': true_sentiment  # For validation purposes
            })

        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} sample feedback entries")

        return df

    def get_sample_dashboard_data(self) -> Dict[str, Any]:
        """Generate sample dashboard data for demonstration"""

        # Generate sample data
        sample_df = self.create_sample_feedback_data(50)

        # Calculate sample metrics
        sentiment_counts = sample_df['true_sentiment'].value_counts()
        total = len(sample_df)

        dashboard_data = {
            'key_metrics': {
                'total_feedback': total,
                'overall_sentiment': sentiment_counts.index[0] if not sentiment_counts.empty else 'neutral',
                'average_confidence': round(random.uniform(0.7, 0.9), 2),
                'sentiment_distribution': {
                    'positive': sentiment_counts.get('positive', 0) / total,
                    'negative': sentiment_counts.get('negative', 0) / total,
                    'neutral': sentiment_counts.get('neutral', 0) / total
                },
                'high_confidence_ratio': round(random.uniform(0.6, 0.8), 2)
            },
            'top_themes': [
                {'name': 'Product Quality', 'frequency': random.randint(10, 25), 'sentiment': 'positive'},
                {'name': 'Customer Service', 'frequency': random.randint(8, 20), 'sentiment': 'positive'},
                {'name': 'Delivery Speed', 'frequency': random.randint(5, 15), 'sentiment': 'neutral'},
                {'name': 'Pricing', 'frequency': random.randint(3, 12), 'sentiment': 'negative'},
                {'name': 'User Experience', 'frequency': random.randint(4, 10), 'sentiment': 'positive'}
            ],
            'emotion_summary': {
                'dominant_emotion': random.choice(['joy', 'trust', 'anticipation']),
                'total_emotions_detected': random.randint(3, 8),
                'emotion_distribution': {
                    'joy': {'percentage': random.uniform(0.2, 0.4), 'count': random.randint(8, 15)},
                    'trust': {'percentage': random.uniform(0.1, 0.3), 'count': random.randint(5, 12)},
                    'anger': {'percentage': random.uniform(0.05, 0.15), 'count': random.randint(2, 8)}
                }
            },
            'alerts_count': random.randint(1, 5),
            'processing_stats': {
                'last_updated': datetime.now().isoformat(),
                'processing_time': round(random.uniform(0.5, 2.5), 2),
                'success_rate': round(random.uniform(0.95, 1.0), 3)
            },
            'recent_feedback': sample_df.head(5).to_dict('records')
        }

        return dashboard_data

    def validate_feedback_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate feedback data format and quality"""
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'stats': {}
        }

        # Check required columns
        required_columns = ['feedback']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing_columns}")

        # Check for empty feedback
        if 'feedback' in df.columns:
            empty_feedback = df['feedback'].isna().sum()
            if empty_feedback > 0:
                validation_results['warnings'].append(f"{empty_feedback} empty feedback entries found")

            # Check feedback length
            if not df['feedback'].isna().all():
                feedback_lengths = df['feedback'].dropna().str.len()
                validation_results['stats']['avg_feedback_length'] = feedback_lengths.mean()
                validation_results['stats']['min_feedback_length'] = feedback_lengths.min()
                validation_results['stats']['max_feedback_length'] = feedback_lengths.max()

                # Warn about very short or very long feedback
                too_short = (feedback_lengths < 5).sum()
                too_long = (feedback_lengths > 1000).sum()

                if too_short > 0:
                    validation_results['warnings'].append(f"{too_short} feedback entries are very short (< 5 characters)")
                if too_long > 0:
                    validation_results['warnings'].append(f"{too_long} feedback entries are very long (> 1000 characters)")

        # General data quality checks
        validation_results['stats']['total_rows'] = len(df)
        validation_results['stats']['total_columns'] = len(df.columns)
        validation_results['stats']['duplicate_rows'] = df.duplicated().sum()

        if df.duplicated().sum() > 0:
            validation_results['warnings'].append(f"{df.duplicated().sum()} duplicate rows found")

        return validation_results

    def prepare_data_for_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for analysis by cleaning and formatting"""
        # Make a copy to avoid modifying original
        processed_df = df.copy()

        # Ensure feedback column exists and is string type
        if 'feedback' in processed_df.columns:
            processed_df['feedback'] = processed_df['feedback'].astype(str)

            # Remove rows where feedback is just 'nan' or empty
            processed_df = processed_df[processed_df['feedback'] != 'nan']
            processed_df = processed_df[processed_df['feedback'].str.strip() != '']

        # Add timestamp if not present
        if 'timestamp' not in processed_df.columns:
            processed_df['timestamp'] = datetime.now().isoformat()

        # Add source_type if not present
        if 'source_type' not in processed_df.columns:
            processed_df['source_type'] = 'unknown'

        # Add unique ID if not present
        if 'id' not in processed_df.columns:
            processed_df['id'] = [f'entry_{i:04d}' for i in range(len(processed_df))]

        logger.info(f"Prepared {len(processed_df)} entries for analysis")

        return processed_df

    def export_results(self, results: Dict[str, Any], file_path: str, format: str = 'json'):
        """Export analysis results to file"""
        try:
            if format.lower() == 'json':
                with open(file_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            elif format.lower() == 'csv' and 'results' in results:
                # Convert results to DataFrame for CSV export
                df_data = []
                for result in results['results']:
                    row = {
                        'text': result.get('text', ''),
                        'sentiment': result.get('final_sentiment', ''),
                        'confidence': result.get('confidence', 0.0),
                        'models_used': ', '.join(result.get('models_used', []))
                    }
                    df_data.append(row)

                df = pd.DataFrame(df_data)
                df.to_csv(file_path, index=False)
            else:
                raise ValueError(f"Unsupported export format: {format}")

            logger.info(f"Results exported to {file_path}")

        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            raise
