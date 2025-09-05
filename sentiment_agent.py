import logging
from typing import Dict, List, Any, Union
import numpy as np

# Import sentiment analysis libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# For transformers-based sentiment analysis
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# For Azure AI Services (optional)
try:
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

logger = logging.getLogger(__name__)

class SentimentAgent:
    """
    Sentiment Analysis Agent that classifies feedback sentiment using multiple models.
    Second stage of the AuralyTica Tel AI pipeline.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'primary_model': 'vader',
            'secondary_model': 'textblob', 
            'use_ensemble': True,
            'confidence_threshold': 0.6,
            'enable_azure': False
        }

        # Initialize sentiment analyzers
        self.vader_analyzer = SentimentIntensityAnalyzer()

        # Initialize Transformers model if available
        self.transformers_pipeline = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.transformers_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    tokenizer="distilbert-base-uncased-finetuned-sst-2-english"
                )
                logger.info("Transformers sentiment model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load transformers model: {e}")

        # Initialize Azure client if configured
        self.azure_client = None
        if self.config.get('enable_azure', False) and AZURE_AVAILABLE:
            self._initialize_azure()

        # Stats tracking
        self.stats = {
            'total_analyzed': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'high_confidence_count': 0,
            'low_confidence_count': 0
        }

        logger.info("SentimentAgent initialized")

    def analyze(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main sentiment analysis function.

        Args:
            preprocessed_data: Output from PreprocessingAgent

        Returns:
            Dictionary containing sentiment analysis results
        """
        try:
            processed_entries = preprocessed_data.get('processed_entries', [])

            if not processed_entries:
                logger.warning("No processed entries found for sentiment analysis")
                return {'results': [], 'summary': {}, 'stats': self.stats}

            logger.info(f"Analyzing sentiment for {len(processed_entries)} entries")

            # Reset stats
            self._reset_stats()
            self.stats['total_analyzed'] = len(processed_entries)

            # Analyze each entry
            sentiment_results = []
            for entry in processed_entries:
                text = entry['cleaned_text']
                result = self._analyze_single_text(text)

                # Add original data reference
                result['original_index'] = entry['original_index']
                result['source_type'] = entry.get('source_type', 'text')

                sentiment_results.append(result)

                # Update stats
                self._update_stats(result)

            # Calculate summary statistics
            summary = self._calculate_summary(sentiment_results)

            logger.info(f"Sentiment analysis completed for {len(sentiment_results)} entries")

            return {
                'results': sentiment_results,
                'summary': summary,
                'stats': self.stats.copy(),
                'model_info': self._get_model_info()
            }

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            raise

    def _analyze_single_text(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment for a single text entry"""
        results = {
            'text': text,
            'models_used': [],
            'sentiment_scores': {},
            'final_sentiment': None,
            'confidence': 0.0,
            'reasoning': []
        }

        # VADER Analysis
        if self.config.get('primary_model') == 'vader' or self.config.get('use_ensemble', True):
            vader_scores = self.vader_analyzer.polarity_scores(text)
            vader_sentiment = self._interpret_vader_scores(vader_scores)

            results['sentiment_scores']['vader'] = {
                'positive': vader_scores['pos'],
                'negative': vader_scores['neg'],
                'neutral': vader_scores['neu'],
                'compound': vader_scores['compound'],
                'interpreted_sentiment': vader_sentiment['sentiment'],
                'confidence': vader_sentiment['confidence']
            }
            results['models_used'].append('vader')
            results['reasoning'].append(f"VADER: {vader_sentiment['sentiment']} (confidence: {vader_sentiment['confidence']:.2f})")

        # TextBlob Analysis
        if self.config.get('secondary_model') == 'textblob' or self.config.get('use_ensemble', True):
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            textblob_sentiment = self._interpret_textblob_scores(polarity, subjectivity)

            results['sentiment_scores']['textblob'] = {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'interpreted_sentiment': textblob_sentiment['sentiment'],
                'confidence': textblob_sentiment['confidence']
            }
            results['models_used'].append('textblob')
            results['reasoning'].append(f"TextBlob: {textblob_sentiment['sentiment']} (polarity: {polarity:.2f})")

        # Transformers Analysis (if available)
        if self.transformers_pipeline and (self.config.get('use_ensemble', True) or 'transformers' in [self.config.get('primary_model'), self.config.get('secondary_model')]):
            try:
                # Truncate text if too long for the model
                truncated_text = text[:512] if len(text) > 512 else text
                transformers_result = self.transformers_pipeline(truncated_text)[0]

                # Convert to standard format
                label = transformers_result['label'].lower()
                score = transformers_result['score']

                # Map POSITIVE/NEGATIVE to our format
                if label == 'positive':
                    sentiment = 'positive'
                elif label == 'negative':
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'

                results['sentiment_scores']['transformers'] = {
                    'label': sentiment,
                    'confidence': score,
                    'model_output': transformers_result
                }
                results['models_used'].append('transformers')
                results['reasoning'].append(f"Transformers: {sentiment} (confidence: {score:.2f})")

            except Exception as e:
                logger.warning(f"Transformers analysis failed for text: {e}")

        # Azure Analysis (if configured)
        if self.azure_client:
            try:
                azure_result = self._analyze_with_azure(text)
                if azure_result:
                    results['sentiment_scores']['azure'] = azure_result
                    results['models_used'].append('azure')
                    results['reasoning'].append(f"Azure: {azure_result['sentiment']} (confidence: {azure_result['confidence']:.2f})")
            except Exception as e:
                logger.warning(f"Azure analysis failed: {e}")

        # Determine final sentiment using ensemble or primary model
        if self.config.get('use_ensemble', True):
            results['final_sentiment'], results['confidence'] = self._ensemble_decision(results['sentiment_scores'])
        else:
            primary_model = self.config.get('primary_model', 'vader')
            if primary_model in results['sentiment_scores']:
                model_result = results['sentiment_scores'][primary_model]
                if primary_model == 'vader':
                    results['final_sentiment'] = model_result['interpreted_sentiment']
                    results['confidence'] = model_result['confidence']
                elif primary_model == 'textblob':
                    results['final_sentiment'] = model_result['interpreted_sentiment']
                    results['confidence'] = model_result['confidence']
                elif primary_model == 'transformers':
                    results['final_sentiment'] = model_result['label']
                    results['confidence'] = model_result['confidence']

        return results

    def _interpret_vader_scores(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """Interpret VADER sentiment scores"""
        compound = scores['compound']

        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        # Calculate confidence based on compound score magnitude
        confidence = min(abs(compound) * 2, 1.0)  # Scale to 0-1

        return {
            'sentiment': sentiment,
            'confidence': confidence
        }

    def _interpret_textblob_scores(self, polarity: float, subjectivity: float) -> Dict[str, Any]:
        """Interpret TextBlob sentiment scores"""
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        # Calculate confidence based on polarity magnitude and subjectivity
        confidence = min(abs(polarity) * 2, 1.0) * (1 - subjectivity * 0.5)
        confidence = max(confidence, 0.1)  # Minimum confidence

        return {
            'sentiment': sentiment,
            'confidence': confidence
        }

    def _analyze_with_azure(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using Azure AI Services"""
        if not self.azure_client:
            return None

        try:
            response = self.azure_client.analyze_sentiment(documents=[text])[0]

            if response.is_error:
                logger.warning(f"Azure sentiment analysis error: {response.error}")
                return None

            return {
                'sentiment': response.sentiment.lower(),
                'confidence': response.confidence_scores.__dict__[response.sentiment.lower()],
                'confidence_scores': {
                    'positive': response.confidence_scores.positive,
                    'neutral': response.confidence_scores.neutral,
                    'negative': response.confidence_scores.negative
                }
            }
        except Exception as e:
            logger.warning(f"Azure analysis failed: {e}")
            return None

    def _ensemble_decision(self, sentiment_scores: Dict[str, Any]) -> tuple:
        """Make ensemble decision from multiple model results"""
        if not sentiment_scores:
            return 'neutral', 0.0

        # Collect all sentiment predictions with their confidence scores
        predictions = []

        for model, scores in sentiment_scores.items():
            if model == 'vader':
                predictions.append((scores['interpreted_sentiment'], scores['confidence']))
            elif model == 'textblob':
                predictions.append((scores['interpreted_sentiment'], scores['confidence']))
            elif model == 'transformers':
                predictions.append((scores['label'], scores['confidence']))
            elif model == 'azure':
                predictions.append((scores['sentiment'], scores['confidence']))

        if not predictions:
            return 'neutral', 0.0

        # Weighted voting based on confidence
        sentiment_weights = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_weight = 0

        for sentiment, confidence in predictions:
            sentiment_weights[sentiment] += confidence
            total_weight += confidence

        # Normalize weights
        if total_weight > 0:
            for sentiment in sentiment_weights:
                sentiment_weights[sentiment] /= total_weight

        # Determine winner
        final_sentiment = max(sentiment_weights, key=sentiment_weights.get)
        final_confidence = sentiment_weights[final_sentiment]

        return final_sentiment, final_confidence

    def _update_stats(self, result: Dict[str, Any]):
        """Update processing statistics"""
        sentiment = result.get('final_sentiment', 'neutral')
        confidence = result.get('confidence', 0.0)

        if sentiment == 'positive':
            self.stats['positive_count'] += 1
        elif sentiment == 'negative':
            self.stats['negative_count'] += 1
        else:
            self.stats['neutral_count'] += 1

        if confidence >= self.config.get('confidence_threshold', 0.6):
            self.stats['high_confidence_count'] += 1
        else:
            self.stats['low_confidence_count'] += 1

    def _calculate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics"""
        total = len(results)
        if total == 0:
            return {}

        # Sentiment distribution
        sentiment_dist = {
            'positive': self.stats['positive_count'] / total,
            'negative': self.stats['negative_count'] / total,
            'neutral': self.stats['neutral_count'] / total
        }

        # Average confidence
        confidences = [r.get('confidence', 0.0) for r in results]
        avg_confidence = np.mean(confidences) if confidences else 0.0

        # Overall sentiment (majority)
        overall_sentiment = max(sentiment_dist, key=sentiment_dist.get)

        return {
            'total_analyzed': total,
            'sentiment_distribution': sentiment_dist,
            'overall_sentiment': overall_sentiment,
            'average_confidence': avg_confidence,
            'high_confidence_ratio': self.stats['high_confidence_count'] / total,
            'model_agreement': self._calculate_model_agreement(results)
        }

    def _calculate_model_agreement(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate agreement between different models"""
        if not results:
            return {}

        # Count how often models agree
        agreements = 0
        comparisons = 0

        for result in results:
            scores = result.get('sentiment_scores', {})
            if len(scores) > 1:
                sentiments = []
                for model, model_scores in scores.items():
                    if model == 'vader':
                        sentiments.append(model_scores['interpreted_sentiment'])
                    elif model == 'textblob':
                        sentiments.append(model_scores['interpreted_sentiment'])
                    elif model == 'transformers':
                        sentiments.append(model_scores['label'])
                    elif model == 'azure':
                        sentiments.append(model_scores['sentiment'])

                if len(set(sentiments)) == 1:  # All models agree
                    agreements += 1
                comparisons += 1

        agreement_rate = agreements / comparisons if comparisons > 0 else 0.0

        return {
            'agreement_rate': agreement_rate,
            'total_comparisons': comparisons
        }

    def _initialize_azure(self):
        """Initialize Azure Text Analytics client"""
        try:
            endpoint = self.config.get('azure_endpoint')
            key = self.config.get('azure_key')

            if not endpoint or not key:
                logger.warning("Azure endpoint or key not configured")
                return

            self.azure_client = TextAnalyticsClient(
                endpoint=endpoint, 
                credential=AzureKeyCredential(key)
            )
            logger.info("Azure Text Analytics client initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Azure client: {e}")

    def _get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        return {
            'available_models': {
                'vader': True,
                'textblob': True,
                'transformers': TRANSFORMERS_AVAILABLE,
                'azure': self.azure_client is not None
            },
            'primary_model': self.config.get('primary_model'),
            'use_ensemble': self.config.get('use_ensemble'),
            'confidence_threshold': self.config.get('confidence_threshold')
        }

    def _reset_stats(self):
        """Reset processing statistics"""
        for key in self.stats:
            self.stats[key] = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get sentiment analysis statistics"""
        return self.stats.copy()

    def reset(self):
        """Reset agent to initial state"""
        self._reset_stats()
        logger.info("SentimentAgent reset")
