import logging
from typing import Dict, List, Any, Union, Tuple
import numpy as np
import re
from collections import Counter, defaultdict
from datetime import datetime

# For advanced NLP tasks
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True

    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)

    try:
        nltk.data.find('chunkers/maxent_ne_chunker')
    except LookupError:
        nltk.download('maxent_ne_chunker', quiet=True)

    try:
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words', quiet=True)

except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)

class InsightAgent:
    """
    Insight Extraction Agent that extracts themes, emotions, and intensity from sentiment analysis.
    Third stage of the AuralyTica Tel AI pipeline.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'extract_themes': True,
            'emotion_detection': True,
            'intensity_analysis': True,
            'extract_entities': True,
            'min_theme_frequency': 2,
            'top_themes_count': 10
        }

        # Emotion keywords mapping
        self.emotion_keywords = {
            'joy': ['happy', 'joy', 'delighted', 'pleased', 'satisfied', 'excited', 'thrilled', 'glad', 'content'],
            'anger': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated', 'outraged', 'livid'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'terrified', 'panic', 'frightened'],
            'sadness': ['sad', 'depressed', 'disappointed', 'upset', 'miserable', 'unhappy', 'sorrowful'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned', 'bewildered'],
            'disgust': ['disgusted', 'revolted', 'appalled', 'repulsed', 'sickened', 'nauseated'],
            'trust': ['trust', 'reliable', 'dependable', 'faithful', 'confident', 'secure'],
            'anticipation': ['excited', 'eager', 'hopeful', 'optimistic', 'looking forward', 'anticipating']
        }

        # Intensity modifiers
        self.intensity_modifiers = {
            'very_high': ['extremely', 'incredibly', 'absolutely', 'completely', 'totally', 'utterly'],
            'high': ['very', 'really', 'quite', 'pretty', 'highly', 'strongly'],
            'medium': ['somewhat', 'rather', 'fairly', 'moderately'],
            'low': ['slightly', 'a bit', 'a little', 'mildly', 'barely']
        }

        # Common customer service themes
        self.service_themes = {
            'product_quality': ['quality', 'product', 'item', 'goods', 'material', 'build', 'construction'],
            'customer_service': ['service', 'support', 'help', 'staff', 'representative', 'agent', 'team'],
            'delivery': ['delivery', 'shipping', 'arrived', 'package', 'order', 'received', 'sent'],
            'pricing': ['price', 'cost', 'expensive', 'cheap', 'value', 'money', 'worth', 'affordable'],
            'usability': ['easy', 'difficult', 'user-friendly', 'complicated', 'simple', 'intuitive'],
            'communication': ['communication', 'response', 'reply', 'contact', 'information', 'update'],
            'speed': ['fast', 'slow', 'quick', 'rapid', 'delay', 'wait', 'time', 'immediate'],
            'reliability': ['reliable', 'unreliable', 'consistent', 'works', 'broken', 'failed']
        }

        # Initialize NLTK components if available
        self.stop_words = set()
        if NLTK_AVAILABLE:
            try:
                self.stop_words = set(stopwords.words('english'))
            except Exception as e:
                logger.warning(f"Could not load stopwords: {e}")

        # Stats tracking
        self.stats = {
            'total_analyzed': 0,
            'themes_extracted': 0,
            'emotions_detected': 0,
            'entities_found': 0,
            'high_intensity_count': 0
        }

        logger.info("InsightAgent initialized")

    def extract_insights(self, preprocessed_data: Dict[str, Any], sentiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main insight extraction function.

        Args:
            preprocessed_data: Output from PreprocessingAgent
            sentiment_results: Output from SentimentAgent

        Returns:
            Dictionary containing extracted insights
        """
        try:
            processed_entries = preprocessed_data.get('processed_entries', [])
            sentiment_data = sentiment_results.get('results', [])

            if not processed_entries or not sentiment_data:
                logger.warning("No data available for insight extraction")
                return self._empty_results()

            logger.info(f"Extracting insights from {len(processed_entries)} entries")

            # Reset stats
            self._reset_stats()
            self.stats['total_analyzed'] = len(processed_entries)

            # Combine preprocessed and sentiment data
            combined_data = self._combine_data(processed_entries, sentiment_data)

            insights = {
                'themes': {},
                'emotions': {},
                'intensity_analysis': {},
                'entities': {},
                'patterns': {},
                'actionable_insights': [],
                'metadata': {}
            }

            # Extract themes
            if self.config.get('extract_themes', True):
                insights['themes'] = self._extract_themes(combined_data)
                self.stats['themes_extracted'] = len(insights['themes'].get('theme_analysis', {}))

            # Detect emotions
            if self.config.get('emotion_detection', True):
                insights['emotions'] = self._detect_emotions(combined_data)
                self.stats['emotions_detected'] = sum(
                    data['count'] for data in insights['emotions'].get('emotion_distribution', {}).values()
                )

            # Analyze intensity
            if self.config.get('intensity_analysis', True):
                insights['intensity_analysis'] = self._analyze_intensity(combined_data)
                self.stats['high_intensity_count'] = insights['intensity_analysis'].get('high_intensity_count', 0)

            # Extract entities
            if self.config.get('extract_entities', True):
                insights['entities'] = self._extract_entities(combined_data)
                self.stats['entities_found'] = len(insights['entities'].get('named_entities', []))

            # Identify patterns
            insights['patterns'] = self._identify_patterns(combined_data)

            # Generate actionable insights
            insights['actionable_insights'] = self._generate_actionable_insights(insights, combined_data)

            # Add metadata
            insights['metadata'] = {
                'extraction_timestamp': datetime.now().isoformat(),
                'total_entries_analyzed': len(combined_data),
                'config_used': self.config.copy(),
                'stats': self.stats.copy()
            }

            logger.info(f"Insight extraction completed. Found {self.stats['themes_extracted']} themes, "
                       f"{self.stats['emotions_detected']} emotions, {self.stats['entities_found']} entities")

            return insights

        except Exception as e:
            logger.error(f"Error in insight extraction: {str(e)}")
            raise

    def _combine_data(self, processed_entries: List[Dict], sentiment_data: List[Dict]) -> List[Dict]:
        """Combine preprocessed and sentiment data"""
        combined = []

        # Create mapping from original index to sentiment data
        sentiment_map = {s.get('original_index'): s for s in sentiment_data}

        for entry in processed_entries:
            original_index = entry.get('original_index')
            sentiment_info = sentiment_map.get(original_index, {})

            combined_entry = {
                'original_text': entry.get('original_text', ''),
                'cleaned_text': entry.get('cleaned_text', ''),
                'sentiment': sentiment_info.get('final_sentiment', 'neutral'),
                'confidence': sentiment_info.get('confidence', 0.0),
                'sentiment_scores': sentiment_info.get('sentiment_scores', {}),
                'source_type': entry.get('source_type', 'text')
            }

            combined.append(combined_entry)

        return combined

    def _extract_themes(self, data: List[Dict]) -> Dict[str, Any]:
        """Extract themes from the text data"""
        theme_counts = defaultdict(int)
        theme_sentiments = defaultdict(list)
        theme_examples = defaultdict(list)

        for entry in data:
            text = entry['cleaned_text'].lower()
            sentiment = entry['sentiment']

            # Check for predefined service themes
            for theme_name, keywords in self.service_themes.items():
                if any(keyword in text for keyword in keywords):
                    theme_counts[theme_name] += 1
                    theme_sentiments[theme_name].append(sentiment)
                    if len(theme_examples[theme_name]) < 3:  # Keep up to 3 examples
                        theme_examples[theme_name].append(entry['original_text'][:150])

            # Extract noun phrases as potential themes
            if NLTK_AVAILABLE:
                try:
                    tokens = word_tokenize(text)
                    pos_tags = pos_tag(tokens)

                    # Look for noun phrases
                    noun_phrases = []
                    current_phrase = []

                    for word, pos in pos_tags:
                        if pos.startswith('NN') or pos.startswith('JJ'):  # Nouns and adjectives
                            current_phrase.append(word)
                        else:
                            if len(current_phrase) > 1:
                                phrase = ' '.join(current_phrase)
                                if phrase not in self.stop_words and len(phrase) > 3:
                                    noun_phrases.append(phrase)
                            current_phrase = []

                    # Add final phrase if exists
                    if len(current_phrase) > 1:
                        phrase = ' '.join(current_phrase)
                        if phrase not in self.stop_words and len(phrase) > 3:
                            noun_phrases.append(phrase)

                    # Count noun phrases as themes
                    for phrase in noun_phrases:
                        theme_counts[f"auto_{phrase}"] += 1
                        theme_sentiments[f"auto_{phrase}"].append(sentiment)
                        if len(theme_examples[f"auto_{phrase}"]) < 3:
                            theme_examples[f"auto_{phrase}"].append(entry['original_text'][:150])

                except Exception as e:
                    logger.warning(f"Error in noun phrase extraction: {e}")

        # Filter themes by minimum frequency
        min_freq = self.config.get('min_theme_frequency', 2)
        filtered_themes = {k: v for k, v in theme_counts.items() if v >= min_freq}

        # Calculate sentiment distribution for each theme
        theme_analysis = {}
        for theme, count in filtered_themes.items():
            sentiments = theme_sentiments[theme]
            sentiment_dist = {
                'positive': sentiments.count('positive') / len(sentiments),
                'negative': sentiments.count('negative') / len(sentiments), 
                'neutral': sentiments.count('neutral') / len(sentiments)
            }

            theme_analysis[theme] = {
                'frequency': count,
                'sentiment_distribution': sentiment_dist,
                'dominant_sentiment': max(sentiment_dist, key=sentiment_dist.get),
                'examples': theme_examples[theme]
            }

        # Get top themes
        top_themes = sorted(theme_analysis.items(), key=lambda x: x[1]['frequency'], reverse=True)
        top_themes = top_themes[:self.config.get('top_themes_count', 10)]

        return {
            'theme_analysis': dict(top_themes),
            'total_themes_found': len(theme_analysis),
            'theme_frequency_distribution': dict(filtered_themes)
        }

    def _detect_emotions(self, data: List[Dict]) -> Dict[str, Any]:
        """Detect emotions in the text data"""
        emotion_counts = defaultdict(int)
        emotion_examples = defaultdict(list)
        emotion_intensities = defaultdict(list)

        for entry in data:
            text = entry['cleaned_text'].lower()
            confidence = entry.get('confidence', 0.0)

            detected_emotions = []

            # Check for emotion keywords
            for emotion, keywords in self.emotion_keywords.items():
                emotion_score = 0
                for keyword in keywords:
                    if keyword in text:
                        emotion_score += 1

                if emotion_score > 0:
                    # Weight by sentiment confidence
                    weighted_score = emotion_score * confidence
                    emotion_counts[emotion] += weighted_score
                    emotion_intensities[emotion].append(weighted_score)
                    detected_emotions.append(emotion)

                    if len(emotion_examples[emotion]) < 3:
                        emotion_examples[emotion].append(entry['original_text'][:150])

        # Calculate emotion distribution
        total_emotion_score = sum(emotion_counts.values())
        emotion_distribution = {}

        for emotion, count in emotion_counts.items():
            distribution_score = count / total_emotion_score if total_emotion_score > 0 else 0
            avg_intensity = np.mean(emotion_intensities[emotion]) if emotion_intensities[emotion] else 0

            emotion_distribution[emotion] = {
                'count': count,
                'percentage': distribution_score,
                'average_intensity': avg_intensity,
                'examples': emotion_examples[emotion]
            }

        # Identify dominant emotion
        dominant_emotion = max(emotion_distribution.keys(), key=lambda k: emotion_distribution[k]['count']) if emotion_distribution else None

        return {
            'emotion_distribution': emotion_distribution,
            'dominant_emotion': dominant_emotion,
            'total_emotions_detected': len(emotion_distribution),
            'emotion_summary': self._create_emotion_summary(emotion_distribution)
        }

    def _analyze_intensity(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment intensity"""
        intensity_levels = defaultdict(int)
        high_intensity_examples = []

        for entry in data:
            text = entry['cleaned_text'].lower()
            confidence = entry.get('confidence', 0.0)
            sentiment = entry.get('sentiment', 'neutral')

            # Determine intensity based on modifiers and confidence
            intensity_score = confidence

            # Check for intensity modifiers
            for level, modifiers in self.intensity_modifiers.items():
                for modifier in modifiers:
                    if modifier in text:
                        if level == 'very_high':
                            intensity_score = min(intensity_score * 1.5, 1.0)
                        elif level == 'high':
                            intensity_score = min(intensity_score * 1.3, 1.0)
                        elif level == 'medium':
                            intensity_score = min(intensity_score * 1.1, 1.0)
                        # Low modifiers might reduce intensity
                        elif level == 'low':
                            intensity_score = max(intensity_score * 0.8, 0.1)
                        break

            # Categorize intensity
            if intensity_score >= 0.8:
                intensity_levels['very_high'] += 1
                if sentiment != 'neutral':
                    high_intensity_examples.append({
                        'text': entry['original_text'][:200],
                        'sentiment': sentiment,
                        'intensity_score': intensity_score
                    })
            elif intensity_score >= 0.6:
                intensity_levels['high'] += 1
            elif intensity_score >= 0.4:
                intensity_levels['medium'] += 1
            else:
                intensity_levels['low'] += 1

        total = sum(intensity_levels.values())
        intensity_distribution = {level: count/total for level, count in intensity_levels.items()} if total > 0 else {}

        return {
            'intensity_distribution': intensity_distribution,
            'intensity_counts': dict(intensity_levels),
            'high_intensity_count': intensity_levels['very_high'] + intensity_levels['high'],
            'high_intensity_examples': high_intensity_examples[:5],  # Top 5 examples
            'average_intensity': np.mean([entry.get('confidence', 0.0) for entry in data])
        }

    def _extract_entities(self, data: List[Dict]) -> Dict[str, Any]:
        """Extract named entities from text"""
        entities = defaultdict(list)
        entity_sentiments = defaultdict(list)

        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available, skipping entity extraction")
            return {'named_entities': [], 'entity_analysis': {}}

        for entry in data:
            text = entry['cleaned_text']
            sentiment = entry['sentiment']

            try:
                # Tokenize and tag
                tokens = word_tokenize(text)
                pos_tags = pos_tag(tokens)

                # Named entity recognition
                chunks = ne_chunk(pos_tags, binary=False)

                # Extract entities
                for chunk in chunks:
                    if hasattr(chunk, 'label'):
                        entity_name = ' '.join([token for token, pos in chunk.leaves()])
                        entity_type = chunk.label()
                        entities[entity_type].append(entity_name)
                        entity_sentiments[f"{entity_type}_{entity_name}"].append(sentiment)

            except Exception as e:
                logger.warning(f"Error in entity extraction: {e}")
                continue

        # Process entity analysis
        entity_analysis = {}
        all_entities = []

        for entity_type, entity_list in entities.items():
            entity_counter = Counter(entity_list)
            for entity, count in entity_counter.most_common(10):  # Top 10 per type
                entity_key = f"{entity_type}_{entity}"
                sentiments = entity_sentiments.get(entity_key, [])

                if sentiments:
                    sentiment_dist = {
                        'positive': sentiments.count('positive') / len(sentiments),
                        'negative': sentiments.count('negative') / len(sentiments),
                        'neutral': sentiments.count('neutral') / len(sentiments)
                    }
                else:
                    sentiment_dist = {'positive': 0, 'negative': 0, 'neutral': 1}

                entity_info = {
                    'entity': entity,
                    'type': entity_type,
                    'frequency': count,
                    'sentiment_distribution': sentiment_dist,
                    'dominant_sentiment': max(sentiment_dist, key=sentiment_dist.get)
                }

                entity_analysis[entity_key] = entity_info
                all_entities.append(entity_info)

        return {
            'named_entities': all_entities,
            'entity_analysis': entity_analysis,
            'entity_types_found': list(entities.keys()),
            'total_entities': len(all_entities)
        }

    def _identify_patterns(self, data: List[Dict]) -> Dict[str, Any]:
        """Identify patterns in the data"""
        patterns = {}

        # Sentiment patterns
        sentiments = [entry['sentiment'] for entry in data]
        patterns['sentiment_pattern'] = {
            'most_common': max(set(sentiments), key=sentiments.count) if sentiments else 'neutral',
            'distribution': {
                sentiment: sentiments.count(sentiment) / len(sentiments) 
                for sentiment in set(sentiments)
            } if sentiments else {}
        }

        # Confidence patterns
        confidences = [entry.get('confidence', 0.0) for entry in data]
        patterns['confidence_pattern'] = {
            'average': np.mean(confidences) if confidences else 0.0,
            'high_confidence_ratio': len([c for c in confidences if c >= 0.7]) / len(confidences) if confidences else 0.0,
            'low_confidence_ratio': len([c for c in confidences if c < 0.4]) / len(confidences) if confidences else 0.0
        }

        # Text length patterns
        text_lengths = [len(entry['cleaned_text']) for entry in data]
        patterns['text_length_pattern'] = {
            'average_length': np.mean(text_lengths) if text_lengths else 0,
            'length_vs_sentiment': self._analyze_length_sentiment_correlation(data)
        }

        return patterns

    def _analyze_length_sentiment_correlation(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze correlation between text length and sentiment"""
        sentiment_lengths = {'positive': [], 'negative': [], 'neutral': []}

        for entry in data:
            length = len(entry['cleaned_text'])
            sentiment = entry['sentiment']
            sentiment_lengths[sentiment].append(length)

        avg_lengths = {}
        for sentiment, lengths in sentiment_lengths.items():
            avg_lengths[sentiment] = np.mean(lengths) if lengths else 0

        return {
            'average_lengths_by_sentiment': avg_lengths,
            'longest_sentiment': max(avg_lengths, key=avg_lengths.get) if avg_lengths else None
        }

    def _generate_actionable_insights(self, insights: Dict[str, Any], data: List[Dict]) -> List[Dict[str, Any]]:
        """Generate actionable insights from the analysis"""
        actionable = []

        # Theme-based insights
        themes = insights.get('themes', {}).get('theme_analysis', {})
        for theme, analysis in themes.items():
            if analysis['frequency'] > len(data) * 0.1:  # If theme appears in >10% of feedback
                dominant_sentiment = analysis['dominant_sentiment']
                if dominant_sentiment == 'negative':
                    actionable.append({
                        'type': 'theme_improvement',
                        'priority': 'high',
                        'theme': theme,
                        'issue': f"Frequent negative feedback about {theme}",
                        'recommendation': f"Focus on improving {theme} to address customer concerns",
                        'affected_percentage': (analysis['frequency'] / len(data)) * 100
                    })

        # Emotion-based insights
        emotions = insights.get('emotions', {}).get('emotion_distribution', {})
        if 'anger' in emotions and emotions['anger']['percentage'] > 0.2:
            actionable.append({
                'type': 'emotion_alert',
                'priority': 'high', 
                'emotion': 'anger',
                'issue': 'High levels of customer anger detected',
                'recommendation': 'Implement immediate customer service intervention protocols',
                'affected_percentage': emotions['anger']['percentage'] * 100
            })

        # Intensity-based insights
        intensity = insights.get('intensity_analysis', {})
        if intensity.get('high_intensity_count', 0) > len(data) * 0.3:
            actionable.append({
                'type': 'intensity_alert',
                'priority': 'medium',
                'issue': 'High intensity feedback detected',
                'recommendation': 'Review high-intensity feedback for urgent issues',
                'affected_count': intensity.get('high_intensity_count', 0)
            })

        # Overall sentiment insights
        patterns = insights.get('patterns', {})
        sentiment_pattern = patterns.get('sentiment_pattern', {})
        if sentiment_pattern.get('most_common') == 'negative':
            negative_ratio = sentiment_pattern.get('distribution', {}).get('negative', 0)
            if negative_ratio > 0.6:
                actionable.append({
                    'type': 'overall_sentiment',
                    'priority': 'critical',
                    'issue': 'Majority of feedback is negative',
                    'recommendation': 'Implement comprehensive customer experience improvement plan',
                    'negative_percentage': negative_ratio * 100
                })

        return actionable

    def _create_emotion_summary(self, emotion_distribution: Dict[str, Any]) -> str:
        """Create a summary of emotions detected"""
        if not emotion_distribution:
            return "No specific emotions detected in the feedback."

        sorted_emotions = sorted(emotion_distribution.items(), key=lambda x: x[1]['count'], reverse=True)
        top_emotion = sorted_emotions[0]

        summary = f"The dominant emotion is {top_emotion[0]} ({top_emotion[1]['percentage']:.1%} of emotional content)"

        if len(sorted_emotions) > 1:
            second_emotion = sorted_emotions[1]
            summary += f", followed by {second_emotion[0]} ({second_emotion[1]['percentage']:.1%})"

        return summary + "."

    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure"""
        return {
            'themes': {'theme_analysis': {}, 'total_themes_found': 0},
            'emotions': {'emotion_distribution': {}, 'dominant_emotion': None},
            'intensity_analysis': {'intensity_distribution': {}, 'high_intensity_count': 0},
            'entities': {'named_entities': [], 'entity_analysis': {}},
            'patterns': {},
            'actionable_insights': [],
            'metadata': {'total_entries_analyzed': 0, 'stats': self.stats}
        }

    def _reset_stats(self):
        """Reset processing statistics"""
        for key in self.stats:
            self.stats[key] = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get insight extraction statistics"""
        return self.stats.copy()

    def reset(self):
        """Reset agent to initial state"""
        self._reset_stats()
        logger.info("InsightAgent reset")
