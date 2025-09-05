import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union
import string
import logging
from collections import Counter

logger = logging.getLogger(__name__)

class PreprocessingAgent:
    """
    Preprocessing Agent that cleans, deduplicates, and normalizes customer feedback.
    First stage of the AuralyTica Tel AI pipeline.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'clean_text': True,
            'remove_duplicates': True,
            'normalize': True,
            'min_length': 3,
            'max_length': 10000,
            'remove_urls': True,
            'remove_emails': True,
            'remove_extra_whitespace': True
        }

        # Stats tracking
        self.stats = {
            'total_processed': 0,
            'duplicates_removed': 0,
            'empty_removed': 0,
            'too_short_removed': 0,
            'too_long_removed': 0,
            'urls_removed': 0,
            'emails_removed': 0
        }

        logger.info("PreprocessingAgent initialized")

    def process(self, feedback_list: List[str], source_type: str = 'text') -> Dict[str, Any]:
        """
        Main preprocessing function that cleans and normalizes feedback data.

        Args:
            feedback_list: List of raw feedback strings
            source_type: Type of source data

        Returns:
            Dictionary containing processed data and metadata
        """
        try:
            logger.info(f"Processing {len(feedback_list)} feedback entries")

            # Reset stats
            self._reset_stats()
            self.stats['total_processed'] = len(feedback_list)

            # Step 1: Remove duplicates
            if self.config.get('remove_duplicates', True):
                feedback_list = self._remove_duplicates(feedback_list)

            # Step 2: Clean each text entry
            cleaned_feedback = []
            for i, feedback in enumerate(feedback_list):
                if feedback is None or pd.isna(feedback):
                    self.stats['empty_removed'] += 1
                    continue

                # Convert to string if not already
                feedback = str(feedback)

                # Clean the text
                cleaned = self._clean_text(feedback)

                # Length filtering
                if len(cleaned) < self.config.get('min_length', 3):
                    self.stats['too_short_removed'] += 1
                    continue
                elif len(cleaned) > self.config.get('max_length', 10000):
                    self.stats['too_long_removed'] += 1
                    continue

                cleaned_feedback.append({
                    'original_index': i,
                    'original_text': feedback,
                    'cleaned_text': cleaned,
                    'source_type': source_type
                })

            # Step 3: Extract metadata
            metadata = self._extract_metadata(cleaned_feedback)

            logger.info(f"Preprocessing completed. {len(cleaned_feedback)} entries remain after cleaning")

            return {
                'processed_entries': cleaned_feedback,
                'metadata': metadata,
                'stats': self.stats.copy(),
                'total_entries': len(cleaned_feedback),
                'source_type': source_type
            }

        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def _clean_text(self, text: str) -> str:
        """Clean individual text entry"""
        if not text:
            return ""

        original_text = text

        # Remove URLs
        if self.config.get('remove_urls', True):
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            if re.search(url_pattern, text):
                self.stats['urls_removed'] += 1
            text = re.sub(url_pattern, ' ', text)

        # Remove email addresses
        if self.config.get('remove_emails', True):
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            if re.search(email_pattern, text):
                self.stats['emails_removed'] += 1
            text = re.sub(email_pattern, ' ', text)

        if self.config.get('clean_text', True):
            # Remove special characters but keep basic punctuation
            text = re.sub(r'[^\w\s.,!?;:()\'-]', ' ', text)

            # Remove extra whitespace
            if self.config.get('remove_extra_whitespace', True):
                text = re.sub(r'\s+', ' ', text).strip()

            # Fix common encoding issues
            text = text.replace('\'', "'").replace('\"', '"')
            text = text.replace('â\x80\x99', "'").replace('â\x80\x9c', '"').replace('â\x80\x9d', '"')

        return text

    def _remove_duplicates(self, feedback_list: List[str]) -> List[str]:
        """Remove duplicate feedback entries"""
        original_count = len(feedback_list)

        # Convert to set to remove exact duplicates, then back to list
        unique_feedback = list(dict.fromkeys(feedback_list))  # Preserves order

        duplicates_removed = original_count - len(unique_feedback)
        self.stats['duplicates_removed'] = duplicates_removed

        logger.info(f"Removed {duplicates_removed} duplicate entries")

        return unique_feedback

    def _extract_metadata(self, processed_entries: List[Dict]) -> Dict[str, Any]:
        """Extract metadata from processed entries"""
        if not processed_entries:
            return {}

        texts = [entry['cleaned_text'] for entry in processed_entries]

        # Calculate text statistics
        text_lengths = [len(text) for text in texts]
        word_counts = [len(text.split()) for text in texts]

        # Extract common words (simple approach)
        all_words = []
        for text in texts:
            words = text.lower().split()
            # Filter out common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
            filtered_words = [word.strip(string.punctuation) for word in words if word.lower() not in stop_words and len(word) > 2]
            all_words.extend(filtered_words)

        word_freq = Counter(all_words)

        metadata = {
            'text_statistics': {
                'avg_length': np.mean(text_lengths),
                'min_length': min(text_lengths) if text_lengths else 0,
                'max_length': max(text_lengths) if text_lengths else 0,
                'avg_word_count': np.mean(word_counts),
                'total_words': sum(word_counts)
            },
            'common_words': word_freq.most_common(10),
            'language_indicators': self._detect_language_indicators(texts),
            'preprocessing_summary': {
                'original_entries': self.stats['total_processed'],
                'final_entries': len(processed_entries),
                'removal_stats': {
                    'duplicates': self.stats['duplicates_removed'],
                    'empty': self.stats['empty_removed'],
                    'too_short': self.stats['too_short_removed'],
                    'too_long': self.stats['too_long_removed']
                }
            }
        }

        return metadata

    def _detect_language_indicators(self, texts: List[str]) -> Dict[str, Any]:
        """Simple language detection based on common words"""
        english_indicators = ['the', 'and', 'is', 'was', 'are', 'were', 'a', 'an', 'this', 'that']

        english_score = 0
        total_words = 0

        for text in texts:
            words = text.lower().split()
            total_words += len(words)
            for word in words:
                if word in english_indicators:
                    english_score += 1

        english_ratio = english_score / total_words if total_words > 0 else 0

        return {
            'likely_language': 'english' if english_ratio > 0.05 else 'unknown',
            'english_indicator_ratio': english_ratio,
            'confidence': 'high' if english_ratio > 0.1 else 'medium' if english_ratio > 0.05 else 'low'
        }

    def _reset_stats(self):
        """Reset processing statistics"""
        for key in self.stats:
            self.stats[key] = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics"""
        return self.stats.copy()

    def reset(self):
        """Reset agent to initial state"""
        self._reset_stats()
        logger.info("PreprocessingAgent reset")
