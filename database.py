import sqlite3
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Database manager for the AuralyTica Tel AI system.
    Handles SQLite database operations for storing feedback and analysis results.
    """

    def __init__(self, db_path: str = 'auralytica.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database with required tables"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Create feedback table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feedback_id TEXT UNIQUE,
                    original_text TEXT NOT NULL,
                    cleaned_text TEXT,
                    source_type TEXT DEFAULT 'text',
                    timestamp TEXT,
                    customer_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                ''')

                # Create sentiment_results table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feedback_id TEXT,
                    sentiment TEXT NOT NULL,
                    confidence REAL,
                    models_used TEXT,
                    sentiment_scores TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (feedback_id) REFERENCES feedback (feedback_id)
                )
                ''')

                # Create insights table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT,
                    themes TEXT,
                    emotions TEXT,
                    entities TEXT,
                    intensity_data TEXT,
                    actionable_insights TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                ''')

                # Create processing_history table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT UNIQUE,
                    total_entries INTEGER,
                    processing_time REAL,
                    success_rate REAL,
                    config_used TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                ''')

                conn.commit()
                logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def store_feedback(self, feedback_data: List[Dict[str, Any]]) -> bool:
        """Store feedback data in database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                for entry in feedback_data:
                    cursor.execute('''
                    INSERT OR REPLACE INTO feedback 
                    (feedback_id, original_text, cleaned_text, source_type, timestamp, customer_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        entry.get('id', ''),
                        entry.get('original_text', ''),
                        entry.get('cleaned_text', ''),
                        entry.get('source_type', 'text'),
                        entry.get('timestamp', datetime.now().isoformat()),
                        entry.get('customer_id', '')
                    ))

                conn.commit()
                logger.info(f"Stored {len(feedback_data)} feedback entries")
                return True

        except Exception as e:
            logger.error(f"Error storing feedback: {e}")
            return False

    def store_sentiment_results(self, results: List[Dict[str, Any]]) -> bool:
        """Store sentiment analysis results"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                for result in results:
                    cursor.execute('''
                    INSERT INTO sentiment_results 
                    (feedback_id, sentiment, confidence, models_used, sentiment_scores)
                    VALUES (?, ?, ?, ?, ?)
                    ''', (
                        result.get('original_index', ''),
                        result.get('final_sentiment', ''),
                        result.get('confidence', 0.0),
                        json.dumps(result.get('models_used', [])),
                        json.dumps(result.get('sentiment_scores', {}))
                    ))

                conn.commit()
                logger.info(f"Stored {len(results)} sentiment results")
                return True

        except Exception as e:
            logger.error(f"Error storing sentiment results: {e}")
            return False

    def store_insights(self, analysis_id: str, insights: Dict[str, Any]) -> bool:
        """Store insight analysis results"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute('''
                INSERT OR REPLACE INTO insights 
                (analysis_id, themes, emotions, entities, intensity_data, actionable_insights)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    analysis_id,
                    json.dumps(insights.get('themes', {})),
                    json.dumps(insights.get('emotions', {})),
                    json.dumps(insights.get('entities', {})),
                    json.dumps(insights.get('intensity_analysis', {})),
                    json.dumps(insights.get('actionable_insights', []))
                ))

                conn.commit()
                logger.info(f"Stored insights for analysis {analysis_id}")
                return True

        except Exception as e:
            logger.error(f"Error storing insights: {e}")
            return False

    def store_processing_history(self, analysis_id: str, metadata: Dict[str, Any]) -> bool:
        """Store processing history and metadata"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute('''
                INSERT OR REPLACE INTO processing_history 
                (analysis_id, total_entries, processing_time, success_rate, config_used)
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    analysis_id,
                    metadata.get('total_entries', 0),
                    metadata.get('processing_time_seconds', 0.0),
                    metadata.get('success_rate', 1.0),
                    json.dumps(metadata.get('config_used', {}))
                ))

                conn.commit()
                logger.info(f"Stored processing history for analysis {analysis_id}")
                return True

        except Exception as e:
            logger.error(f"Error storing processing history: {e}")
            return False

    def get_feedback_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get feedback history from database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT feedback_id, original_text, source_type, timestamp, created_at
                FROM feedback 
                ORDER BY created_at DESC 
                LIMIT ?
                ''', (limit,))

                rows = cursor.fetchall()
                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Error retrieving feedback history: {e}")
            return []

    def get_sentiment_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get sentiment statistics from recent analysis"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Get sentiment distribution
                cursor.execute('''
                SELECT sentiment, COUNT(*) as count
                FROM sentiment_results 
                WHERE datetime(created_at) >= datetime('now', '-{} days')
                GROUP BY sentiment
                '''.format(days))

                sentiment_counts = dict(cursor.fetchall())
                total = sum(sentiment_counts.values())

                if total == 0:
                    return {'sentiment_distribution': {}, 'total_analyzed': 0}

                sentiment_distribution = {
                    sentiment: count / total 
                    for sentiment, count in sentiment_counts.items()
                }

                # Get average confidence
                cursor.execute('''
                SELECT AVG(confidence) as avg_confidence
                FROM sentiment_results 
                WHERE datetime(created_at) >= datetime('now', '-{} days')
                '''.format(days))

                avg_confidence = cursor.fetchone()[0] or 0.0

                return {
                    'sentiment_distribution': sentiment_distribution,
                    'sentiment_counts': sentiment_counts,
                    'total_analyzed': total,
                    'average_confidence': avg_confidence
                }

        except Exception as e:
            logger.error(f"Error retrieving sentiment statistics: {e}")
            return {'sentiment_distribution': {}, 'total_analyzed': 0}

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing performance statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Get recent processing stats
                cursor.execute('''
                SELECT 
                    COUNT(*) as total_analyses,
                    AVG(processing_time) as avg_processing_time,
                    AVG(success_rate) as avg_success_rate,
                    SUM(total_entries) as total_entries_processed
                FROM processing_history 
                WHERE datetime(created_at) >= datetime('now', '-30 days')
                ''')

                row = cursor.fetchone()

                return {
                    'total_analyses': row[0] or 0,
                    'avg_processing_time': row[1] or 0.0,
                    'avg_success_rate': row[2] or 1.0,
                    'total_entries_processed': row[3] or 0
                }

        except Exception as e:
            logger.error(f"Error retrieving processing statistics: {e}")
            return {
                'total_analyses': 0,
                'avg_processing_time': 0.0,
                'avg_success_rate': 1.0,
                'total_entries_processed': 0
            }

    def export_data_to_csv(self, table: str, file_path: str) -> bool:
        """Export table data to CSV file"""
        try:
            with self.get_connection() as conn:
                df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                df.to_csv(file_path, index=False)
                logger.info(f"Exported {len(df)} rows from {table} to {file_path}")
                return True

        except Exception as e:
            logger.error(f"Error exporting data to CSV: {e}")
            return False

    def cleanup_old_data(self, days: int = 90) -> bool:
        """Clean up old data from database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Delete old processing history
                cursor.execute('''
                DELETE FROM processing_history 
                WHERE datetime(created_at) < datetime('now', '-{} days')
                '''.format(days))

                # Delete old sentiment results (keeping feedback)
                cursor.execute('''
                DELETE FROM sentiment_results 
                WHERE datetime(created_at) < datetime('now', '-{} days')
                '''.format(days))

                # Delete old insights
                cursor.execute('''
                DELETE FROM insights 
                WHERE datetime(created_at) < datetime('now', '-{} days')
                '''.format(days))

                conn.commit()
                logger.info(f"Cleaned up data older than {days} days")
                return True

        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return False

    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Get table counts
                tables = ['feedback', 'sentiment_results', 'insights', 'processing_history']
                table_counts = {}

                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    table_counts[table] = cursor.fetchone()[0]

                # Get database size (approximate)
                cursor.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                cursor.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                db_size = page_size * page_count

                return {
                    'database_path': self.db_path,
                    'table_counts': table_counts,
                    'database_size_bytes': db_size,
                    'database_size_mb': round(db_size / (1024 * 1024), 2)
                }

        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {'database_path': self.db_path, 'error': str(e)}
