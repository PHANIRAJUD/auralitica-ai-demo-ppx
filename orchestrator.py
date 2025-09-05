from typing import Dict, List, Any, Union
import pandas as pd
from datetime import datetime
import logging

from .preprocessing_agent import PreprocessingAgent
from .sentiment_agent import SentimentAgent
from .insight_agent import InsightAgent
from .visualization_agent import VisualizationAgent

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """
    Main orchestrator that coordinates all agents in the AuralyTica Tel AI system.
    Implements the multi-agent architecture for customer sentiment analysis.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Initialize agents
        self.preprocessing_agent = PreprocessingAgent()
        self.sentiment_agent = SentimentAgent()
        self.insight_agent = InsightAgent()
        self.visualization_agent = VisualizationAgent()

        # Agent status tracking
        self.agent_status = {
            'preprocessing': 'initialized',
            'sentiment': 'initialized', 
            'insight': 'initialized',
            'visualization': 'initialized'
        }

        logger.info("AgentOrchestrator initialized with all agents")

    def process_feedback(self, feedback: Union[str, List[str]], source_type: str = 'text') -> Dict[str, Any]:
        """
        Process single or multiple feedback entries through the agent pipeline.

        Args:
            feedback: Single feedback string or list of feedback strings
            source_type: Type of source data (text, email, chat, etc.)

        Returns:
            Dictionary containing results from all agents
        """
        try:
            # Start processing timestamp
            start_time = datetime.now()

            # Convert single feedback to list for uniform processing
            if isinstance(feedback, str):
                feedback_list = [feedback]
                is_single = True
            else:
                feedback_list = feedback
                is_single = False

            # Step 1: Preprocessing Agent
            logger.info("Starting preprocessing phase")
            self.agent_status['preprocessing'] = 'running'

            processed_data = self.preprocessing_agent.process(
                feedback_list, 
                source_type=source_type
            )

            self.agent_status['preprocessing'] = 'completed'

            # Step 2: Sentiment Analysis Agent
            logger.info("Starting sentiment analysis phase")
            self.agent_status['sentiment'] = 'running'

            sentiment_results = self.sentiment_agent.analyze(processed_data)

            self.agent_status['sentiment'] = 'completed'

            # Step 3: Insight Extraction Agent
            logger.info("Starting insight extraction phase")
            self.agent_status['insight'] = 'running'

            insights = self.insight_agent.extract_insights(
                processed_data,
                sentiment_results
            )

            self.agent_status['insight'] = 'completed'

            # Step 4: Visualization Agent
            logger.info("Starting visualization phase")
            self.agent_status['visualization'] = 'running'

            visualizations = self.visualization_agent.create_visualizations(
                processed_data,
                sentiment_results,
                insights
            )

            self.agent_status['visualization'] = 'completed'

            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # Compile final results
            results = {
                'metadata': {
                    'total_entries': len(feedback_list),
                    'source_type': source_type,
                    'processing_time_seconds': processing_time,
                    'timestamp': end_time.isoformat(),
                    'is_single_entry': is_single
                },
                'preprocessing': {
                    'original_data': feedback_list,
                    'processed_data': processed_data,
                    'preprocessing_stats': self.preprocessing_agent.get_stats()
                },
                'sentiment_analysis': sentiment_results,
                'insights': insights,
                'visualizations': visualizations,
                'agent_status': self.agent_status.copy()
            }

            logger.info(f"Processing completed in {processing_time:.2f} seconds")

            return results

        except Exception as e:
            logger.error(f"Error in agent orchestration: {str(e)}")
            # Reset agent status on error
            for agent in self.agent_status:
                if self.agent_status[agent] == 'running':
                    self.agent_status[agent] = 'error'
            raise

    def process_batch(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process batch data from DataFrame through the agent pipeline.

        Args:
            df: DataFrame containing feedback data

        Returns:
            Dictionary containing batch processing results
        """
        try:
            if 'feedback' not in df.columns:
                raise ValueError("DataFrame must contain 'feedback' column")

            feedback_list = df['feedback'].dropna().tolist()
            source_type = df.get('source_type', ['text'] * len(feedback_list))

            if isinstance(source_type, pd.Series):
                source_type = source_type.tolist()
            elif isinstance(source_type, str):
                source_type = [source_type] * len(feedback_list)

            # Process all feedback entries
            results = self.process_feedback(feedback_list, 'batch')

            # Add batch-specific metadata
            results['batch_metadata'] = {
                'total_rows': len(df),
                'processed_rows': len(feedback_list),
                'skipped_rows': len(df) - len(feedback_list),
                'columns': df.columns.tolist()
            }

            return results

        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise

    def get_agent_status(self) -> Dict[str, str]:
        """Get current status of all agents"""
        return self.agent_status.copy()

    def reset_agents(self):
        """Reset all agents to initial state"""
        self.preprocessing_agent.reset()
        self.sentiment_agent.reset()
        self.insight_agent.reset()
        self.visualization_agent.reset()

        for agent in self.agent_status:
            self.agent_status[agent] = 'initialized'

        logger.info("All agents reset to initial state")

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of the processing pipeline"""
        return {
            'agents': [
                {
                    'name': 'Preprocessing Agent',
                    'description': 'Cleans, deduplicates, and normalizes feedback',
                    'status': self.agent_status['preprocessing']
                },
                {
                    'name': 'Sentiment Analysis Agent', 
                    'description': 'Classifies feedback sentiment (positive, negative, neutral)',
                    'status': self.agent_status['sentiment']
                },
                {
                    'name': 'Insight Extraction Agent',
                    'description': 'Extracts themes, emotions, and intensity',
                    'status': self.agent_status['insight']
                },
                {
                    'name': 'Visualization Agent',
                    'description': 'Creates graphs, dashboards, and alerts',
                    'status': self.agent_status['visualization']
                }
            ],
            'technical_components': ['Python', 'NLP', 'Azure', 'LangChain'],
            'infrastructure': 'Cloud-based'
        }
