import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""

    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    DEBUG = os.environ.get('DEBUG', 'True').lower() in ['true', '1', 'yes']

    # OpenAI API settings (for LangChain)
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

    # Azure AI Services (optional)
    AZURE_AI_ENDPOINT = os.environ.get('AZURE_AI_ENDPOINT')
    AZURE_AI_KEY = os.environ.get('AZURE_AI_KEY')

    # Database settings (if using a database)
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///auralytica.db'

    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = 'uploads'

    # Sentiment analysis settings
    SENTIMENT_MODELS = {
        'primary': 'vader',  # vader, textblob, transformers, azure
        'secondary': 'textblob',
        'enable_azure': False
    }

    # Agent settings
    AGENT_CONFIG = {
        'preprocessing': {
            'clean_text': True,
            'remove_duplicates': True,
            'normalize': True
        },
        'sentiment': {
            'confidence_threshold': 0.6,
            'use_ensemble': True
        },
        'insights': {
            'extract_themes': True,
            'emotion_detection': True,
            'intensity_analysis': True
        },
        'visualization': {
            'generate_charts': True,
            'create_wordcloud': True,
            'dashboard_alerts': True
        }
    }

    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

    @staticmethod
    def init_app(app):
        """Initialize Flask app with config"""
        # Create upload folder if it doesn't exist
        if not os.path.exists(Config.UPLOAD_FOLDER):
            os.makedirs(Config.UPLOAD_FOLDER)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
