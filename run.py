#!/usr/bin/env python3
"""
AuralyTica Tel AI - Customer Sentiment Analysis using Agentic AI
Main application runner

Usage:
    python run.py [--port PORT] [--host HOST] [--debug]
"""

import os
import sys
import argparse
from datetime import datetime

def setup_environment():
    """Setup environment and paths"""
    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    # Set environment variables
    os.environ.setdefault('FLASK_ENV', 'development')
    os.environ.setdefault('PYTHONPATH', current_dir)

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'flask', 'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'plotly', 'scikit-learn', 'textblob', 'vaderSentiment',
        'requests', 'python-dotenv', 'transformers', 'nltk'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install missing packages using:")
        print("   pip install -r requirements.txt")
        return False

    print("✅ All dependencies are installed")
    return True

def download_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk

        # List of required NLTK data
        nltk_data = [
            'punkt', 'stopwords', 'averaged_perceptron_tagger',
            'maxent_ne_chunker', 'words', 'vader_lexicon'
        ]

        print("📦 Downloading NLTK data...")
        for data in nltk_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                try:
                    nltk.data.find(f'corpora/{data}')
                except LookupError:
                    try:
                        nltk.data.find(f'taggers/{data}')
                    except LookupError:
                        try:
                            nltk.data.find(f'chunkers/{data}')
                        except LookupError:
                            try:
                                nltk.data.find(f'sentiment/{data}')
                            except LookupError:
                                print(f"   Downloading {data}...")
                                nltk.download(data, quiet=True)

        print("✅ NLTK data ready")

    except Exception as e:
        print(f"⚠️  Warning: Could not download NLTK data: {e}")

def create_directories():
    """Create necessary directories"""
    directories = [
        'uploads',
        'logs', 
        'data',
        'exports',
        'static/css',
        'static/js',
        'templates',
        'agents',
        'utils'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("✅ Directory structure ready")

def initialize_logging():
    """Initialize logging configuration"""
    import logging
    from datetime import datetime

    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Setup logging
    log_filename = f"logs/auralytica_{datetime.now().strftime('%Y%m%d')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)

def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                     AuralyTica Tel AI                        ║
    ║              Customer Sentiment Analysis                     ║
    ║              Using Agentic AI Architecture                   ║
    ║                                                              ║
    ║  🤖 Multi-Agent System                                       ║
    ║  📊 Real-time Analytics                                      ║
    ║  🧠 AI-Powered Insights                                      ║
    ║  📈 Interactive Dashboard                                    ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description='AuralyTica Tel AI - Customer Sentiment Analysis using Agentic AI'
    )
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--no-browser', action='store_true', help='Don\'t open browser automatically')

    args = parser.parse_args()

    # Print banner
    print_banner()

    # Setup environment
    setup_environment()

    # Initialize logging
    logger = initialize_logging()
    logger.info("Starting AuralyTica Tel AI application")

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Download NLTK data
    download_nltk_data()

    # Create directories
    create_directories()

    # Import and run Flask app
    try:
        from app import app

        print(f"\n🚀 Starting AuralyTica Tel AI server...")
        print(f"   Host: {args.host}")
        print(f"   Port: {args.port}")
        print(f"   Debug: {args.debug}")
        print(f"   URL: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")

        if not args.no_browser:
            # Open browser automatically
            import webbrowser
            import threading

            def open_browser():
                import time
                time.sleep(1.5)  # Give the server time to start
                webbrowser.open(f'http://localhost:{args.port}')

            threading.Thread(target=open_browser).start()

        print(f"\n📱 Access the application:")
        print(f"   Main Page: http://localhost:{args.port}")
        print(f"   Dashboard: http://localhost:{args.port}/dashboard")
        print(f"   API Health: http://localhost:{args.port}/health")

        print(f"\n💡 Tips:")
        print(f"   - Use Ctrl+C to stop the server")
        print(f"   - Check logs in the 'logs' directory")
        print(f"   - Upload CSV files for batch analysis")
        print(f"   - API documentation available at /health")

        # Run the Flask application
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            use_reloader=args.debug
        )

    except ImportError as e:
        logger.error(f"Failed to import Flask app: {e}")
        print(f"❌ Error: Could not start application: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
