from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import io
import base64

# Import our agent system
from agents.orchestrator import AgentOrchestrator
from utils.data_loader import DataLoader
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Initialize the agent orchestrator
orchestrator = AgentOrchestrator()
data_loader = DataLoader()

@app.route('/')
def index():
    """Main landing page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_feedback():
    """Main API endpoint for sentiment analysis"""
    try:
        data = request.get_json()

        if 'feedback' not in data:
            return jsonify({'error': 'No feedback provided'}), 400

        feedback = data['feedback']
        source_type = data.get('source_type', 'text')

        # Process through agent orchestrator
        results = orchestrator.process_feedback(feedback, source_type)

        return jsonify({
            'success': True,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/batch_analyze', methods=['POST'])
def batch_analyze():
    """Batch analysis endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Read CSV file
        df = pd.read_csv(file)

        if 'feedback' not in df.columns:
            return jsonify({'error': 'CSV must contain a "feedback" column'}), 400

        # Process batch through orchestrator
        results = orchestrator.process_batch(df)

        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len(df),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/dashboard_data')
def get_dashboard_data():
    """Get dashboard data"""
    try:
        # Load sample data for demonstration
        sample_data = data_loader.get_sample_dashboard_data()
        return jsonify(sample_data)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/export_results', methods=['POST'])
def export_results():
    """Export analysis results"""
    try:
        data = request.get_json()
        results = data.get('results', [])

        # Create CSV
        df = pd.DataFrame(results)

        # Create Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sentiment_Analysis')

        output.seek(0)

        return send_file(
            output,
            as_attachment=True,
            download_name=f'sentiment_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'agents_status': orchestrator.get_agent_status()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
