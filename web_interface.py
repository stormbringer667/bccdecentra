# -*- coding: utf-8 -*-
"""
Web Interface for Push Notification Generation
Provides a simple web UI to interact with the model interface
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import json
import os
from datetime import datetime
from model_interface import ModelInterface
from data_loader import load_clients, load_client_tables

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Initialize model interface
interface = ModelInterface()

# Load client data
clients_data = None
tables_data = None

def load_data():
    """Load client and transaction data"""
    global clients_data, tables_data
    try:
        clients_data = load_clients("data/clients.csv")
        tables_data = load_client_tables()
        print(f"Loaded {len(clients_data)} clients")
    except Exception as e:
        print(f"Error loading data: {e}")
        clients_data = pd.DataFrame()
        tables_data = {}

# Load data on startup
load_data()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/clients')
def get_clients():
    """Get list of all clients"""
    if clients_data is None or clients_data.empty:
        return jsonify({'error': 'No client data available'})
    
    clients_list = []
    for _, row in clients_data.iterrows():
        clients_list.append({
            'client_code': int(row['client_code']),
            'name': row['name'],
            'status': row['status'],
            'age': int(row['age']) if pd.notna(row['age']) else 0,
            'city': row['city'],
            'avg_balance': float(row['avg_monthly_balance_KZT'])
        })
    
    return jsonify({'clients': clients_list})

@app.route('/client/<int:client_id>')
def get_client_details(client_id):
    """Get detailed client information"""
    if clients_data is None or clients_data.empty:
        return jsonify({'error': 'No client data available'})
    
    # Find client
    client_row = clients_data[clients_data['client_code'] == client_id]
    if client_row.empty:
        return jsonify({'error': 'Client not found'})
    
    client_row = client_row.iloc[0]
    
    # Get transaction and transfer data
    tx = tables_data.get(client_id, {}).get("tx", pd.DataFrame())
    tr = tables_data.get(client_id, {}).get("tr", pd.DataFrame())
    
    # Prepare client info
    client_info = {
        'client_code': int(client_row['client_code']),
        'name': client_row['name'],
        'status': client_row['status'],
        'age': int(client_row['age']) if pd.notna(client_row['age']) else 0,
        'city': client_row['city'],
        'avg_balance': float(client_row['avg_monthly_balance_KZT'])
    }
    
    # Prepare transaction summary
    tx_summary = {}
    if not tx.empty:
        tx_summary = {
            'total_transactions': len(tx),
            'total_amount': float(tx['amount'].sum()),
            'categories': tx['category'].value_counts().to_dict(),
            'monthly_spending': tx.groupby(tx['date'].str[:7])['amount'].sum().to_dict() if 'date' in tx.columns else {}
        }
    
    # Prepare transfer summary
    tr_summary = {}
    if not tr.empty:
        tr_summary = {
            'total_transfers': len(tr),
            'transfer_types': tr['type'].value_counts().to_dict(),
            'directions': tr['direction'].value_counts().to_dict() if 'direction' in tr.columns else {}
        }
    
    return jsonify({
        'client': client_info,
        'transactions': tx_summary,
        'transfers': tr_summary
    })

@app.route('/recommend', methods=['POST'])
def generate_recommendation():
    """Generate product recommendation and push notification"""
    data = request.get_json()
    client_id = data.get('client_id')
    ollama_model = data.get('ollama_model', None)
    method = data.get('method', 'hybrid')
    
    if not client_id:
        return jsonify({'error': 'Client ID is required'})
    
    # Find client
    client_row = clients_data[clients_data['client_code'] == client_id]
    if client_row.empty:
        return jsonify({'error': 'Client not found'})
    
    client_row = client_row.iloc[0]
    
    # Get transaction and transfer data
    tx = tables_data.get(client_id, {}).get("tx", pd.DataFrame())
    tr = tables_data.get(client_id, {}).get("tr", pd.DataFrame())
    
    try:
        # Generate recommendation
        result = interface.generate_push_notification(
            client_row, tx, tr, ollama_model, method
        )
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/batch-process', methods=['POST'])
def batch_process():
    """Process all clients and generate CSV output"""
    data = request.get_json()
    ollama_model = data.get('ollama_model', None)
    method = data.get('method', 'hybrid')
    
    try:
        # Process all clients
        results_df = interface.process_all_clients(
            clients_csv="data/clients.csv",
            ollama_model=ollama_model,
            prediction_method=method,
            output_path="out/batch_results.csv"
        )
        
        return jsonify({
            'success': True,
            'message': f'Processed {len(results_df)} clients',
            'download_url': '/download/batch_results.csv'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/download/<filename>')
def download_file(filename):
    """Download generated files"""
    file_path = os.path.join('out', filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({'error': 'File not found'}), 404

@app.route('/model-info')
def model_info():
    """Get model information"""
    return jsonify({
        'ml_model_available': interface.ml_model_available,
        'product_classes': interface.product_classes,
        'model_meta': interface.model_meta,
        'config': interface.config
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('out', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
