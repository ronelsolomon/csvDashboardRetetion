from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_from_directory
from flask_session import Session
import os
from werkzeug.utils import secure_filename
import pandas as pd

# Add this function before the create_app() function
def format_number(value, decimals=0):
    """Format number with thousands separator and optional decimal places"""
    try:
        if pd.isna(value):
            return 'N/A'
        if isinstance(value, (int, float)):
            if decimals == 0:
                return f"{int(value):,}"
            return f"{value:,.{decimals}f}"
        return str(value)
    except (ValueError, TypeError):
        return str(value)
    
# Add this function with the other filters
def format_datetime(value, format='%Y-%m-%d %H:%M:%S'):
    """Format a datetime object to a string"""
    if value is None:
        return ''
    try:
        return value.strftime(format)
    except (AttributeError, ValueError):
        return str(value)
    
def format_percentage(value, decimals=1):
    """Format a decimal as a percentage string"""
    try:
        if pd.isna(value):
            return 'N/A'
        return f"{float(value) * 100:.{decimals}f}%"
    except (ValueError, TypeError):
        return str(value)
    
def format_boolean(value, true_text='Yes', false_text='No'):
    """Format a boolean value as Yes/No or custom text"""
    return true_text if value else false_text

def format_score_color(score, good=80, warning=50):
    """Return a Bootstrap color class based on score thresholds"""
    if score >= good:
        return 'success'
    elif score >= warning:
        return 'warning'
    else:
        return 'danger'

def truncate_text(text, max_length=50, ellipsis='...'):
    """Truncate text to a maximum length"""
    if not text:
        return ''
    text = str(text)
    return (text[:max_length] + ellipsis) if len(text) > max_length else text

# Create Flask app
def create_app():
    app = Flask(__name__)
    
    app.jinja_env.filters['number_format'] = format_number
    app.jinja_env.filters['datetimeformat'] = format_datetime
    app.jinja_env.filters['percentage'] = format_percentage
    app.jinja_env.filters['boolean'] = format_boolean
    app.jinja_env.filters['score_color'] = format_score_color
    app.jinja_env.filters['truncate'] = truncate_text
    
    # Configuration
    app.config['SECRET_KEY'] = os.urandom(24)
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize extensions
    Session(app)
    
    # Register blueprints
    from eda_routes import eda_bp
    app.register_blueprint(eda_bp)
    
    # Main route
    @app.route('/')
    def index():
        return redirect(url_for('eda.upload_eda'))
    
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('404.html'), 404
        
    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template('500.html'), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)