import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import datetime as dt
import json
import csv
from io import StringIO
from collections import Counter
from flask import (
    Blueprint, render_template, request, jsonify, 
    flash, redirect, url_for, session, Response, current_app, make_response
)
from functools import wraps
import time
from werkzeug.utils import secure_filename
from ollama_report_generator import analyze_student_responses
import traceback

# Create Blueprint for EDA routes
eda_bp = Blueprint('eda', __name__, url_prefix='/eda')

class AssessmentEDA:
    """Comprehensive EDA for assessment data"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_df = df.copy()
        self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_')
        
    def get_data_overview(self):
        """Get basic data overview"""
        return {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'columns': list(self.df.columns),
            'memory_usage': f"{self.df.memory_usage(deep=True).sum() / 1024:.2f} KB",
            'date_range': self._get_date_range()
        }
    
    def _get_date_range(self):
        """Get date range if timestamp column exists"""
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
            return {
                'start': str(self.df['timestamp'].min()),
                'end': str(self.df['timestamp'].max()),
                'days': (self.df['timestamp'].max() - self.df['timestamp'].min()).days
            }
        return None
    
    def get_missing_data_analysis(self):
        """Analyze missing data"""
        missing_data = []
        for col in self.df.columns:
            total = len(self.df)
            missing = self.df[col].isna().sum()
            empty_strings = (self.df[col] == '').sum() if self.df[col].dtype == 'object' else 0
            
            missing_data.append({
                'column': col,
                'missing_count': int(missing),
                'empty_count': int(empty_strings),
                'total_missing': int(missing + empty_strings),
                'missing_pct': round((missing + empty_strings) / total * 100, 2),
                'data_type': str(self.df[col].dtype),
                'unique_values': int(self.df[col].nunique()),
                'sample_values': list(self.df[col].dropna().unique()[:5])
            })
        
        return sorted(missing_data, key=lambda x: x['missing_pct'], reverse=True)
    
    def get_column_statistics(self):
        """Get detailed column statistics"""
        stats = []
        
        for col in self.df.columns:
            col_stats = {
                'column': col,
                'dtype': str(self.df[col].dtype),
                'unique_count': int(self.df[col].nunique()),
                'null_count': int(self.df[col].isna().sum()),
                'null_pct': round(self.df[col].isna().sum() / len(self.df) * 100, 2)
            }
            
            # Numeric columns
            if pd.api.types.is_numeric_dtype(self.df[col]):
                col_stats.update({
                    'min': float(self.df[col].min()),
                    'max': float(self.df[col].max()),
                    'mean': float(self.df[col].mean()),
                    'median': float(self.df[col].median()),
                    'std': float(self.df[col].std())
                })
            
            # Categorical columns
            else:
                top_values = self.df[col].value_counts().head(10)
                col_stats['top_values'] = {
                    str(k): int(v) for k, v in top_values.items()
                }
            
            stats.append(col_stats)
        
        return stats
    
    def get_student_analysis(self):
        """Analyze student performance"""
        current_app.logger.info('Starting student analysis...')
        if 'student_id' not in self.df.columns:
            current_app.logger.warning('student_id column not found in dataframe')
            return []  # Return empty list instead of None for consistent return type
        
        student_stats = []
        for student_id in self.df['student_id'].unique():
            student_data = self.df[self.df['student_id'] == student_id]
            
            total = len(student_data)
            if 'is_correct' in self.df.columns:
                correct = student_data['is_correct'].sum()
                accuracy = (correct / total * 100) if total > 0 else 0
            else:
                correct = 0
                accuracy = 0
            
            student_stats.append({
                'student_id': str(student_id),
                'total_questions': int(total),
                'correct_answers': int(correct),
                'accuracy': round(accuracy, 2),
                'topics_covered': list(student_data['topic'].unique()) if 'topic' in self.df.columns else []
            })
        
        # Ensure we return a list, not a single dictionary
        if not isinstance(student_stats, list):
            student_stats = [student_stats] if student_stats else []
            
        return sorted(student_stats, key=lambda x: x.get('accuracy', 0), reverse=True)
    
    def get_topic_analysis(self):
        """Analyze performance by topic"""
        current_app.logger.info('Starting topic analysis...')
        if 'topic' not in self.df.columns:
            current_app.logger.warning('topic column not found in dataframe')
            return None
        
        topic_stats = []
        for topic in self.df['topic'].unique():
            topic_data = self.df[self.df['topic'] == topic]
            
            total = len(topic_data)
            if 'is_correct' in self.df.columns:
                correct = topic_data['is_correct'].sum()
                accuracy = (correct / total * 100) if total > 0 else 0
            else:
                correct = 0
                accuracy = 0
            
            topic_stats.append({
                'topic': str(topic),
                'total_questions': int(total),
                'correct_answers': int(correct),
                'incorrect_answers': int(total - correct),
                'accuracy': round(accuracy, 2),
                'students': list(topic_data['student_id'].unique()) if 'student_id' in self.df.columns else []
            })
        
        return sorted(topic_stats, key=lambda x: x['accuracy'])
    
    def get_category_analysis(self):
        """Analyze performance by question category"""
        if 'category' not in self.df.columns:
            return None
        
        category_stats = []
        for category in self.df['category'].unique():
            cat_data = self.df[self.df['category'] == category]
            
            total = len(cat_data)
            if 'is_correct' in self.df.columns:
                correct = cat_data['is_correct'].sum()
                accuracy = (correct / total * 100) if total > 0 else 0
            else:
                correct = 0
                accuracy = 0
            
            category_stats.append({
                'category': str(category),
                'total_questions': int(total),
                'correct_answers': int(correct),
                'accuracy': round(accuracy, 2)
            })
        
        return sorted(category_stats, key=lambda x: x['accuracy'])
    
    def get_learning_objective_analysis(self):
        """Analyze performance by learning objective"""
        if 'learning_objective' not in self.df.columns:
            return None
        
        lo_stats = []
        for lo in self.df['learning_objective'].unique():
            lo_data = self.df[self.df['learning_objective'] == lo]
            
            total = len(lo_data)
            if 'is_correct' in self.df.columns:
                correct = lo_data['is_correct'].sum()
                accuracy = (correct / total * 100) if total > 0 else 0
            else:
                correct = 0
                accuracy = 0
            
            lo_stats.append({
                'learning_objective': str(lo),
                'total_questions': int(total),
                'correct_answers': int(correct),
                'accuracy': round(accuracy, 2),
                'topics': list(lo_data['topic'].unique()) if 'topic' in self.df.columns else []
            })
        
        return sorted(lo_stats, key=lambda x: x['accuracy'])
    
    def get_error_analysis(self):
        """Analyze common errors"""
        if 'why_wrong' not in self.df.columns:
            return None
        
        errors = self.df[self.df['why_wrong'].notna() & (self.df['why_wrong'] != '')]
        
        if len(errors) == 0:
            return None
        
        error_counts = errors['why_wrong'].value_counts()
        
        return [
            {
                'error_type': str(error),
                'count': int(count),
                'percentage': round(count / len(errors) * 100, 2)
            }
            for error, count in error_counts.items()
        ]
    
    def get_temporal_analysis(self):
        """Analyze performance over time"""
        if 'timestamp' not in self.df.columns:
            return None
        
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
        self.df = self.df.dropna(subset=['timestamp'])
        
        if len(self.df) == 0:
            return None
        
        self.df['date'] = self.df['timestamp'].dt.date
        
        temporal_stats = []
        for date in sorted(self.df['date'].unique()):
            date_data = self.df[self.df['date'] == date]
            
            total = len(date_data)
            if 'is_correct' in self.df.columns:
                correct = date_data['is_correct'].sum()
                accuracy = (correct / total * 100) if total > 0 else 0
            else:
                correct = 0
                accuracy = 0
            
            temporal_stats.append({
                'date': str(date),
                'total_questions': int(total),
                'correct_answers': int(correct),
                'accuracy': round(accuracy, 2)
            })
        
        return temporal_stats
    
    def get_data_quality_report(self):
        """Generate comprehensive data quality report"""
        quality_issues = []
        
        # Check for duplicate rows
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            quality_issues.append({
                'type': 'Duplicate Rows',
                'severity': 'High',
                'count': int(duplicates),
                'description': f'Found {duplicates} duplicate rows in the dataset'
            })
        
        # Check for missing critical columns
        critical_cols = ['student_id', 'question', 'answer_key', 'student_answer']
        for col in critical_cols:
            if col in self.df.columns:
                missing = self.df[col].isna().sum() + (self.df[col] == '').sum()
                if missing > 0:
                    quality_issues.append({
                        'type': f'Missing {col}',
                        'severity': 'Critical',
                        'count': int(missing),
                        'description': f'{missing} rows missing {col} values'
                    })
        
        # Check for inconsistent data
        if 'is_correct' in self.df.columns:
            invalid_correct = ~self.df['is_correct'].isin([True, False, 'True', 'False', 1, 0])
            if invalid_correct.sum() > 0:
                quality_issues.append({
                    'type': 'Invalid is_correct values',
                    'severity': 'High',
                    'count': int(invalid_correct.sum()),
                    'description': 'Some is_correct values are not boolean'
                })
        
        return quality_issues
    
    def get_recommendations(self):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Student performance recommendations
        try:
            student_analysis = self.get_student_analysis()
            if student_analysis and isinstance(student_analysis, list) and len(student_analysis) > 0:
                low_performers = []
                for s in student_analysis:
                    if not isinstance(s, dict):
                        continue
                    if 'accuracy' in s and isinstance(s['accuracy'], (int, float)):
                        if s['accuracy'] < 50:
                            low_performers.append(s)
                    elif 'correct_answers' in s and 'total_questions' in s and s['total_questions'] > 0:
                        accuracy = (s['correct_answers'] / s['total_questions']) * 100
                        if accuracy < 50:
                            low_performers.append(s)
                
                if low_performers:
                    recommendations.append({
                        'category': 'Student Performance',
                        'priority': 'High',
                        'recommendation': f'{len(low_performers)} students scoring below 50%. Consider targeted interventions.',
                        'action_items': [
                            'Schedule one-on-one tutoring sessions',
                            'Review fundamental concepts',
                            'Provide additional practice materials'
                        ]
                    })
        except Exception as e:
            current_app.logger.error(f'Error generating student performance recommendations: {str(e)}')
            import traceback
            current_app.logger.error(traceback.format_exc())
        
        # Topic recommendations
        try:
            topic_analysis = self.get_topic_analysis()
            if topic_analysis and isinstance(topic_analysis, list) and len(topic_analysis) > 0:
                weak_topics = []
                for t in topic_analysis:
                    if not isinstance(t, dict):
                        continue
                    if 'accuracy' in t and isinstance(t['accuracy'], (int, float)):
                        if t['accuracy'] < 60:
                            weak_topics.append(t)
                    elif 'correct_answers' in t and 'total_questions' in t and t['total_questions'] > 0:
                        accuracy = (t['correct_answers'] / t['total_questions']) * 100
                        if accuracy < 60:
                            weak_topics.append(t)
                
                if weak_topics:
                    recommendations.append({
                        'category': 'Topic Mastery',
                        'priority': 'Medium',
                        'recommendation': f'{len(weak_topics)} topics with accuracy below 60%',
                        'action_items': [
                            f"Focus on: {', '.join([str(t.get('topic', 'Unknown')) for t in weak_topics[:3]])}",
                            'Create supplementary materials',
                            'Increase practice questions'
                        ]
                    })
        except Exception as e:
            current_app.logger.error(f'Error generating topic recommendations: {str(e)}')
            import traceback
            current_app.logger.error(traceback.format_exc())
        
        # Data quality recommendations
        try:
            quality_issues = self.get_data_quality_report()
            if quality_issues and isinstance(quality_issues, list):
                recommendations.append({
                    'category': 'Data Quality',
                    'priority': 'High',
                    'recommendation': f'Found {len(quality_issues)} data quality issues',
                    'action_items': [
                        'Clean missing data',
                        'Validate data entry processes',
                        'Remove duplicate records'
                    ]
                })
        except Exception as e:
            current_app.logger.error(f'Error generating data quality recommendations: {str(e)}')
        
        return recommendations
    
    def generate_full_report(self):
        """Generate complete EDA report"""
        current_app.logger.info('Starting to generate full EDA report...')
        
        # Initialize report dictionary
        report = {}
        
        try:
            current_app.logger.info('Getting data overview...')
            report['overview'] = self.get_data_overview()
            current_app.logger.info('Getting missing data analysis...')
            report['missing_data'] = self.get_missing_data_analysis()
            current_app.logger.info('Getting column statistics...')
            report['column_stats'] = self.get_column_statistics()
            
            # Student analysis with detailed logging
            try:
                current_app.logger.info('Starting student analysis...')
                student_analysis = self.get_student_analysis()
                current_app.logger.info(f'Student analysis result type: {type(student_analysis)}')
                if isinstance(student_analysis, list):
                    current_app.logger.info(f'Student analysis list length: {len(student_analysis)}')
                    if student_analysis and isinstance(student_analysis[0], dict):
                        current_app.logger.info(f'Student analysis keys: {student_analysis[0].keys()}')
                report['student_analysis'] = student_analysis
            except Exception as e:
                current_app.logger.error(f'Error in student analysis: {str(e)}', exc_info=True)
                report['student_analysis'] = None
            
            # Topic analysis with detailed logging
            try:
                current_app.logger.info('Starting topic analysis...')
                topic_analysis = self.get_topic_analysis()
                current_app.logger.info(f'Topic analysis result type: {type(topic_analysis)}')
                if isinstance(topic_analysis, list):
                    current_app.logger.info(f'Topic analysis list length: {len(topic_analysis)}')
                    if topic_analysis and isinstance(topic_analysis[0], dict):
                        current_app.logger.info(f'Topic analysis keys: {topic_analysis[0].keys()}')
                report['topic_analysis'] = topic_analysis
            except Exception as e:
                current_app.logger.error(f'Error in topic analysis: {str(e)}', exc_info=True)
                report['topic_analysis'] = None
            
            current_app.logger.info('Getting category analysis...')
            report['category_analysis'] = self.get_category_analysis()
            
            current_app.logger.info('Getting learning objective analysis...')
            report['learning_objective_analysis'] = self.get_learning_objective_analysis()
            
            current_app.logger.info('Getting error analysis...')
            report['error_analysis'] = self.get_error_analysis()
            
            current_app.logger.info('Getting temporal analysis...')
            report['temporal_analysis'] = self.get_temporal_analysis()
            
            current_app.logger.info('Getting data quality report...')
            report['quality_issues'] = self.get_data_quality_report()
            
            # Get recommendations last with detailed logging
            try:
                current_app.logger.info('Generating recommendations...')
                recommendations = self.get_recommendations()
                current_app.logger.info(f'Generated {len(recommendations) if recommendations else 0} recommendations')
                report['recommendations'] = recommendations
            except Exception as e:
                current_app.logger.error(f'Error generating recommendations: {str(e)}', exc_info=True)
                report['recommendations'] = []
            
            current_app.logger.info('Successfully generated full EDA report')
            return report
            
        except Exception as e:
            current_app.logger.error(f'Unexpected error generating full report: {str(e)}', exc_info=True)
            print(f"[ERROR] Failed to generate full report: {str(e)}")
            print("[DEBUG] Traceback:", exc_info=True)
            raise


# Helper function to check allowed file
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@eda_bp.route('/clear-session')
def clear_session():
    session.clear()
    return "Session cleared"

@eda_bp.route('/analyze_student_answers', methods=['POST'])
def analyze_student_answers():
    """Analyze student answers using Ollama and generate a report."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Analyze the student responses
            report_path = analyze_student_responses(filepath)
            
            # Return the path to the generated report
            return jsonify({
                'status': 'success',
                'report_path': report_path,
                'download_url': url_for('static', filename=os.path.basename(report_path))
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@eda_bp.route('/analysis_status')
def analysis_status():
    """Check if the analysis is complete.
    
    Returns:
        str: 'complete' if analysis is done, 'waiting' if still processing,
             'not_found' if no analysis is in progress
    """
    try:
        # Check if we have a valid session and analysis data
        if 'eda_report' in session and 'student_analysis' in session['eda_report']:
            return 'complete'
            
        # Check if analysis is in progress (but not yet complete)
        if 'analysis_started' in session:
            # Check if analysis has been running for too long
            time_since_start = datetime.utcnow() - session['analysis_started']
            if time_since_start > timedelta(minutes=5):  # 5 minutes timeout
                # Clean up if it's taking too long
                session.pop('analysis_started', None)
                return 'not_found'
            return 'waiting'
            
        # No analysis in progress
        return 'not_found'
        
    except Exception as e:
        current_app.logger.error(f"Error checking analysis status: {str(e)}")
        return 'not_found'

@eda_bp.route('/check_processing')
def check_processing():
    """Check if processing is in progress and return status"""
    return jsonify({
        'processing': session.get('processing_file', False),
        'has_report': 'eda_report' in session
    })

@eda_bp.route('/reset_processing', methods=['POST'])
def reset_processing():
    """Reset the processing state"""
    session.pop('processing_file', None)
    session.pop('analysis_started', None)
    session.pop('eda_report', None)
    session.modified = True
    return jsonify({'status': 'success'})

# Flask Routes
@eda_bp.route('/upload', methods=['GET', 'POST'])
def upload_eda():
    """Upload CSV for EDA"""
    print(f"\n[DEBUG] === Entered upload_eda function ===")
    print(f"[DEBUG] Request method: {request.method}")
    print(f"[DEBUG] Session ID: {session.sid}")
    print(f"[DEBUG] Current session data: {dict(session)}")  # Add this line to debug session state
    
    # Clear any stale processing state
    if session.get('processing_file'):
        # Check if the processing is actually still ongoing
        if 'analysis_started' in session:
            try:
                
                time_since_start = dt.datetime.utcnow() - session['analysis_started']
                if time_since_start > timedelta(minutes=5):  # 5 minutes timeout
                    print("[DEBUG] Found stale processing state, clearing...")
                    session.pop('processing_file', None)
                    session.pop('analysis_started', None)
                    session.pop('progress_file', None)
                    session.modified = True
            except Exception as e:
                print(f"[DEBUG] Error checking processing time: {str(e)}")
                session.pop('processing_file', None)
                session.pop('analysis_started', None)
                session.pop('progress_file', None)
                session.modified = True

    if request.method == 'GET':
        print("[DEBUG] Handling GET request - Clearing session and showing upload form")
        session.pop('processing_file', None)
        session.pop('analysis_started', None)
        session.pop('progress_file', None)
        session.modified = True
        return render_template('eda_upload.html')
        
    # Check if we're already processing a file
    if session.get('processing_file'):
        print("[DEBUG] File processing already in progress in this session")
        # If we have a report, show it
        if 'eda_report' in session:
            print("[DEBUG] Found existing report, displaying it")
            return render_template('eda_report.html', report=session['eda_report'])
        else:
            flash('A file is being processed. Please wait...', 'info')
            return redirect(url_for('eda.upload_eda'))
    
    # Handle POST request
    print("\n[DEBUG] === Processing POST request ===")
    print(f"[DEBUG] Form data: {request.form}")
    print(f"[DEBUG] Files in request: {request.files}")
    
    # Check if we're already processing a file (duplicate check for safety)
    if session.get('processing_file'):
        print("[DEBUG] File processing already in progress (duplicate check)")
        flash('A file is already being processed. Please wait...', 'info')
        # If we have a report in session, use it, otherwise create an empty one
        report = session.get('eda_report', {'overview': {'total_rows': 0, 'total_columns': 0, 'columns': []}})
        return render_template('eda_report.html', report=report)
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        print("[DEBUG] ERROR: No 'file' in request.files")
        flash('No file part in the request', 'error')
        # Return to upload page with empty report to prevent template errors
        return render_template('eda_upload.html', report={'overview': {'total_rows': 0, 'total_columns': 0, 'columns': []}})
        
    file = request.files['file']
    print(f"\n[DEBUG] File object: {file}")
    print(f"[DEBUG] Filename: {file.filename}")
    print(f"[DEBUG] Content type: {file.content_type}")
    print(f"[DEBUG] Content length: {request.content_length} bytes")
        
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        print("[DEBUG] ERROR: No file selected")
        flash('No file selected', 'error')
        return render_template('eda_upload.html')
            
    if not allowed_file(file.filename):
        print(f"[DEBUG] ERROR: Invalid file type: {file.filename}")
        flash('Please upload a valid CSV or Excel file', 'error')
        return render_template('eda_upload.html')
    
    try:
        print(f"\n[DEBUG] === Starting file processing ===")
        print(f"[DEBUG] Processing file: {file.filename}")
        current_app.logger.info(f'Starting to process file: {file.filename}')
        print(f"[DEBUG] Starting to process file: {file.filename}")
        current_app.logger.info(f'Starting to process file: {file.filename}')
                
        # Read the file based on extension
        print("[DEBUG] Reading file...")
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            else:  # Excel file
                df = pd.read_excel(file)
            print(f"[DEBUG] Successfully read file. Shape: {df.shape}")
            print(f"[DEBUG] Columns: {df.columns.tolist()}")
            print("[DEBUG] First few rows:")
            print(df.head().to_string())
                    
            current_app.logger.info(f'Successfully read file. Shape: {df.shape}, Columns: {df.columns.tolist()}')
            current_app.logger.info(f'First few rows:\n{df.head().to_string()}')
                    
        except Exception as e:
            error_msg = f'Error reading file: {str(e)}'
            print(f"[ERROR] {error_msg}")
            current_app.logger.error(error_msg, exc_info=True)
            flash(error_msg, 'error')
            # Pass an empty report to the template to prevent errors
            return render_template('eda_upload.html', report={'overview': {'total_rows': 0, 'total_columns': 0, 'columns': []}})
            
        # Store the original filename in the session
        session['original_filename'] = secure_filename(file.filename)
            
        try:
            # Initialize EDA and generate report
            print("[DEBUG] Initializing EDA...")
            current_app.logger.info('Initializing EDA...')
            eda = AssessmentEDA(df)
            print("[DEBUG] Generating full report...")
            current_app.logger.info('Generating full report...')
            report = eda.generate_full_report()
            print("[DEBUG] Successfully generated report")
                
            # Store the report in session and clear processing state
            session['processing_file'] = False
            session['analysis_complete'] = True
            session['eda_report'] = report
            session.modified = True
            
            current_app.logger.info('Rendering EDA report...')
            response = make_response(render_template('eda_report.html', report=report))
            
            # Check if we should run student answer analysis in background
            if 'Student_Answer' in df.columns and 'Answer_Key' in df.columns:
                from threading import Thread
                import time
                
                def run_analysis_async(temp_file_path, student_df, app):
                    with app.app_context():
                        try:
                            app.logger.info('Starting background student answer analysis...')
                            
                            # Create a progress file to track analysis progress
                            progress_file = os.path.join(app.config['UPLOAD_FOLDER'], f'analysis_progress_{int(time.time())}.json')
                            
                            def update_progress(progress_data):
                                try:
                                    with open(progress_file, 'w') as f:
                                        json.dump({
                                            'status': 'processing',
                                            'progress': progress_data.get('processed', 0),
                                            'total': progress_data.get('total', 0),
                                            'message': progress_data.get('message', ''),
                                            'eta': progress_data.get('eta', 0),
                                            'timestamp': datetime.utcnow().isoformat()
                                        }, f)
                                except Exception as e:
                                    app.logger.error(f'Error updating progress file: {str(e)}')
                            
                            # Start the analysis with progress tracking
                            analysis_result = analyze_student_responses(
                                temp_file_path,
                                output_dir=os.path.join(app.root_path, 'reports'),
                                batch_size=3,  # Smaller batch size for better progress updates
                                max_workers=2   # Limit concurrency to avoid overwhelming the system
                            )
                            
                            # Store the analysis in the session for the next request
                            with app.test_request_context():
                                if 'eda_report' not in session:
                                    session['eda_report'] = {}
                                
                                if analysis_result['success']:
                                    session['eda_report']['student_analysis'] = {
                                        'status': 'completed',
                                        'report_path': analysis_result['report_path'],
                                        'csv_path': analysis_result['csv_path'],
                                        'num_responses': analysis_result['num_responses'],
                                        'processing_time': analysis_result['processing_time_seconds']
                                    }
                                    app.logger.info(f'Student analysis completed: {analysis_result["message"]}')
                                    
                                    # Update progress to completed
                                    try:
                                        with open(progress_file, 'w') as f:
                                            json.dump({
                                                'status': 'completed',
                                                'message': 'Analysis completed successfully',
                                                'report_path': analysis_result['report_path'],
                                                'csv_path': analysis_result['csv_path'],
                                                'num_responses': analysis_result['num_responses'],
                                                'processing_time': analysis_result['processing_time_seconds'],
                                                'timestamp': datetime.utcnow().isoformat()
                                            }, f)
                                    except Exception as e:
                                        app.logger.error(f'Error writing completion status: {str(e)}')
                                else:
                                    error_msg = f'Analysis failed: {analysis_result["message"]}'
                                    app.logger.error(error_msg)
                                    
                                    # Update progress with error
                                    try:
                                        with open(progress_file, 'w') as f:
                                            json.dump({
                                                'status': 'error',
                                                'message': error_msg,
                                                'timestamp': datetime.utcnow().isoformat()
                                            }, f)
                                    except Exception as e:
                                        app.logger.error(f'Error writing error status: {str(e)}')
                                
                                # Clean up the processing state
                                session.pop('processing_file', None)
                                session.pop('analysis_started', None)
                                session['progress_file'] = progress_file
                                session.modified = True
                                
                        except Exception as e:
                            app.logger.error(f'Unexpected error in background analysis: {str(e)}')
                            app.logger.error(traceback.format_exc())
                            
                            # Ensure we clean up the processing state even on error
                            with app.test_request_context():
                                session.pop('processing_file', None)
                                session.pop('analysis_started', None)
                                session.modified = True
                                
                                # Try to update progress file with error
                                try:
                                    progress_file = session.get('progress_file')
                                    if progress_file and os.path.exists(progress_file):
                                        with open(progress_file, 'w') as f:
                                            json.dump({
                                                'status': 'error',
                                                'message': f'Unexpected error: {str(e)}',
                                                'timestamp': datetime.utcnow().isoformat()
                                            }, f)
                                except Exception as e2:
                                    app.logger.error(f'Error updating progress file on error: {str(e2)}')
                        
                        finally:
                            # Clean up the temporary file
                            if os.path.exists(temp_file_path):
                                try:
                                    os.remove(temp_file_path)
                                except Exception as e:
                                    app.logger.error(f'Error removing temp file: {str(e)}')
                
                # Set processing state
                session['processing_file'] = True
                session['analysis_started'] = datetime.utcnow()
                session.modified = True
                
                # Create and start the background thread
                temp_file = os.path.join(current_app.config['UPLOAD_FOLDER'], f'temp_analysis_{int(time.time())}.csv')
                df.to_csv(temp_file, index=False)
                
                # Create a copy of the app for the background thread
                app = current_app._get_current_object()
                
                thread = Thread(target=run_analysis_async, args=(temp_file, df.copy(), app))
                thread.daemon = True
                thread.start()
                
                # Set a cookie to indicate analysis is in progress (for client-side)
                response.set_cookie('analysis_in_progress', 'true', max_age=300)  # 5 minutes
            
            current_app.logger.info('Successfully generated report')
            return response
        except Exception as e:
            print(f"[ERROR] Error during EDA processing: {str(e)}")
            print("[DEBUG] Traceback:", traceback.format_exc())
            current_app.logger.error(f'Error during EDA processing: {str(e)}', exc_info=True)
            flash(f'Error during analysis: {str(e)}', 'error')
            return render_template('eda_upload.html')
            
    except Exception as e:
        current_app.logger.error(f'Unexpected error processing file: {str(e)}', exc_info=True)
        flash(f'An unexpected error occurred: {str(e)}', 'error')
        return render_template('eda_upload.html')
        return redirect(request.url)
    return render_template('eda_upload.html')


@eda_bp.route('/api/analyze', methods=['POST'])
def analyze_csv_api():
    """API endpoint for CSV analysis"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400
    
    try:
        # Read the file based on extension
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:  # Excel file
            df = pd.read_excel(file)
            
        # Initialize EDA and generate report
        eda = AssessmentEDA(df)
        report = eda.generate_full_report()
        
        return jsonify({
            'success': True,
            'report': report
        })
    
    except Exception as e:
        current_app.logger.error(f'API Error: {str(e)}')
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@eda_bp.route('/export/<format>')
def export_report(format):
    """Export EDA report in different formats"""
    if 'eda_report' not in session:
        flash('No report available to export', 'error')
        return redirect(url_for('eda.upload_eda'))
    
    report = session['eda_report']
    filename = session.get('original_filename', 'report').rsplit('.', 1)[0]
    
    if format == 'csv':
        # Convert report to CSV format
        output = StringIO()
        writer = csv.writer(output)
        
        # Write overview
        writer.writerow(['Section', 'Metric', 'Value'])
        writer.writerow(['Overview', 'Total Rows', report['overview']['total_rows']])
        writer.writerow(['Overview', 'Total Columns', report['overview']['total_columns']])
        
        # Write missing data analysis
        writer.writerow([])
        writer.writerow(['Missing Data Analysis'])
        writer.writerow(['Column', 'Data Type', 'Missing Values', 'Missing %', 'Unique Values'])
        for item in report['missing_data']:
            writer.writerow([
                item['column'],
                item['data_type'],
                item['total_missing'],
                f"{item['missing_pct']}%",
                item['unique_values']
            ])
        
        # Write column statistics
        writer.writerow([])
        writer.writerow(['Column Statistics'])
        for stat in report['column_stats']:
            writer.writerow([f"Column: {stat['column']} ({stat['dtype']})"])
            if 'min' in stat:  # Numeric column
                writer.writerow(['Min', 'Max', 'Mean', 'Median', 'Std Dev'])
                writer.writerow([
                    stat.get('min', 'N/A'),
                    stat.get('max', 'N/A'),
                    stat.get('mean', 'N/A'),
                    stat.get('median', 'N/A'),
                    stat.get('std', 'N/A')
                ])
            else:  # Categorical column
                writer.writerow(['Value', 'Count'])
                for value, count in stat.get('top_values', {}).items():
                    writer.writerow([value, count])
            writer.writerow([])
        
        # Add student analysis if available
        if 'student_analysis' in report and report['student_analysis'] and report['student_analysis'].get('status') == 'completed':
            try:
                # Try to load the detailed analysis CSV
                csv_path = report['student_analysis'].get('csv_path')
                if csv_path and os.path.exists(csv_path):
                    # Read the detailed analysis CSV
                    detailed_df = pd.read_csv(csv_path)
                    
                    # Add detailed analysis section
                    writer.writerow([])
                    writer.writerow(['Detailed Student Analysis'])
                    
                    # Write headers for detailed analysis
                    writer.writerow([
                        'Student ID', 'Topic', 'Question', 'Student Answer', 'Correct Answer', 
                        'Is Correct', 'Key Concepts', 'Step by Step', 'Strategy', 'Rationale'
                    ])
                    
                    # Process each row in the detailed analysis
                    for _, row in detailed_df.iterrows():
                        # Parse the detailed_feedback if it exists
                        detailed_feedback = {}
                        if 'detailed_feedback' in row and pd.notna(row['detailed_feedback']):
                            try:
                                detailed_feedback = json.loads(row['detailed_feedback'].replace("'", '"'))
                            except:
                                detailed_feedback = {}
                        
                        # Extract key concepts, step by step, strategy, and rationale
                        key_concepts = ""
                        step_by_step = ""
                        strategy = ""
                        rationale = ""
                        
                        if 'facts' in detailed_feedback and 'key_concepts' in detailed_feedback['facts']:
                            key_concepts = ", ".join(detailed_feedback['facts']['key_concepts'])
                        
                        if 'step_by_step' in detailed_feedback:
                            step_by_step = " | ".join(detailed_feedback['step_by_step'])
                        
                        if 'strategy' in detailed_feedback:
                            strategy = str(detailed_feedback['strategy'])
                        
                        if 'rationale' in detailed_feedback:
                            rationale = str(detailed_feedback['rationale'])
                        
                        # Write the row
                        writer.writerow([
                            row.get('Student_ID', ''),
                            row.get('Topic', ''),
                            row.get('Question', ''),
                            row.get('Student_Answer', ''),
                            row.get('Answer_Key', ''),
                            row.get('Is_Correct', ''),
                            key_concepts,
                            step_by_step,
                            strategy,
                            rationale
                        ])
            except Exception as e:
                print(f"Error adding detailed analysis to CSV: {str(e)}")
                writer.writerow(['Error', 'Could not include detailed analysis', str(e)])
        
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename={filename}_analysis.csv',
                'Content-Type': 'text/csv; charset=utf-8-sig'
            }
        )
    
    elif format == 'json':
        return Response(
            json.dumps(report, indent=2, default=str),
            mimetype='application/json',
            headers={
                'Content-Disposition': f'attachment; filename={filename}_analysis.json'
            }
        )
    
    flash('Invalid export format', 'error')
    return redirect(url_for('eda.upload_eda'))