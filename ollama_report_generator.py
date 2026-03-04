import os
import json
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import ollama
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
from collections import defaultdict

class OllamaReportGenerator:
    """
    Generate comprehensive PDF reports using Ollama with WizardMath to analyze student answers.
    Supports multiple analysis types and generates detailed educational reports.
    """
    
    def __init__(self, df: pd.DataFrame, output_dir: str = 'reports'):
        """
        Initialize the report generator with student answer data.
        
        Args:
            df: DataFrame containing student answers with columns:
                - Student_ID: Unique identifier for each student
                - Topic: The topic or subject of the question
                - Question: The question text
                - Student_Answer: The student's response
                - Answer_Key: The correct answer
            output_dir: Directory to save generated reports (default: 'reports')
        """
        self.df = df
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.analysis_cache = {}
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _process_single_answer(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Process a single student answer."""
        if pd.isna(row['Student_Answer']):
            return None
            
        prompt = self._create_analysis_prompt(row)
        
        try:
            # Add timeout to prevent hanging
            response = ollama.generate(
                model='wizard-math:7b-v1.1-q5_K_M',
                prompt=prompt,
                format='json',
                options={
                    'temperature': 0.3,
                    'top_p': 0.9,
                    'max_tokens': 1000
                }
            )
            
            # Print the raw response for debugging
            print(f"\n[OLLAMA DEBUG] Question: {row.get('Question', 'N/A')}")
            print(f"[OLLAMA DEBUG] Student Answer: {row.get('Student_Answer', 'N/A')}")
            print(f"[OLLAMA DEBUG] Model Response: {response}")
        except Exception as e:
            print(f"Error generating response for {row.get('Student_ID', 'unknown')}, Question {row.get('Question', 'unknown')}: {str(e)}")
            return None
                    
        try:
            # Print the parsed response if it's valid JSON
            if 'response' in response:
                try:
                    parsed_response = json.loads(response['response'])
                    print(f"[OLLAMA DEBUG] Parsed Analysis: {json.dumps(parsed_response, indent=2)}")
                    
                    # Extract the detailed_feedback from the parsed response
                    detailed_feedback = parsed_response.get('detailed_feedback', {})
                    
                    # Ensure step_by_step is a list
                    if 'step_by_step' not in detailed_feedback:
                        detailed_feedback['step_by_step'] = []
                    
                    analysis = {
                        'analysis': 'Analysis complete',
                        'common_misconceptions': 'Not available',
                        'suggested_intervention': 'Not available',
                        'detailed_feedback': detailed_feedback  # Include the detailed feedback
                    }
                except json.JSONDecodeError:
                    print("[OLLAMA DEBUG] Could not parse response as JSON")
                    analysis = {
                        'analysis': 'Analysis not available',
                        'common_misconceptions': 'Not available',
                        'suggested_intervention': 'Not available',
                        'detailed_feedback': {}
                    }
            else:
                # If no response, create a default analysis
                analysis = {
                    'analysis': 'No analysis available',
                    'common_misconceptions': 'Not available',
                    'suggested_intervention': 'Not available',
                    'detailed_feedback': {}
                }
            
            result = {
                'Student_ID': row['Student_ID'],
                'Topic': row['Topic'],
                'Question': row['Question'],
                'Student_Answer': row['Student_Answer'],
                'Answer_Key': row['Answer_Key'],
                'Is_Correct': row['Answer_Key'].strip().lower() == str(row['Student_Answer']).strip().lower(),
                'Analysis': analysis.get('analysis', ''),
                'Common_Misconceptions': analysis.get('common_misconceptions', ''),
                'Suggested_Intervention': analysis.get('suggested_intervention', ''),
                'detailed_feedback': analysis.get('detailed_feedback', {}),  # Include detailed_feedback
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return result

        except Exception as e:
            print(f"Error analyzing response for {row.get('Student_ID', 'unknown')}, Question {row.get('Question', 'unknown')}: {str(e)}")
            return None
    
    def analyze_student_answers(self, batch_size: int = 5, max_workers: int = 3) -> pd.DataFrame:
        """
        Analyze student answers using Ollama's WizardMath model with parallel processing.
        
        Args:
            batch_size: Number of answers to process in each batch
            max_workers: Maximum number of parallel workers
            
        Returns:
            DataFrame with analysis results
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        results = []
        processed = 0
        total = len(self.df)
        start_time = time.time()
        
        # Process in batches to avoid overwhelming the system
        for i in range(0, len(self.df), batch_size):
            batch = self.df.iloc[i:i + batch_size]
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks in the batch
                future_to_row = {
                    executor.submit(self._process_single_answer, row): idx 
                    for idx, row in batch.iterrows()
                }
                
                # Process completed tasks
                for future in as_completed(future_to_row):
                    result = future.result()
                    if result is not None:
                        results.append(result)
                    
                    # Update progress
                    processed += 1
                    elapsed = time.time() - start_time
                    avg_time = elapsed / processed if processed > 0 else 0
                    remaining = avg_time * (total - processed)
                    
                    print(f"\rProcessed {processed}/{total} answers | "
                          f"Elapsed: {elapsed:.1f}s | "
                          f"Avg: {avg_time:.1f}s/answer | "
                          f"ETA: {remaining/60:.1f} min", end="")
            
            # Small delay between batches
            time.sleep(1)
        
        print("\nAnalysis complete!")
        return pd.DataFrame(results) if results else pd.DataFrame()
        
        return pd.DataFrame(results)
    
    def _create_analysis_prompt(self, row: pd.Series) -> str:
        """Create a prompt for analyzing a student's answer."""
        return f"""You are an expert educational analyst. Analyze the following student response and provide detailed feedback.

Analyze the following input record carefully:
    
Topic: {row['Topic']}
Sub_Section: {row['Sub_Section']}
Sub_Topic: {row['Sub_Topic']}
Learning_Objective: {row['Learning_Objective']}
Category: {row['Category']}
Question: {row['Question']}
Correct Answer: {row['Answer_Key']}
Student's Answer: {row['Student_Answer']}
Is_Correct: {row['Is_Correct']}
Why_Wrong: {row['Why_Wrong']}


Provide a JSON response with this structure:
{{
    "detailed_feedback": {{
        "facts": {{
            "question": "Precise restatement of the question in your own words.",
            "formulas": ["List", "of", "relevant", "formulas"],
            "key_concepts": ["List", "of", "key", "concepts"]
        }},
        "step_by_step": [
            "Step 1: First action taken",
            "Step 2: Second action taken",
            "Step 3: And so on..."
        ],
        "strategy": "High-level explanation of why this method was chosen over alternatives",
        "rationale": "Explanation of why this answer makes sense in the given context and aligns with mathematical patterns"
    }}
}}

Respond with only the JSON object, no additional text. Ensure the JSON is properly formatted and valid."""

    def generate_report(self, analysis_df: pd.DataFrame) -> str:
        """
        Generate a PDF report from the analysis.
        
        Args:
            analysis_df (pd.DataFrame): DataFrame with analysis results
            
        Returns:
            str: Path to the generated PDF
        """
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Add title page
        self._add_title_page(pdf, "Student Response Analysis Report")
        
        # Add summary statistics
        self._add_summary_section(pdf, analysis_df)
        
        # Add detailed analysis by topic
        for topic, group in analysis_df.groupby('Topic'):
            self._add_topic_section(pdf, topic, group)
            
        # Save the PDF
        filename = os.path.join(self.output_dir, f"student_analysis_{self.timestamp}.pdf")
        pdf.output(filename)
        return filename
    
    def _add_title_page(self, pdf: FPDF, title: str) -> None:
        """Add a title page to the PDF."""
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, title, 0, 1, 'C')
        pdf.ln(10)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
        pdf.ln(20)
    
    def _add_summary_section(self, pdf: FPDF, df: pd.DataFrame) -> None:
        """Add summary statistics to the PDF."""
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Summary Statistics', 0, 1)
        pdf.ln(5)
        
        # Calculate statistics
        total_questions = len(df)
        correct_answers = df['Is_Correct'].sum()
        accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"Total Questions Analyzed: {total_questions}", 0, 1)
        pdf.cell(0, 10, f"Correct Answers: {correct_answers} ({accuracy:.1f}%)", 0, 1)
        pdf.ln(10)
        
        # Add topic-wise performance
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Performance by Topic:', 0, 1)
        
        for topic, group in df.groupby('Topic'):
            topic_total = len(group)
            topic_correct = group['Is_Correct'].sum()
            topic_accuracy = (topic_correct / topic_total) * 100 if topic_total > 0 else 0
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 8, f"- {topic}: {topic_correct}/{topic_total} correct ({topic_accuracy:.1f}%)", 0, 1)
        
        pdf.ln(10)
    
    def _add_topic_section(self, pdf: FPDF, topic: str, df: pd.DataFrame) -> None:
        """Add a section for a specific topic with detailed analysis."""
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, f"Topic: {topic}", 0, 1)
        pdf.ln(5)
        
        for _, row in df.iterrows():
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, f"Question: {row['Question']}", 0, 1)
            
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 6, f"Student Answer: {row['Student_Answer']}", 0, 1)
            pdf.cell(0, 6, f"Correct Answer: {row['Answer_Key']}", 0, 1)
            
            # Add detailed analysis if available
            if 'detailed_feedback' in row and pd.notna(row['detailed_feedback']):
                try:
                    detailed = json.loads(str(row['detailed_feedback']).replace("'", '"'))
                    
                    # Add step-by-step solution
                    if 'step_by_step' in detailed:
                        pdf.set_font('Arial', 'B', 10)
                        pdf.cell(0, 6, "Step-by-Step Solution:", 0, 1)
                        pdf.set_font('Arial', '', 10)
                        
                        for i, step in enumerate(detailed['step_by_step'], 1):
                            # Handle both string and dictionary formats
                            if isinstance(step, dict) and 'step' in step:
                                step_text = step['step']
                            elif isinstance(step, str):
                                step_text = step
                            else:
                                continue
                                
                            pdf.multi_cell(0, 6, f"{i}. {step_text}")
                            pdf.ln(2)
                    
                    # Add strategy and rationale if available
                    if 'strategy' in detailed:
                        pdf.set_font('Arial', 'B', 10)
                        pdf.cell(0, 6, "Strategy:", 0, 1)
                        pdf.set_font('Arial', '', 10)
                        pdf.multi_cell(0, 6, str(detailed['strategy']))
                        pdf.ln(2)
                        
                    if 'rationale' in detailed:
                        pdf.set_font('Arial', 'B', 10)
                        pdf.cell(0, 6, "Rationale:", 0, 1)
                        pdf.set_font('Arial', '', 10)
                        pdf.multi_cell(0, 6, str(detailed['rationale']))
                        pdf.ln(2)
                        
                except Exception as e:
                    print(f"Error processing detailed feedback: {str(e)}")
                    pdf.set_font('Arial', 'I', 10)
                    pdf.multi_cell(0, 6, "Detailed analysis could not be displayed due to formatting issues.")
            
            # Add common misconceptions and interventions for incorrect answers
            if not row['Is_Correct']:
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 6, "Common Misconceptions:", 0, 1)
                pdf.set_font('Arial', '', 10)
                pdf.multi_cell(0, 6, row.get('Common_Misconceptions', 'Not available'))
                
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 6, "Suggested Intervention:", 0, 1)
                pdf.set_font('Arial', '', 10)
                pdf.multi_cell(0, 6, row.get('Suggested_Intervention', 'Not available'))
            
            pdf.ln(10)

def analyze_student_responses(csv_path: str, output_dir: str = 'reports', batch_size: int = 5, max_workers: int = 3) -> dict:
    """
    Analyze student responses from a CSV file and generate a report.
    
    Args:
        csv_path (str): Path to the CSV file with student responses
        output_dir (str): Directory to save the generated report
        batch_size (int): Number of answers to process in each batch
        max_workers (int): Maximum number of parallel workers
        
    Returns:
        dict: Dictionary containing analysis results and report path
    """
    try:
        print(f"[INFO] Starting analysis of {csv_path}")
        start_time = datetime.now()
        
        # Read the CSV file
        print("[INFO] Reading CSV file...")
        df = pd.read_csv(csv_path)
        print(f"[INFO] Loaded {len(df)} student responses")
        
        # Initialize the analyzer
        analyzer = OllamaReportGenerator(df, output_dir)
        
        # Analyze the responses with progress tracking
        print("\n[INFO] Starting analysis of student responses...")
        print("Progress will be shown below. This may take some time depending on the number of responses.\n")
        
        analysis_df = analyzer.analyze_student_answers(
            batch_size=batch_size,
            max_workers=max_workers
        )
        
        if analysis_df.empty:
            print("\n[WARNING] No valid student responses were analyzed.")
            return {
                'success': False,
                'message': 'No valid student responses were analyzed.',
                'report_path': None,
                'csv_path': None
            }
        
        # Save the analysis to a new CSV
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_csv = os.path.join(output_dir, f"analyzed_responses_{timestamp}.csv")
        analysis_df.to_csv(output_csv, index=False)
        
        # Generate the PDF report
        print("\n[INFO] Generating PDF report...")
        report_path = analyzer.generate_report(analysis_df)
        
        # Calculate total processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n[SUCCESS] Analysis completed in {processing_time:.1f} seconds")
        print(f"- Analyzed {len(analysis_df)} student responses")
        print(f"- CSV saved to: {output_csv}")
        print(f"- PDF report saved to: {report_path}")
        
        return {
            'success': True,
            'message': 'Analysis completed successfully',
            'report_path': report_path,
            'csv_path': output_csv,
            'num_responses': len(analysis_df),
            'processing_time_seconds': processing_time
        }
        
    except Exception as e:
        error_msg = f"Error during analysis: {str(e)}"
        print(f"\n[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'message': error_msg,
            'report_path': None,
            'csv_path': None
        }

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python ollama_report_generator.py <path_to_student_responses.csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    report_path = analyze_student_responses(csv_path)
    print(f"Analysis complete! Report saved to: {report_path}")
