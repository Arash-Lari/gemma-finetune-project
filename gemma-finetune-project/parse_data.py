import os
import random
import json
import re
import pandas as pd
import fitz  # PyMuPDF
import requests
from PIL import Image
import io
import base64
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from tqdm import tqdm
import time
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataParser:
    def __init__(self, data_dir: str, vision_api_url: str = "http://localhost:1234/v1"):
        """
        Initialize the data parser.
        
        Args:
            data_dir: Path to training data directory
            vision_api_url: URL for LM Studio vision API
        """
        self.data_dir = os.path.expanduser(data_dir)
        self.vision_api_url = vision_api_url
        self.supported_extensions = {'.pdf', '.xlsx', '.txt', '.csv'}
        self.progress_callback = None
        self.pause_flag = False
        
    def set_progress_callback(self, callback):
        """Set callback for progress updates."""
        self.progress_callback = callback
        
    def set_pause_flag(self, pause: bool):
        """Set pause flag for processing."""
        self.pause_flag = pause
        
    def _update_progress(self, current: int, total: int, stage: str, file_name: str = "", eta: str = ""):
        """Update progress through callback."""
        if self.progress_callback:
            self.progress_callback({
                'current': current,
                'total': total,
                'percentage': (current / total * 100) if total > 0 else 0,
                'stage': stage,
                'file_name': file_name,
                'eta': eta,
                'timestamp': datetime.now().isoformat()
            })
    
    def sample_files_for_validation(self, num_samples: int = 5) -> List[Dict[str, Any]]:
        """
        Sample files for validation/testing purposes only.
        This is used to verify parsing works correctly.
        
        Args:
            num_samples: Number of files to sample
            
        Returns:
            List of file information dictionaries
        """
        logger.info(f"Sampling {num_samples} files for validation from {self.data_dir}")
        
        # Get all files by extension
        files_by_ext = {}
        for ext in self.supported_extensions:
            files_by_ext[ext] = []
            
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                if ext in self.supported_extensions:
                    files_by_ext[ext].append(file_path)
        
        # Sample files (balanced across formats)
        sampled_files = []
        samples_per_type = max(1, num_samples // len(self.supported_extensions))
        
        for ext, files in files_by_ext.items():
            if files:
                # Sample up to samples_per_type files of this type
                sample_size = min(samples_per_type, len(files))
                sampled = random.sample(files, sample_size)
                for file_path in sampled:
                    sampled_files.append({
                        'path': file_path,
                        'extension': ext,
                        'filename': os.path.basename(file_path)
                    })
        
        logger.info(f"Sampled {len(sampled_files)} files for validation")
        return sampled_files
    
    def extract_text_from_pdf_vision(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from PDF using vision model via LM Studio API.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of extracted text entries
        """
        logger.info(f"Extracting text from PDF using vision: {pdf_path}")
        
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            extracted_data = []
            
            for page_num in range(len(doc)):
                # Check for pause
                while self.pause_flag:
                    time.sleep(0.1)
                
                # Convert page to image
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution
                img_data = pix.tobytes("png")
                
                # Encode image to base64
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                
                # Prepare vision prompt
                vision_prompt = f"""Analyze this phone chat screenshot and extract all messages in the following format:
                - Sender name
                - Timestamp (if visible)
                - Message content
                
                If this is a chat interface, identify each message bubble and extract the sender and content.
                If timestamps are visible, include them.
                Format the output as structured text that can be parsed."""
                
                # Call vision API
                response = self._call_vision_api(img_base64, vision_prompt)
                
                if response:
                    extracted_data.append({
                        'page': page_num + 1,
                        'extracted_text': response,
                        'source_file': pdf_path
                    })
            
            doc.close()
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error extracting from PDF {pdf_path}: {e}")
            return []
    
    def _call_vision_api(self, img_base64: str, prompt: str) -> Optional[str]:
        """
        Call LM Studio vision API.
        
        Args:
            img_base64: Base64 encoded image
            prompt: Vision prompt
            
        Returns:
            API response or None if failed
        """
        try:
            payload = {
                "model": "gemma-3-vision-latex",  # Adjust based on your loaded model
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                        ]
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.1
            }
            
            response = requests.post(
                f"{self.vision_api_url}/chat/completions",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"Vision API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling vision API: {e}")
            return None
    
    def parse_txt_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse TXT file with chat logs.
        
        Args:
            file_path: Path to TXT file
            
        Returns:
            List of parsed entries
        """
        logger.info(f"Parsing TXT file: {file_path}")
        
        entries = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Try different parsing strategies
            for line_num, line in enumerate(lines):
                # Check for pause
                while self.pause_flag:
                    time.sleep(0.1)
                
                line = line.strip()
                if not line:
                    continue
                
                # Try to extract timestamp and sender
                # Common patterns: [timestamp] sender: message
                timestamp_match = re.search(r'\[([^\]]+)\]', line)
                sender_match = re.search(r'(\w+):\s*(.+)', line)
                
                if timestamp_match and sender_match:
                    timestamp = timestamp_match.group(1)
                    sender = sender_match.group(1)
                    content = sender_match.group(2)
                elif sender_match:
                    sender = sender_match.group(1)
                    content = sender_match.group(2)
                    timestamp = None
                else:
                    # Fallback: treat as general content
                    sender = "Unknown"
                    content = line
                    timestamp = None
                
                entries.append({
                    'source_file': file_path,
                    'file_type': 'txt',
                    'line_number': line_num + 1,
                    'speaker': sender,
                    'timestamp': timestamp,
                    'content': content
                })
            
            logger.info(f"Parsed {len(entries)} entries from TXT file")
            return entries
            
        except Exception as e:
            logger.error(f"Error parsing TXT file {file_path}: {e}")
            return []
    
    def parse_csv_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse CSV file with structured data.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of parsed entries
        """
        logger.info(f"Parsing CSV file: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            entries = []
            
            # Try to identify relevant columns
            possible_sender_cols = ['sender', 'speaker', 'user', 'name', 'author']
            possible_content_cols = ['message', 'content', 'text', 'body']
            possible_time_cols = ['timestamp', 'time', 'date', 'datetime']
            
            sender_col = None
            content_col = None
            time_col = None
            
            # Find relevant columns
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in possible_sender_cols:
                    sender_col = col
                elif col_lower in possible_content_cols:
                    content_col = col
                elif col_lower in possible_time_cols:
                    time_col = col
            
            # If we can't find expected columns, use first two columns
            if not sender_col and len(df.columns) >= 2:
                sender_col = df.columns[0]
            if not content_col and len(df.columns) >= 2:
                content_col = df.columns[1]
            
            for idx, row in df.iterrows():
                # Check for pause
                while self.pause_flag:
                    time.sleep(0.1)
                
                sender = str(row[sender_col]) if sender_col else "Unknown"
                content = str(row[content_col]) if content_col else str(row.iloc[0])
                timestamp = str(row[time_col]) if time_col else None
                
                entries.append({
                    'source_file': file_path,
                    'file_type': 'csv',
                    'row_number': idx + 1,
                    'speaker': sender,
                    'timestamp': timestamp,
                    'content': content
                })
            
            logger.info(f"Parsed {len(entries)} entries from CSV file")
            return entries
            
        except Exception as e:
            logger.error(f"Error parsing CSV file {file_path}: {e}")
            return []
    
    def parse_xlsx_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse XLSX file with structured data.
        
        Args:
            file_path: Path to XLSX file
            
        Returns:
            List of parsed entries
        """
        logger.info(f"Parsing XLSX file: {file_path}")
        
        try:
            df = pd.read_excel(file_path)
            entries = []
            
            # Similar logic to CSV parsing
            possible_sender_cols = ['sender', 'speaker', 'user', 'name', 'author']
            possible_content_cols = ['message', 'content', 'text', 'body']
            possible_time_cols = ['timestamp', 'time', 'date', 'datetime']
            
            sender_col = None
            content_col = None
            time_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in possible_sender_cols:
                    sender_col = col
                elif col_lower in possible_content_cols:
                    content_col = col
                elif col_lower in possible_time_cols:
                    time_col = col
            
            if not sender_col and len(df.columns) >= 2:
                sender_col = df.columns[0]
            if not content_col and len(df.columns) >= 2:
                content_col = df.columns[1]
            
            for idx, row in df.iterrows():
                # Check for pause
                while self.pause_flag:
                    time.sleep(0.1)
                
                sender = str(row[sender_col]) if sender_col else "Unknown"
                content = str(row[content_col]) if content_col else str(row.iloc[0])
                timestamp = str(row[time_col]) if time_col else None
                
                entries.append({
                    'source_file': file_path,
                    'file_type': 'xlsx',
                    'row_number': idx + 1,
                    'speaker': sender,
                    'timestamp': timestamp,
                    'content': content
                })
            
            logger.info(f"Parsed {len(entries)} entries from XLSX file")
            return entries
            
        except Exception as e:
            logger.error(f"Error parsing XLSX file {file_path}: {e}")
            return []
    
    def parse_all_files(self, chunk_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Parse ALL files in the data directory thoroughly.
        
        Args:
            chunk_size: Number of entries to process in each chunk
            
        Returns:
            List of all parsed entries
        """
        logger.info("Starting to parse ALL files thoroughly...")
        
        all_entries = []
        
        # Get all files
        all_files = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                if ext in self.supported_extensions:
                    all_files.append((file_path, ext))
        
        logger.info(f"Found {len(all_files)} files to parse thoroughly")
        
        # Parse files with progress tracking
        start_time = time.time()
        for i, (file_path, ext) in enumerate(all_files):
            # Check for pause
            while self.pause_flag:
                time.sleep(0.1)
            
            # Calculate ETA
            if i > 0:
                elapsed_time = time.time() - start_time
                avg_time_per_file = elapsed_time / i
                remaining_files = len(all_files) - i
                eta_seconds = avg_time_per_file * remaining_files
                eta = str(timedelta(seconds=int(eta_seconds)))
            else:
                eta = "Calculating..."
            
            # Update progress
            self._update_progress(
                current=i + 1,
                total=len(all_files),
                stage="Parsing files",
                file_name=os.path.basename(file_path),
                eta=eta
            )
            
            try:
                if ext == '.pdf':
                    entries = self.extract_text_from_pdf_vision(file_path)
                    # Convert vision extraction to standard format
                    for entry in entries:
                        all_entries.append({
                            'source_file': file_path,
                            'file_type': 'pdf',
                            'page': entry.get('page'),
                            'speaker': 'Extracted',
                            'timestamp': None,
                            'content': entry.get('extracted_text', '')
                        })
                elif ext == '.txt':
                    entries = self.parse_txt_file(file_path)
                    all_entries.extend(entries)
                elif ext == '.csv':
                    entries = self.parse_csv_file(file_path)
                    all_entries.extend(entries)
                elif ext == '.xlsx':
                    entries = self.parse_xlsx_file(file_path)
                    all_entries.extend(entries)
                
                # Process in chunks to avoid memory issues
                if len(all_entries) >= chunk_size:
                    logger.info(f"Processed {len(all_entries)} entries so far...")
                    
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue
        
        logger.info(f"Finished parsing ALL files. Total entries: {len(all_entries)}")
        return all_entries
    
    def save_parsed_data(self, entries: List[Dict[str, Any]], output_path: str):
        """
        Save parsed data to JSON file.
        
        Args:
            entries: List of parsed entries
            output_path: Output file path
        """
        logger.info(f"Saving {len(entries)} entries to {output_path}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Successfully saved parsed data to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving parsed data: {e}")

def main():
    """Test the parser with sample files."""
    parser = DataParser("~/Desktop/trainingdata/")
    
    # Sample files first for validation
    sampled_files = parser.sample_files_for_validation(num_samples=5)
    print("Sampled files for validation:")
    for file_info in sampled_files:
        print(f"  {file_info['filename']} ({file_info['extension']})")
    
    # Parse all files thoroughly
    entries = parser.parse_all_files()
    
    # Save parsed data
    output_path = "output/parsed_data.json"
    parser.save_parsed_data(entries, output_path)

if __name__ == "__main__":
    main() 