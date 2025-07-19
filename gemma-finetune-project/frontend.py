#!/usr/bin/env python3
"""
Sleek Frontend for Gemma Fine-tuning Project

Provides a modern GUI with progress visualization, pause/resume functionality,
and detailed progress tracking.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import queue
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from parse_data import DataParser
from prepare_dataset import DatasetPreparer
from finetune import GemmaFineTuner

class FineTuningFrontend:
    def __init__(self, root):
        self.root = root
        self.root.title("Gemma Fine-tuning Project")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2b2b2b')
        
        # Initialize variables
        self.progress_queue = queue.Queue()
        self.pause_flag = False
        self.current_process = None
        self.process_thread = None
        
        # Setup UI
        self.setup_ui()
        self.setup_styles()
        
        # Start progress monitoring
        self.monitor_progress()
        
    def setup_styles(self):
        """Setup modern styling."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', 
                       font=('Arial', 16, 'bold'),
                       foreground='#ffffff',
                       background='#2b2b2b')
        
        style.configure('Subtitle.TLabel',
                       font=('Arial', 12),
                       foreground='#cccccc',
                       background='#2b2b2b')
        
        style.configure('Progress.Horizontal.TProgressbar',
                       troughcolor='#404040',
                       background='#4CAF50',
                       bordercolor='#2b2b2b',
                       lightcolor='#4CAF50',
                       darkcolor='#4CAF50')
        
        style.configure('Status.TLabel',
                       font=('Arial', 10),
                       foreground='#00ff00',
                       background='#2b2b2b')
        
    def setup_ui(self):
        """Setup the user interface."""
        # Main container
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(main_frame, 
                               text="Gemma-3-27B Fine-tuning Project",
                               style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Configuration section
        self.setup_config_section(main_frame)
        
        # Progress section
        self.setup_progress_section(main_frame)
        
        # Control section
        self.setup_control_section(main_frame)
        
        # Log section
        self.setup_log_section(main_frame)
        
    def setup_config_section(self, parent):
        """Setup configuration section."""
        config_frame = tk.LabelFrame(parent, text="Configuration", 
                                   fg='#ffffff', bg='#2b2b2b', 
                                   font=('Arial', 12, 'bold'))
        config_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Model path
        tk.Label(config_frame, text="Model Path:", fg='#cccccc', bg='#2b2b2b').grid(row=0, column=0, sticky='w', padx=10, pady=5)
        self.model_path_var = tk.StringVar(value="~/models/mlabonne/gemma-3-27B-it-abliterated-GGUF/gemma-3-27B-it-abliterated.q4_k_m.gguf")
        self.model_path_entry = tk.Entry(config_frame, textvariable=self.model_path_var, width=50, bg='#404040', fg='#ffffff')
        self.model_path_entry.grid(row=0, column=1, padx=10, pady=5)
        
        # Data directory
        tk.Label(config_frame, text="Data Directory:", fg='#cccccc', bg='#2b2b2b').grid(row=1, column=0, sticky='w', padx=10, pady=5)
        self.data_dir_var = tk.StringVar(value="~/Desktop/trainingdata/")
        self.data_dir_entry = tk.Entry(config_frame, textvariable=self.data_dir_var, width=50, bg='#404040', fg='#ffffff')
        self.data_dir_entry.grid(row=1, column=1, padx=10, pady=5)
        
        # Training parameters
        params_frame = tk.Frame(config_frame, bg='#2b2b2b')
        params_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        tk.Label(params_frame, text="Epochs:", fg='#cccccc', bg='#2b2b2b').grid(row=0, column=0, padx=10)
        self.epochs_var = tk.StringVar(value="3")
        tk.Entry(params_frame, textvariable=self.epochs_var, width=10, bg='#404040', fg='#ffffff').grid(row=0, column=1, padx=5)
        
        tk.Label(params_frame, text="Batch Size:", fg='#cccccc', bg='#2b2b2b').grid(row=0, column=2, padx=10)
        self.batch_size_var = tk.StringVar(value="4")
        tk.Entry(params_frame, textvariable=self.batch_size_var, width=10, bg='#404040', fg='#ffffff').grid(row=0, column=3, padx=5)
        
        tk.Label(params_frame, text="Learning Rate:", fg='#cccccc', bg='#2b2b2b').grid(row=0, column=4, padx=10)
        self.lr_var = tk.StringVar(value="1e-5")
        tk.Entry(params_frame, textvariable=self.lr_var, width=10, bg='#404040', fg='#ffffff').grid(row=0, column=5, padx=5)
        
    def setup_progress_section(self, parent):
        """Setup progress visualization section."""
        progress_frame = tk.LabelFrame(parent, text="Progress", 
                                     fg='#ffffff', bg='#2b2b2b',
                                     font=('Arial', 12, 'bold'))
        progress_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Overall progress
        tk.Label(progress_frame, text="Overall Progress:", fg='#cccccc', bg='#2b2b2b').pack(anchor='w', padx=10, pady=(10, 5))
        self.overall_progress = ttk.Progressbar(progress_frame, style='Progress.Horizontal.TProgressbar', length=400)
        self.overall_progress.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Stage progress
        tk.Label(progress_frame, text="Current Stage:", fg='#cccccc', bg='#2b2b2b').pack(anchor='w', padx=10, pady=(10, 5))
        self.stage_progress = ttk.Progressbar(progress_frame, style='Progress.Horizontal.TProgressbar', length=400)
        self.stage_progress.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Status labels
        self.status_frame = tk.Frame(progress_frame, bg='#2b2b2b')
        self.status_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.stage_label = tk.Label(self.status_frame, text="Ready", fg='#00ff00', bg='#2b2b2b', font=('Arial', 10, 'bold'))
        self.stage_label.pack(anchor='w')
        
        self.file_label = tk.Label(self.status_frame, text="", fg='#cccccc', bg='#2b2b2b', font=('Arial', 9))
        self.file_label.pack(anchor='w')
        
        self.eta_label = tk.Label(self.status_frame, text="", fg='#cccccc', bg='#2b2b2b', font=('Arial', 9))
        self.eta_label.pack(anchor='w')
        
        self.percentage_label = tk.Label(self.status_frame, text="", fg='#4CAF50', bg='#2b2b2b', font=('Arial', 12, 'bold'))
        self.percentage_label.pack(anchor='w')
        
    def setup_control_section(self, parent):
        """Setup control buttons section."""
        control_frame = tk.Frame(parent, bg='#2b2b2b')
        control_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Buttons
        self.start_button = tk.Button(control_frame, text="Start Processing", 
                                     command=self.start_processing,
                                     bg='#4CAF50', fg='white', 
                                     font=('Arial', 12, 'bold'),
                                     relief=tk.FLAT, padx=20, pady=10)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.pause_button = tk.Button(control_frame, text="Pause", 
                                     command=self.toggle_pause,
                                     bg='#FF9800', fg='white',
                                     font=('Arial', 12, 'bold'),
                                     relief=tk.FLAT, padx=20, pady=10,
                                     state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = tk.Button(control_frame, text="Stop", 
                                    command=self.stop_processing,
                                    bg='#f44336', fg='white',
                                    font=('Arial', 12, 'bold'),
                                    relief=tk.FLAT, padx=20, pady=10,
                                    state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Test setup button
        self.test_button = tk.Button(control_frame, text="Test Setup", 
                                    command=self.test_setup,
                                    bg='#2196F3', fg='white',
                                    font=('Arial', 12, 'bold'),
                                    relief=tk.FLAT, padx=20, pady=10)
        self.test_button.pack(side=tk.RIGHT)
        
    def setup_log_section(self, parent):
        """Setup log display section."""
        log_frame = tk.LabelFrame(parent, text="Log Output", 
                                 fg='#ffffff', bg='#2b2b2b',
                                 font=('Arial', 12, 'bold'))
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, 
                                                bg='#1e1e1e', fg='#ffffff',
                                                font=('Consolas', 9),
                                                height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def log_message(self, message):
        """Add message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def update_progress(self, progress_data):
        """Update progress display."""
        try:
            current = progress_data.get('current', 0)
            total = progress_data.get('total', 1)
            percentage = progress_data.get('percentage', 0)
            stage = progress_data.get('stage', 'Unknown')
            file_name = progress_data.get('file_name', '')
            eta = progress_data.get('eta', '')
            
            # Update progress bars
            if total > 0:
                self.overall_progress['value'] = percentage
                self.stage_progress['value'] = (current / total) * 100
            
            # Update labels
            self.stage_label.config(text=f"Stage: {stage}")
            self.file_label.config(text=f"File: {file_name}" if file_name else "")
            self.eta_label.config(text=f"ETA: {eta}" if eta else "")
            self.percentage_label.config(text=f"{percentage:.1f}%")
            
            # Update button states
            if self.current_process:
                self.pause_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.NORMAL)
                self.start_button.config(state=tk.DISABLED)
            
        except Exception as e:
            self.log_message(f"Error updating progress: {e}")
            
    def monitor_progress(self):
        """Monitor progress queue."""
        try:
            while True:
                try:
                    progress_data = self.progress_queue.get_nowait()
                    self.update_progress(progress_data)
                except queue.Empty:
                    break
        except Exception as e:
            self.log_message(f"Error in progress monitoring: {e}")
        finally:
            self.root.after(100, self.monitor_progress)
            
    def progress_callback(self, progress_data):
        """Callback for progress updates."""
        self.progress_queue.put(progress_data)
        
    def start_processing(self):
        """Start the processing pipeline."""
        if self.process_thread and self.process_thread.is_alive():
            messagebox.showwarning("Warning", "Processing is already running!")
            return
            
        self.log_message("Starting fine-tuning pipeline...")
        self.current_process = "fine_tuning"
        
        # Start processing in separate thread
        self.process_thread = threading.Thread(target=self.run_pipeline)
        self.process_thread.daemon = True
        self.process_thread.start()
        
    def run_pipeline(self):
        """Run the complete pipeline."""
        try:
            # Get configuration
            model_path = self.model_path_var.get()
            data_dir = self.data_dir_var.get()
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_size_var.get())
            learning_rate = float(self.lr_var.get())
            
            self.log_message(f"Configuration loaded:")
            self.log_message(f"  Model: {model_path}")
            self.log_message(f"  Data: {data_dir}")
            self.log_message(f"  Epochs: {epochs}")
            self.log_message(f"  Batch Size: {batch_size}")
            self.log_message(f"  Learning Rate: {learning_rate}")
            
            # Step 1: Parse data
            self.log_message("Step 1: Parsing training data...")
            parser = DataParser(data_dir)
            parser.set_progress_callback(self.progress_callback)
            parser.set_pause_flag(self.pause_flag)
            
            parsed_data = parser.parse_all_files()
            
            if not parsed_data:
                self.log_message("ERROR: No data found to parse!")
                return
                
            self.log_message(f"Parsed {len(parsed_data)} entries")
            
            # Save parsed data
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            parsed_path = output_dir / "parsed_data.json"
            parser.save_parsed_data(parsed_data, str(parsed_path))
            
            # Step 2: Prepare dataset
            self.log_message("Step 2: Preparing training dataset...")
            preparer = DatasetPreparer()
            train_data_path = output_dir / "train.jsonl"
            preparer.prepare_training_data(parsed_data, str(train_data_path))
            
            # Create validation split
            val_data_path = output_dir / "train_val.jsonl"
            preparer.create_validation_split(str(train_data_path))
            
            # Step 3: Fine-tuning
            self.log_message("Step 3: Starting fine-tuning...")
            tuner = GemmaFineTuner(model_path)
            
            results = tuner.run_finetuning_pipeline(
                data_dir=data_dir,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            self.log_message("Fine-tuning completed successfully!")
            self.log_message("Output files:")
            for key, path in results.items():
                self.log_message(f"  {key}: {path}")
                
        except Exception as e:
            self.log_message(f"ERROR: {e}")
            messagebox.showerror("Error", f"Processing failed: {e}")
        finally:
            self.current_process = None
            self.root.after(0, self.reset_buttons)
            
    def toggle_pause(self):
        """Toggle pause/resume."""
        self.pause_flag = not self.pause_flag
        if self.pause_flag:
            self.pause_button.config(text="Resume", bg='#4CAF50')
            self.log_message("Processing paused")
        else:
            self.pause_button.config(text="Pause", bg='#FF9800')
            self.log_message("Processing resumed")
            
    def stop_processing(self):
        """Stop processing."""
        if messagebox.askyesno("Confirm", "Are you sure you want to stop processing?"):
            self.pause_flag = True
            self.current_process = None
            self.log_message("Processing stopped by user")
            self.reset_buttons()
            
    def reset_buttons(self):
        """Reset button states."""
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED, text="Pause", bg='#FF9800')
        self.stop_button.config(state=tk.DISABLED)
        
    def test_setup(self):
        """Test the project setup."""
        self.log_message("Testing project setup...")
        
        try:
            # Import and run test
            from test_setup import main as test_main
            
            # Capture test output
            import io
            import sys
            
            # Redirect stdout to capture test output
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                test_main()
                output = sys.stdout.getvalue()
                self.log_message("Test output:")
                for line in output.split('\n'):
                    if line.strip():
                        self.log_message(line.strip())
            finally:
                sys.stdout = old_stdout
                
        except Exception as e:
            self.log_message(f"Test failed: {e}")
            
    def on_closing(self):
        """Handle window closing."""
        if self.current_process:
            if messagebox.askyesno("Confirm", "Processing is running. Do you want to stop and exit?"):
                self.pause_flag = True
                self.current_process = None
        self.root.destroy()

def main():
    """Main function to run the frontend."""
    root = tk.Tk()
    app = FineTuningFrontend(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start the GUI
    root.mainloop()

if __name__ == "__main__":
    main() 