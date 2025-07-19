# Gemma-3-27B-it-Abliterated Fine-tuning Project

This project fine-tunes the Gemma-3-27B-it-Abliterated model on personal chat logs and journal data using MLX-LM with LoRA adapters. The system supports multiple data formats including PDF (with vision extraction), XLSX, TXT, and CSV files.

## Features

- **Multi-format Data Processing**: Handles PDF, XLSX, TXT, and CSV files
- **Vision-based PDF Extraction**: Uses Gemma-3-Vision-Latex for extracting text from visual PDFs
- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning with MLX-LM
- **Local Processing**: All processing done locally, no cloud dependencies
- **Conversation Context**: Maintains conversation context across entries
- **Flexible Output**: Supports both MLX and GGUF model formats
- **Sleek Frontend**: Modern GUI with progress visualization, pause/resume, and ETA
- **Thorough Processing**: Processes ALL files completely, no random sampling

## Project Structure

```
gemma-finetune-project/
├── finetune.py          # Main fine-tuning script
├── parse_data.py        # Data parsing and vision extraction
├── prepare_dataset.py   # Dataset preparation for Gemma format
├── frontend.py          # Sleek GUI frontend
├── config.py           # Configuration file for customization
├── test_setup.py       # Setup verification script
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── output/             # Output directory for results
    ├── adapters/       # LoRA adapters
    ├── merged_model/   # Merged fine-tuned model
    └── logs/          # Training logs
```

## Prerequisites

### System Requirements
- macOS with Apple Silicon (M1/M2/M3)
- 128GB RAM recommended for 27B model
- Python 3.8+

### Model Requirements
- **Base Model**: Gemma-3-27B-it-Abliterated (GGUF or MLX format)
  - GGUF: `~/models/mlabonne/gemma-3-27B-it-abliterated-GGUF/gemma-3-27B-it-abliterated.q4_k_m.gguf`
  - MLX: `~/models/mlx-community/gemma-3-27B-it-qat-4bit`

- **Vision Model** (for PDF processing): Gemma-3-Vision-Latex
  - `~/models/gemma-3-vision-latex.Q4_K_S.gguf`
  - `~/models/gemma-3-vision-latex.mmproj-f16.gguf`

### Data Requirements
- Training data in `~/Desktop/trainingdata/`
- Supported formats: PDF, XLSX, TXT, CSV

## Installation

1. **Clone and setup the project**:
```bash
cd gemma-finetune-project
pip install -r requirements.txt
```

2. **Install MLX-LM** (if not already installed):
```bash
pip install mlx-lm
```

3. **Setup LM Studio for vision processing**:
   - Install LM Studio
   - Load the vision model (Gemma-3-Vision-Latex)
   - Start the API server on `http://localhost:1234`

## Usage

### GUI Frontend (Recommended)

Launch the sleek frontend for the best experience:

```bash
python frontend.py
```

**Frontend Features:**
- 🎨 Modern dark theme GUI
- 📊 Real-time progress visualization
- ⏸️ Pause/Resume functionality
- ⏱️ ETA calculation and display
- 📝 Live log output
- ⚙️ Easy configuration management
- 🧪 Built-in setup testing

### Command Line Usage

**Quick Start**:
```bash
python finetune.py \
    --model_path ~/models/mlabonne/gemma-3-27B-it-abliterated-GGUF/gemma-3-27B-it-abliterated.q4_k_m.gguf \
    --data_dir ~/Desktop/trainingdata/ \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 1e-5
```

**Advanced Usage**:
```bash
python finetune.py \
    --model_path ~/models/mlx-community/gemma-3-27B-it-qat-4bit \
    --vision_model_path ~/models/gemma-3-vision-latex.Q4_K_S.gguf \
    --data_dir ~/Desktop/trainingdata/ \
    --epochs 5 \
    --batch_size 2 \
    --learning_rate 5e-6 \
    --lora_rank 32 \
    --max_entries 10000 \
    --convert_to_gguf
```

**Step-by-step processing**:
```bash
# 1. Parse data only
python parse_data.py

# 2. Prepare dataset only
python prepare_dataset.py

# 3. Run fine-tuning only
python finetune.py --model_path /path/to/model --skip_data_processing
```

## Frontend Features

### Progress Visualization
- **Overall Progress Bar**: Shows complete pipeline progress
- **Stage Progress Bar**: Shows current stage progress
- **Real-time Status**: Current file being processed
- **ETA Display**: Estimated time to completion
- **Percentage Display**: Exact completion percentage

### Control Features
- **Start Processing**: Begin the complete pipeline
- **Pause/Resume**: Pause processing at any time
- **Stop**: Safely stop processing
- **Test Setup**: Verify all dependencies and paths

### Configuration Management
- **Model Path**: Set your Gemma model path
- **Data Directory**: Set your training data location
- **Training Parameters**: Epochs, batch size, learning rate
- **Real-time Validation**: Check paths and settings

### Log Output
- **Live Logging**: Real-time log display
- **Timestamped Messages**: All events with timestamps
- **Error Highlighting**: Clear error identification
- **Scrollable History**: View complete processing history

## Configuration

### Model Paths
- **Base Model**: Path to your Gemma-3-27B-it-Abliterated model
- **Vision Model**: Path to vision model for PDF processing (optional)

### Training Parameters
- `--epochs`: Number of training epochs (default: 3)
- `--batch_size`: Training batch size (default: 4)
- `--learning_rate`: Learning rate (default: 1e-5)
- `--lora_rank`: LoRA rank (default: 16)
- `--max_entries`: Maximum entries to process (default: all)

### Data Processing
- `--data_dir`: Path to training data directory
- `--max_entries`: Limit number of entries for testing
- `--convert_to_gguf`: Convert final model to GGUF format

## Data Format Support

### PDF Files
- **Visual PDFs**: Uses vision model to extract text from chat screenshots
- **Text PDFs**: Direct text extraction with PyMuPDF
- **Processing**: Page-by-page analysis with conversation context

### XLSX/CSV Files
- **Auto-detection**: Automatically identifies sender, content, and timestamp columns
- **Fallback**: Uses first two columns if standard column names not found
- **Structured**: Maintains conversation order and metadata

### TXT Files
- **Pattern Matching**: Detects common chat formats (`[timestamp] sender: message`)
- **Flexible**: Handles various text formats with fallback parsing
- **Context**: Maintains conversation flow across lines

## Processing Details

### Thorough Processing
- **Complete Coverage**: Processes ALL files in your data directory
- **No Random Sampling**: Every file is processed completely
- **Conversation Context**: Maintains conversation flow across entries
- **Memory Efficient**: Processes in chunks to handle large datasets

### Progress Tracking
- **Real-time Updates**: Live progress updates through GUI
- **ETA Calculation**: Accurate time estimates based on processing speed
- **Stage Tracking**: Clear indication of current processing stage
- **File-level Progress**: Shows which file is currently being processed

## Output Files

After successful fine-tuning, you'll find:

```
output/
├── parsed_data.json           # Raw parsed data
├── train.jsonl               # Training dataset (Gemma format)
├── train_train.jsonl         # Training split
├── train_val.jsonl          # Validation split
├── adapters/                # LoRA adapters
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── merged_model/            # Merged fine-tuned model
│   ├── config.json
│   ├── tokenizer.json
│   └── model.safetensors
├── finetune_results.json    # Summary of results
└── finetune.log            # Training logs
```

## Model Loading in LM Studio

After fine-tuning, you can load the model in LM Studio:

1. **For MLX models**: Use the `merged_model/` directory
2. **For GGUF models**: Use the converted `.gguf` file

### Loading Instructions
1. Open LM Studio
2. Click "Load Model"
3. Navigate to your output directory
4. Select the merged model or GGUF file
5. The model will load with your personal fine-tuning

## Troubleshooting

### Common Issues

**Out of Memory (OOM)**:
- Reduce `--batch_size` (try 2 or 1)
- Reduce `--max_entries` for testing
- Ensure you have sufficient RAM (128GB recommended)

**Vision API Errors**:
- Ensure LM Studio is running on `http://localhost:1234`
- Verify vision model is loaded in LM Studio
- Check vision model path is correct

**Model Loading Errors**:
- Verify model path exists
- Ensure model is in correct format (GGUF or MLX)
- Check model is the abliterated version

**Data Parsing Issues**:
- Check file formats are supported
- Verify file encoding (UTF-8 recommended)
- Review logs for specific parsing errors

**Frontend Issues**:
- Ensure tkinter is installed: `pip install tk`
- Check Python version compatibility
- Verify all dependencies are installed

### Debug Mode

Enable verbose logging:
```bash
export PYTHONPATH=.
python -u finetune.py --model_path /path/to/model --data_dir /path/to/data 2>&1 | tee training.log
```

## Performance Optimization

### For Large Datasets
- Use `--max_entries` to limit processing for testing
- Process data in chunks (handled automatically)
- Monitor memory usage during training
- Use pause/resume for long processing sessions

### For Faster Training
- Reduce `--lora_rank` (try 8 or 16)
- Use smaller `--batch_size`
- Reduce `--epochs` for quick testing

## Customization

### Adding New Data Formats
1. Extend `DataParser` class in `parse_data.py`
2. Add new file extension to `supported_extensions`
3. Implement parsing method following existing patterns

### Custom Training Prompts
1. Modify `create_instruction_prompt()` in `prepare_dataset.py`
2. Add new prompt types to `prompt_types` list
3. Customize instruction/output generation logic

### Vision Model Integration
1. Update `_call_vision_api()` in `parse_data.py`
2. Modify vision prompt for your specific use case
3. Adjust API parameters as needed

### Frontend Customization
1. Modify `frontend.py` for custom UI elements
2. Add new configuration options
3. Customize progress display format

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for personal use. Please respect the licenses of the underlying models (Gemma, MLX-LM, etc.).

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `output/finetune.log`
3. Ensure all prerequisites are met
4. Test with a small dataset first
5. Use the GUI frontend for easier debugging

## Acknowledgments

- MLX team for the excellent MLX-LM library
- Google for the Gemma models
- LM Studio team for the local inference platform 