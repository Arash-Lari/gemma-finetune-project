#!/bin/bash

# Activation script for Gemma Fine-tuning Project

echo "🚀 Activating Gemma Fine-tuning Project Environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/lib/python3.13/site-packages/mlx" ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
fi

echo "✅ Environment ready!"
echo ""
echo "Available commands:"
echo "  python frontend.py     - Launch the GUI frontend"
echo "  python test_setup.py   - Test the setup"
echo "  python finetune.py     - Run fine-tuning (command line)"
echo "  python parse_data.py   - Parse data only"
echo "  python prepare_dataset.py - Prepare dataset only"
echo ""
echo "To deactivate: deactivate"
echo ""
