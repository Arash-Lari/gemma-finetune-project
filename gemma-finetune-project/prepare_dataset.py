import json
import jsonlines
import random
from typing import List, Dict, Any
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetPreparer:
    def __init__(self):
        """
        Initialize the dataset preparer for Gemma fine-tuning.
        """
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.start_of_turn = "<start_of_turn>"
        self.end_of_turn = "<end_of_turn>"
        
    def create_instruction_prompt(self, entry: Dict[str, Any], context_entries: List[Dict[str, Any]] = None) -> str:
        """
        Create an instruction prompt for fine-tuning based on the entry.
        
        Args:
            entry: Parsed data entry
            context_entries: Optional context entries for conversation history
            
        Returns:
            Formatted instruction prompt
        """
        speaker = entry.get('speaker', 'Unknown')
        content = entry.get('content', '')
        timestamp = entry.get('timestamp', '')
        
        # Build context if available
        context = ""
        if context_entries:
            context_lines = []
            for ctx_entry in context_entries[-5:]:  # Last 5 entries for context
                ctx_speaker = ctx_entry.get('speaker', 'Unknown')
                ctx_content = ctx_entry.get('content', '')
                context_lines.append(f"{ctx_speaker}: {ctx_content}")
            context = "\n".join(context_lines) + "\n"
        
        # Create different types of prompts
        prompt_types = [
            {
                "instruction": f"Emulate {speaker}'s communication style based on this message: {content}",
                "output": f"Here's how {speaker} would respond: {content}"
            },
            {
                "instruction": f"Analyze this message from {speaker}: {content}",
                "output": f"Analysis: This message shows {speaker}'s communication style and intent."
            },
            {
                "instruction": f"Continue the conversation as {speaker}: {context}{speaker}: {content}",
                "output": f"{speaker}: {content}"
            },
            {
                "instruction": f"Summarize this message from {speaker}: {content}",
                "output": f"Summary: {content[:100]}..." if len(content) > 100 else f"Summary: {content}"
            }
        ]
        
        # Randomly select a prompt type
        selected_prompt = random.choice(prompt_types)
        
        return selected_prompt["instruction"], selected_prompt["output"]
    
    def format_gemma_instruction(self, instruction: str, output: str) -> str:
        """
        Format the instruction and output in Gemma's instruction format.
        
        Args:
            instruction: The instruction text
            output: The expected output text
            
        Returns:
            Formatted string in Gemma instruction format
        """
        formatted = f"{self.bos_token}{self.start_of_turn}user\n{instruction}{self.end_of_turn}\n{self.start_of_turn}model\n{output}{self.end_of_turn}\n{self.eos_token}"
        return formatted
    
    def prepare_training_data(self, parsed_data: List[Dict[str, Any]], output_path: str, 
                            max_entries: int = None, conversation_window: int = 5) -> None:
        """
        Prepare training data in Gemma instruction format.
        
        Args:
            parsed_data: List of parsed data entries
            output_path: Path to save the JSONL file
            max_entries: Maximum number of entries to process (None for all)
            conversation_window: Number of previous entries to use as context
        """
        logger.info(f"Preparing training data from {len(parsed_data)} entries")
        
        if max_entries:
            parsed_data = parsed_data[:max_entries]
        
        training_examples = []
        
        # Group entries by source file to maintain conversation context
        entries_by_file = {}
        for entry in parsed_data:
            source_file = entry.get('source_file', 'unknown')
            if source_file not in entries_by_file:
                entries_by_file[source_file] = []
            entries_by_file[source_file].append(entry)
        
        # Process each file's entries
        for source_file, entries in tqdm(entries_by_file.items(), desc="Processing files"):
            # Sort entries by line/row number if available
            entries.sort(key=lambda x: x.get('line_number', x.get('row_number', 0)))
            
            for i, entry in enumerate(entries):
                # Get context from previous entries in the same file
                context_entries = []
                if i > 0:
                    start_idx = max(0, i - conversation_window)
                    context_entries = entries[start_idx:i]
                
                # Create instruction and output
                instruction, output = self.create_instruction_prompt(entry, context_entries)
                
                # Format for Gemma
                formatted_example = self.format_gemma_instruction(instruction, output)
                
                training_examples.append({
                    "text": formatted_example,
                    "source_file": source_file,
                    "entry_index": i,
                    "speaker": entry.get('speaker', 'Unknown'),
                    "original_content": entry.get('content', '')
                })
        
        # Save to JSONL file
        logger.info(f"Saving {len(training_examples)} training examples to {output_path}")
        
        with jsonlines.open(output_path, mode='w') as writer:
            for example in training_examples:
                writer.write(example)
        
        logger.info(f"Successfully saved training data to {output_path}")
        
        # Print some statistics
        speakers = set(ex['speaker'] for ex in training_examples)
        logger.info(f"Training data statistics:")
        logger.info(f"  Total examples: {len(training_examples)}")
        logger.info(f"  Unique speakers: {len(speakers)}")
        logger.info(f"  Speakers: {list(speakers)[:10]}...")  # Show first 10 speakers
    
    def create_validation_split(self, training_data_path: str, validation_ratio: float = 0.1) -> None:
        """
        Create a validation split from the training data.
        
        Args:
            training_data_path: Path to the training data JSONL file
            validation_ratio: Ratio of data to use for validation
        """
        logger.info(f"Creating validation split with ratio {validation_ratio}")
        
        # Read training data
        training_examples = []
        with jsonlines.open(training_data_path, mode='r') as reader:
            for example in reader:
                training_examples.append(example)
        
        # Shuffle and split
        random.shuffle(training_examples)
        split_idx = int(len(training_examples) * (1 - validation_ratio))
        
        train_examples = training_examples[:split_idx]
        val_examples = training_examples[split_idx:]
        
        # Save splits
        train_path = training_data_path.replace('.jsonl', '_train.jsonl')
        val_path = training_data_path.replace('.jsonl', '_val.jsonl')
        
        with jsonlines.open(train_path, mode='w') as writer:
            for example in train_examples:
                writer.write(example)
        
        with jsonlines.open(val_path, mode='w') as writer:
            for example in val_examples:
                writer.write(example)
        
        logger.info(f"Created train split: {len(train_examples)} examples -> {train_path}")
        logger.info(f"Created validation split: {len(val_examples)} examples -> {val_path}")
    
    def analyze_dataset(self, jsonl_path: str) -> Dict[str, Any]:
        """
        Analyze the prepared dataset.
        
        Args:
            jsonl_path: Path to the JSONL file
            
        Returns:
            Dictionary with dataset statistics
        """
        logger.info(f"Analyzing dataset: {jsonl_path}")
        
        examples = []
        with jsonlines.open(jsonl_path, mode='r') as reader:
            for example in reader:
                examples.append(example)
        
        # Calculate statistics
        total_length = sum(len(ex['text']) for ex in examples)
        avg_length = total_length / len(examples) if examples else 0
        
        speakers = set(ex['speaker'] for ex in examples)
        source_files = set(ex['source_file'] for ex in examples)
        
        stats = {
            'total_examples': len(examples),
            'total_characters': total_length,
            'avg_example_length': avg_length,
            'unique_speakers': len(speakers),
            'unique_source_files': len(source_files),
            'speakers': list(speakers),
            'source_files': list(source_files)
        }
        
        logger.info("Dataset statistics:")
        for key, value in stats.items():
            if key not in ['speakers', 'source_files']:
                logger.info(f"  {key}: {value}")
        
        return stats

def main():
    """Test the dataset preparer."""
    preparer = DatasetPreparer()
    
    # Load sample parsed data
    try:
        with open("output/parsed_data.json", 'r', encoding='utf-8') as f:
            parsed_data = json.load(f)
        
        # Prepare training data
        output_path = "output/train.jsonl"
        preparer.prepare_training_data(parsed_data, output_path, max_entries=1000)
        
        # Create validation split
        preparer.create_validation_split(output_path)
        
        # Analyze dataset
        preparer.analyze_dataset(output_path)
        
    except FileNotFoundError:
        logger.error("No parsed data found. Run parse_data.py first.")

if __name__ == "__main__":
    main() 