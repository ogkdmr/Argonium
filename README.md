# Reasoning Trace Parsing and RAG Evaluation Workflow

This repository contains tools for parsing reasoning traces from JSON files and creating a system to evaluate RAG (Retrieval Augmented Generation) retrieval accuracy.

## Overview

The workflow allows you to:
1. Parse reasoning traces from complex JSON files into separate JSONL files by reasoning mode
2. Add unique identifiers (hashes) to both questions and reasoning traces
3. Evaluate whether your RAG system retrieves the correct reasoning traces for given questions

## Files

- `parse_reasoning_traces.py` - Main script to parse reasoning traces into JSONL format
- `add_question_hashes.py` - Tool to add hashes to questions and verify retrieval accuracy
- `README.md` - This documentation

## Quick Start

### Step 1: Add Hashes to Your MCQ File

First, add unique identifiers to your multiple choice question file:

```bash
python3 add_question_hashes.py add_hashes \
  --questions_file your_mcq_file.json \
  --output_file mcq_with_hashes.json
```

**Input Format** (MCQ file):
```json
[
  {
    "question": "Your question text here...",
    "answer": "The correct answer is...",
    "text": "Supporting text content...",
    "type": "multiple-choice"
  }
]
```

**Output**: Same file with added `question_hash` field for each question.

### Step 2: Parse Reasoning Traces

Convert your reasoning traces JSON file into separate JSONL files by mode:

```bash
python3 parse_reasoning_traces.py \
  /path/to/reasoning_traces.json \
  --output_dir ./parsed_traces
```

**Input Format** (Reasoning traces file):
```json
[
  {
    "question_id": 0,
    "question": "Question text...",
    "text": "Supporting text...",
    "correct_answer": "Answer...",
    "reasoning_traces": {
      "focused": { ... },
      "detailed": { ... },
      "efficient": { ... }
    },
    "predictions": {
      "focused": { ... },
      "detailed": { ... },
      "efficient": { ... }
    },
    "processing_metadata": {
      "modes_successful": ["focused", "detailed", "efficient"]
    }
  }
]
```

**Output**: Creates separate JSONL files:
- `focused.jsonl`
- `detailed.jsonl` 
- `efficient.jsonl`

Each line contains:
```json
{
  "path": "/path/to/file.json#question_id",
  "text": "Combined question, text, and reasoning trace content",
  "metadata": {
    "correct_answer": "...",
    "prediction_data": { ... },
    "question_hash": "unique_hash_here",
    "original_question_id": 0,
    "reasoning_mode": "focused"
  }
}
```

### Step 3: Verify Matching

Check that questions and reasoning traces can be properly matched:

```bash
python3 add_question_hashes.py verify \
  --questions_file mcq_with_hashes.json \
  --reasoning_dir ./parsed_traces
```

**Output Example**:
```
Loaded 50 questions with hashes
focused.jsonl: 45/50 reasoning traces can be matched to questions
detailed.jsonl: 48/50 reasoning traces can be matched to questions
efficient.jsonl: 47/50 reasoning traces can be matched to questions
```

## Evaluating RAG Retrieval Accuracy

### How It Works

1. **Question Hash**: Each question gets a unique hash based on question text + correct answer
2. **Trace Hash**: Each reasoning trace gets the same hash in its metadata
3. **Matching**: When your RAG system retrieves a reasoning trace, compare the hashes

### In Your RAG Evaluation Code

```python
import json

def evaluate_retrieval_accuracy(question, retrieved_traces):
    """
    Check if retrieved reasoning traces match the question.
    
    Args:
        question (dict): Question with question_hash field
        retrieved_traces (list): List of retrieved reasoning trace records
    
    Returns:
        dict: Accuracy metrics
    """
    question_hash = question.get("question_hash")
    
    correct_retrievals = 0
    for trace in retrieved_traces:
        trace_hash = trace.get("metadata", {}).get("question_hash")
        if trace_hash == question_hash:
            correct_retrievals += 1
    
    return {
        "total_retrieved": len(retrieved_traces),
        "correct_retrievals": correct_retrievals,
        "accuracy": correct_retrievals / len(retrieved_traces) if retrieved_traces else 0
    }
```

## Advanced Usage

### Parsing Single Files

Both scripts can handle single objects or arrays:

```bash
# Works with both single question and question arrays
python3 add_question_hashes.py add_hashes --questions_file single_question.json --output_file output.json

# Works with both single reasoning trace and arrays
python3 parse_reasoning_traces.py single_trace.json --output_dir ./output
```

### Custom Output Directory

```bash
python3 parse_reasoning_traces.py input.json --output_dir /custom/path/
```

### Appending to Existing Files

The parsing script appends to existing JSONL files, so you can process multiple input files:

```bash
python3 parse_reasoning_traces.py file1.json --output_dir ./traces
python3 parse_reasoning_traces.py file2.json --output_dir ./traces
# Both will be combined in the same output files
```

## Key Features

### Reasoning Trace Filtering

The parser excludes certain fields from the text output to prevent "cheating":
- `reasoning_summary` - Excluded to prevent RAG model from seeing conclusions
- `reasoning_mode` - Excluded to prevent mode identification shortcuts

### Hash Generation

Hashes are created using:
- SHA256 of `cleaned_question + "||" + correct_answer`
- 16-character hash for easy tracking
- Consistent across question and reasoning trace files

### Metadata Tracking

Each reasoning trace includes comprehensive metadata:
- `question_hash` - For matching questions
- `original_question_id` - Original ID from source
- `reasoning_mode` - Which reasoning approach (focused/detailed/efficient)
- `correct_answer` - The correct answer
- `prediction_data` - Mode-specific prediction information

## Troubleshooting

### Common Issues

1. **"'list' object has no attribute 'get'"**: Your JSON file contains an array, this is now handled automatically

2. **Hash mismatches**: Ensure both files use the exact same question text and answer format

3. **Missing modes**: Check that `modes_successful` in your reasoning file matches available reasoning traces

4. **File not found**: Verify file paths are correct and files exist

### Getting Help

Run scripts with `-h` for help:
```bash
python3 parse_reasoning_traces.py -h
python3 add_question_hashes.py -h
```

## Example Complete Workflow

```bash
# 1. Add hashes to questions
python3 add_question_hashes.py add_hashes \
  --questions_file /path/to/questions.json \
  --output_file questions_with_hashes.json

# 2. Parse reasoning traces  
python3 parse_reasoning_traces.py \
  /path/to/reasoning_traces.json \
  --output_dir ./parsed_output

# 3. Verify everything matches
python3 add_question_hashes.py verify \
  --questions_file questions_with_hashes.json \
  --reasoning_dir ./parsed_output

# 4. Use in your RAG evaluation pipeline
# (Your RAG system retrieves from parsed_output/*.jsonl files)
# (Your evaluation compares question_hash values)
```

This workflow enables robust evaluation of whether your RAG system retrieves the correct reasoning traces for each question.