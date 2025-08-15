#!/usr/bin/env python3
"""
Parse reasoning traces from JSON file and create separate JSONL files for each mode.
"""

import json
import os
import hashlib
from pathlib import Path


def create_question_hash(question_text, correct_answer):
    """
    Create a consistent hash for a question based on its content.
    This can be used to match questions across different files.

    Args:
        question_text (str): The question text
        correct_answer (str): The correct answer

    Returns:
        str: A hash that uniquely identifies this question
    """
    # Clean and normalize the question text for consistent hashing
    cleaned_question = question_text.strip().replace("\n", " ").replace("  ", " ")
    cleaned_answer = correct_answer.strip()

    # Create hash from question + answer combination
    content = f"{cleaned_question}||{cleaned_answer}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def parse_reasoning_file(input_file_path, output_dir):
    """
    Parse the input JSON file and create separate JSONL files for each reasoning mode.

    Args:
        input_file_path (str): Path to the input JSON file
        output_dir (str): Directory to save the output JSONL files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the input JSON file
    with open(input_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both single object and list of objects
    if isinstance(data, list):
        questions_data = data
    else:
        questions_data = [data]

    # Get the input file path for creating the path field
    input_path = str(input_file_path)

    # Process each question in the data
    for question_data in questions_data:
        # Extract base information
        question_id = question_data.get("question_id")
        question = question_data.get("question", "")
        text = question_data.get("text", "")
        correct_answer = question_data.get("correct_answer", "")

        # Get available reasoning modes from modes_successful
        modes_successful = question_data.get("processing_metadata", {}).get(
            "modes_successful", []
        )

        # Get reasoning traces and predictions
        reasoning_traces = question_data.get("reasoning_traces", {})
        predictions = question_data.get("predictions", {})

        # Create a unique hash for this question to enable tracking
        question_hash = create_question_hash(question, correct_answer)

        # Process each successful mode
        for mode in modes_successful:
            if mode in reasoning_traces:
                # Create the path field by concatenating input path and question_id
                path_field = f"{input_path}#{question_id}"

                # Get reasoning trace content for this mode
                reasoning_content = reasoning_traces[mode]

                # Concatenate question, text, and reasoning_traces content
                combined_text = f"Question: {question}\n\nText: {text}\n\nReasoning Trace ({mode}):\n"

                # Add reasoning trace content as a formatted string, excluding summary fields
                if isinstance(reasoning_content, dict):
                    for key, value in reasoning_content.items():
                        # Skip reasoning_summary and reasoning_mode fields
                        if key not in ["reasoning_summary", "reasoning_mode"]:
                            combined_text += f"\n{key}: {json.dumps(value, indent=2)}"
                else:
                    combined_text += f"\n{json.dumps(reasoning_content, indent=2)}"

                # Create metadata
                metadata = {
                    "correct_answer": correct_answer,
                    "prediction_data": predictions.get(mode, {}),
                    "question_hash": question_hash,
                    "original_question_id": question_id,
                    "reasoning_mode": mode,
                }

                # Create the output record
                output_record = {
                    "path": path_field,
                    "text": combined_text,
                    "metadata": metadata,
                }

                # Write to the appropriate file
                output_file = os.path.join(output_dir, f"{mode}.jsonl")
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(output_record) + "\n")

                print(f"Added record to {output_file}")


def main():
    """Main function to parse the reasoning traces file."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse reasoning traces from JSON file"
    )
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument(
        "--output_dir",
        default="./parsed_output",
        help="Output directory for JSONL files (default: ./parsed_output)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return 1

    try:
        parse_reasoning_file(args.input_file, args.output_dir)
        print(f"Successfully parsed {args.input_file}")
        print(f"Output files saved to {args.output_dir}")
    except Exception as e:
        print(f"Error parsing file: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
