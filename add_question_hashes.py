#!/usr/bin/env python3
"""
Add question hashes to a question file to enable tracking of retrieved reasoning traces.
"""

import json
import hashlib
import argparse
import os


def create_question_hash(question_text, correct_answer):
    """
    Create a consistent hash for a question based on its content.
    This matches the function in parse_reasoning_traces.py

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


def add_hashes_to_questions(input_file, output_file):
    """
    Add question hashes to a question file.

    Args:
        input_file (str): Path to input JSON file with questions
        output_file (str): Path to output JSON file with hashes added
    """
    with open(input_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Handle both single question and list of questions
    if isinstance(questions, dict):
        questions = [questions]

    # Add hashes to each question
    for question_data in questions:
        question_text = question_data.get("question", "")
        answer_text = question_data.get("answer", "")

        # Create the hash
        question_hash = create_question_hash(question_text, answer_text)
        question_data["question_hash"] = question_hash

        print(f"Added hash {question_hash} for question: {question_text[:100]}...")

    # Save the updated questions
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(questions)} questions with hashes to {output_file}")


def verify_retrieval_accuracy(questions_file, reasoning_traces_dir):
    """
    Verify if the correct reasoning traces can be retrieved for questions.

    Args:
        questions_file (str): Path to questions file with hashes
        reasoning_traces_dir (str): Directory containing reasoning trace JSONL files
    """
    # Load questions
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if isinstance(questions, dict):
        questions = [questions]

    # Create a mapping of question hashes to questions
    question_hash_map = {
        q.get("question_hash"): q for q in questions if q.get("question_hash")
    }

    print(f"Loaded {len(question_hash_map)} questions with hashes")

    # Check each reasoning mode file
    reasoning_files = ["focused.jsonl", "detailed.jsonl", "efficient.jsonl"]

    for mode_file in reasoning_files:
        file_path = os.path.join(reasoning_traces_dir, mode_file)
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found")
            continue

        matched = 0
        total = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                reasoning_trace = json.loads(line.strip())
                total += 1

                trace_hash = reasoning_trace.get("metadata", {}).get("question_hash")
                if trace_hash in question_hash_map:
                    matched += 1
                else:
                    print(f"No match found for hash: {trace_hash}")

        print(
            f"{mode_file}: {matched}/{total} reasoning traces can be matched to questions"
        )

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Add question hashes and verify retrieval accuracy"
    )
    parser.add_argument(
        "command", choices=["add_hashes", "verify"], help="Command to execute"
    )
    parser.add_argument(
        "--questions_file", required=True, help="Path to questions JSON file"
    )
    parser.add_argument("--output_file", help="Output file for add_hashes command")
    parser.add_argument(
        "--reasoning_dir",
        help="Directory with reasoning trace JSONL files for verify command",
    )

    args = parser.parse_args()

    if args.command == "add_hashes":
        if not args.output_file:
            print("Error: --output_file required for add_hashes command")
            return 1
        add_hashes_to_questions(args.questions_file, args.output_file)

    elif args.command == "verify":
        if not args.reasoning_dir:
            print("Error: --reasoning_dir required for verify command")
            return 1
        verify_retrieval_accuracy(args.questions_file, args.reasoning_dir)

    return 0


if __name__ == "__main__":
    exit(main())
