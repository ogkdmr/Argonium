#!/usr/bin/env python3
"""
mcqa_generation_distllm_v2.py - Multiple Choice Question Generation from Hugging Face Dataset (Version 2)

This script generates high-quality multiple-choice questions from a Hugging Face dataset
containing pre-processed text chunks. It bypasses PDF parsing and text chunking,
working directly with the provided dataset.

NEW in V2:
- Content relevance checking before question generation
- Enhanced quality evaluation with cleaning metadata
- Structured cleaning_metadata output with relevance_check and quality_check
- Timestamp and version tracking for data lineage
- Direct path inclusion in output for improved traceability

Usage:
    python mcqa_generation_distllm_v2.py <dataset_path> --output <output.json> --model <model_name>

Dataset Requirements:
    - "text" column: Text chunks for question generation
    - "embedding" column: Embeddings (not used, but can be present)
    - "path" column: Source file path for traceability

Features:
    - Parallel processing with configurable workers
    - Checkpoint system for resumable processing
    - Terminal UI with progress tracking
    - Comprehensive error handling and logging
    - Content relevance filtering and quality evaluation
    - Full traceability from questions to source files via chunk_id and path
    - Enhanced cleaning metadata for data quality assessment
    - Direct path inclusion eliminates need for chunk_id -> path lookups
"""

import os
import sys
import json
import re
import time
import random
import hashlib
import curses
import threading
import queue
import atexit
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Dict, Any, Tuple, Optional, Set
from enum import Enum
import argparse
import signal
from tqdm import tqdm

# Third-party imports
import yaml
import datasets
from openai import OpenAI

# Global variables for state tracking
_total_chunks = 0
_processed_chunks = 0
_exit_requested = False
_log_queue = queue.Queue()
_use_split_screen = True
_start_time = time.time()
_active_workers = 0
_max_workers = None
_result_queue = None
_worker_pool = None
_counter_lock = threading.Lock()
_error_log_file = None
_max_error_threshold = 200
_openai_client = None


class QuestionType(Enum):
    MULTIPLE_CHOICE = "mc"
    FREE_FORM = "qa"

    def __str__(self):
        return self.value


class TerminalUI:
    """
    Class to manage a split-screen terminal UI with curses.
    The top pane shows global progress statistics and the bottom pane shows scrolling logs.
    """

    def __init__(self):
        self.screen = None
        self.top_pane = None
        self.bottom_pane = None
        self.top_height = 20
        self.max_log_lines = 1000
        self.log_lines = []
        self.log_position = 0
        self.running = False
        self.ui_thread = None
        self.curses_enabled = False

        # Current statistics to display
        self.stats = {
            "chunks_processed": 0,
            "total_chunks": 0,
            "questions_generated": 0,
            "success_rate": 0.0,
            "completion_percentage": 0.0,
            "elapsed_time": 0,
            "eta": "Unknown",
            "avg_chunk_time": "Unknown",
            "current_chunk": "",
            "status_message": "Initializing...",
            "active_workers": 0,
            "max_workers": 0,
            # Error counters
            "error_chunk_reading": 0,
            "error_summarizing": 0,
            "error_question_gen": 0,
            "error_question_eval": 0,
            "error_api": 0,
            "error_other": 0,
            "low_score_questions": 0,
            "filtered_non_relevant": 0,
            "total_errors": 0,
        }

    def start(self):
        """Start the terminal UI in a separate thread."""
        if not _use_split_screen:
            return

        self.running = True
        self.ui_thread = threading.Thread(target=self._run_ui, daemon=True)
        self.ui_thread.start()

    def stop(self):
        """Stop the terminal UI."""
        self.running = False
        if self.ui_thread and self.ui_thread.is_alive():
            self.ui_thread.join(timeout=1)

    def _run_ui(self):
        """Main UI loop running in separate thread."""
        try:
            curses.wrapper(self._curses_main)
            self.curses_enabled = True
        except Exception as e:
            log_message(f"Terminal UI failed to start: {e}", log_level="WARNING")
            self.curses_enabled = False

    def _curses_main(self, stdscr):
        """Main curses interface."""
        self.screen = stdscr
        self.screen.nodelay(True)
        curses.curs_set(0)

        # Initialize color pairs if available
        if curses.has_colors():
            curses.start_colors()
            curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
            curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
            curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
            curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)

        while self.running:
            try:
                self._update_display()
                time.sleep(0.1)
            except Exception as e:
                break

    def _update_display(self):
        """Update the terminal display."""
        if not self.screen:
            return

        try:
            height, width = self.screen.getmaxyx()
            self.screen.clear()

            # Update top pane
            self._update_stats_pane(height, width)

            # Update log pane
            self._update_log_pane(height, width)

            self.screen.refresh()
        except Exception:
            pass

    def _update_stats_pane(self, height, width):
        """Update the statistics pane."""
        try:
            # Calculate current statistics
            elapsed_time = time.time() - _start_time
            self.stats["elapsed_time"] = elapsed_time

            # Calculate ETA and average time
            if self.stats["chunks_processed"] > 0:
                avg_time = elapsed_time / self.stats["chunks_processed"]
                remaining_chunks = max(
                    0, self.stats["total_chunks"] - self.stats["chunks_processed"]
                )
                eta_seconds = avg_time * remaining_chunks
                self.stats["avg_chunk_time"] = f"{avg_time:.2f}s"
                self.stats["eta"] = self._format_time(eta_seconds)
            else:
                self.stats["avg_chunk_time"] = "N/A"
                self.stats["eta"] = "Unknown"

            # Update completion percentage
            if self.stats["total_chunks"] > 0:
                self.stats["completion_percentage"] = (
                    self.stats["chunks_processed"] / self.stats["total_chunks"]
                ) * 100
            else:
                self.stats["completion_percentage"] = 0

            # Header
            title = "MCQA Generation v2 - HF Dataset Processing"
            self.screen.addstr(0, 0, title.center(width)[:width])
            self.screen.addstr(1, 0, "=" * min(width, len(title)))

            # Progress information
            row = 3
            self.screen.addstr(
                row, 0, f"Chunks Processed: {self.stats['chunks_processed']}"
            )
            self.screen.addstr(row, 25, f"Total Chunks: {self.stats['total_chunks']}")
            self.screen.addstr(
                row, 45, f"Completion: {self.stats['completion_percentage']:.1f}%"
            )

            row += 1
            self.screen.addstr(
                row, 0, f"Questions Generated: {self.stats['questions_generated']}"
            )
            self.screen.addstr(
                row, 25, f"Success Rate: {self.stats['success_rate']:.1f}%"
            )

            row += 1
            self.screen.addstr(
                row, 0, f"Elapsed Time: {self._format_time(elapsed_time)}"
            )
            self.screen.addstr(row, 25, f"ETA: {self.stats['eta']}")
            self.screen.addstr(
                row, 45, f"Avg Time/Chunk: {self.stats['avg_chunk_time']}"
            )

            row += 1
            self.screen.addstr(
                row, 0, f"Active Workers: {self.stats['active_workers']}"
            )
            self.screen.addstr(row, 25, f"Max Workers: {self.stats['max_workers']}")

            # Progress bar
            row += 2
            progress_width = min(60, width - 10)
            if self.stats["total_chunks"] > 0:
                progress = self.stats["chunks_processed"] / self.stats["total_chunks"]
                filled = int(progress * progress_width)
                bar = "█" * filled + "░" * (progress_width - filled)
                self.screen.addstr(row, 0, f"Progress: [{bar}]")

            # Current status
            row += 2
            status = self.stats["status_message"][:width]
            self.screen.addstr(row, 0, f"Status: {status}")

            # Current chunk
            row += 1
            current_chunk = self.stats["current_chunk"][: width - 20]
            self.screen.addstr(row, 0, f"Current Chunk: {current_chunk}")

            # Error statistics
            row += 2
            self.screen.addstr(row, 0, "ERROR STATISTICS:")
            row += 1
            self.screen.addstr(
                row, 0, f"  Chunk Reading: {self.stats['error_chunk_reading']}"
            )
            self.screen.addstr(
                row, 25, f"  Summarizing: {self.stats['error_summarizing']}"
            )
            row += 1
            self.screen.addstr(
                row, 0, f"  Question Gen: {self.stats['error_question_gen']}"
            )
            self.screen.addstr(
                row, 25, f"  Question Eval: {self.stats['error_question_eval']}"
            )
            row += 1
            self.screen.addstr(row, 0, f"  API Errors: {self.stats['error_api']}")
            self.screen.addstr(row, 25, f"  Other: {self.stats['error_other']}")
            row += 1
            self.screen.addstr(
                row, 0, f"  Low Score: {self.stats['low_score_questions']}"
            )
            self.screen.addstr(
                row, 25, f"  Non-Relevant: {self.stats['filtered_non_relevant']}"
            )
            row += 1
            self.screen.addstr(row, 0, f"  Total Errors: {self.stats['total_errors']}")

        except Exception:
            pass

    def _update_log_pane(self, height, width):
        """Update the log display pane."""
        try:
            log_start_row = self.top_height
            log_height = height - log_start_row - 1

            if log_height <= 0:
                return

            # Process any new log messages
            self._process_logs()

            # Display recent log lines
            self.screen.addstr(log_start_row, 0, "LOGS:")
            self.screen.addstr(log_start_row + 1, 0, "-" * min(width, 60))

            visible_logs = self.log_lines[-log_height + 2 :]
            for i, log_line in enumerate(visible_logs):
                row = log_start_row + 2 + i
                if row < height - 1:
                    display_line = log_line[:width]
                    self.screen.addstr(row, 0, display_line)

        except Exception:
            pass

    def _process_logs(self):
        """Process log messages from the queue."""
        try:
            while not _log_queue.empty():
                try:
                    log_message = _log_queue.get_nowait()
                    self.log_lines.append(log_message)
                    if len(self.log_lines) > self.max_log_lines:
                        self.log_lines = self.log_lines[-self.max_log_lines :]
                except queue.Empty:
                    break
        except Exception:
            pass

    def _format_time(self, seconds):
        """Format time in a human-readable way."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"

    def update_stats(self, **kwargs):
        """Update statistics values."""
        for key, value in kwargs.items():
            if key in self.stats:
                self.stats[key] = value

    def log(self, message):
        """Add a log message."""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        _log_queue.put(formatted_message)


def init_terminal_ui():
    """Initialize the terminal UI."""
    global terminal_ui
    if _use_split_screen:
        terminal_ui = TerminalUI()
        terminal_ui.start()
        return terminal_ui
    return None


def cleanup_ui():
    """Clean up the terminal UI."""
    global terminal_ui
    if "terminal_ui" in globals() and terminal_ui:
        terminal_ui.stop()


def batched_openai_completion(model: str, messages: list, **kwargs):
    """
    Call OpenAI API with the given parameters.
    This function handles the API call and returns the response.
    """
    global _openai_client

    if _openai_client is None:
        raise RuntimeError("OpenAI client not initialized")

    try:
        response = _openai_client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        return {
            "choices": [{"message": {"content": response.choices[0].message.content}}]
        }
    except Exception as e:
        log_message(f"OpenAI API error: {e}", log_level="ERROR", error_type="api")
        raise


def clean_answer_content(answer_text: str) -> str:
    """
    Clean up answer content to remove evaluation commentary and keep only technical content.

    Args:
        answer_text: The raw answer text that may contain evaluation comments

    Returns:
        str: Cleaned answer text with only technical content
    """
    # Remove common evaluation phrases that should not be in technical answers
    evaluation_patterns = [
        r"The question is clear[^.]*\.",
        r"The question is well-structured[^.]*\.",
        r"The stem is unambiguous[^.]*\.",
        r"The answer choices are plausible[^.]*\.",
        r"The question is factually accurate[^.]*\.",
        r"The difficulty is appropriate[^.]*\.",
        r"The distractors are well-chosen[^.]*\.",
        r"The educational value is high[^.]*\.",
        r"The only minor deduction[^.]*\.",
        r"The content block[^.]*\.",
        r"This question tests[^.]*\.",
        r"The question requires[^.]*\.",
        r"This is a good question[^.]*\.",
        r"This question is appropriate[^.]*\.",
        r"The provided context[^.]*\.",
        r"Overall[,\s]*this is[^.]*\.",
        r"In summary[^.]*\.",
        r"The answer provided[^.]*\.",
        r"This answer[^.]*demonstrates[^.]*\.",
        r"The response shows[^.]*\.",
        r"This explanation[^.]*\.",
        r"The text states[^.]*\.",
        r"According to the passage[^.]*\.",
        r"The passage mentions[^.]*\.",
        r"As described in[^.]*\.",
        r"The document indicates[^.]*\.",
        r"This demonstrates[^.]*understanding[^.]*\.",
        r"requiring careful reading[^.]*\.",
        r"potential[ly]* overwhelm[^.]*\.",
    ]

    cleaned_text = answer_text

    # Remove evaluation patterns
    for pattern in evaluation_patterns:
        cleaned_text = re.sub(
            pattern, "", cleaned_text, flags=re.IGNORECASE | re.DOTALL
        )

    # Remove any remaining references to letter labels (A), (B), etc.
    cleaned_text = re.sub(r"\([A-Z]\)", "", cleaned_text)

    # Clean up extra whitespace
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text


def clean_answer_choices(choices: List[str]) -> List[str]:
    """
    Clean up answer choices to remove evaluation content.

    Args:
        choices: List of raw answer choices

    Returns:
        List[str]: Cleaned answer choices
    """
    cleaned_choices = []
    for choice in choices:
        cleaned = clean_answer_content(choice)
        if cleaned:  # Only keep non-empty choices
            cleaned_choices.append(cleaned)

    return cleaned_choices


def log_message(message, log_level="INFO", error_type=None):
    """
    Log a message to the console and/or file.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] [{log_level}] {message}"

    # Print to console if not using split screen
    if not _use_split_screen:
        print(formatted_message)
    else:
        # Add to queue for terminal UI
        _log_queue.put(formatted_message)

    # Write to error log file if it's an error and file is configured
    if log_level in ["ERROR", "WARNING"] and _error_log_file:
        try:
            with open(_error_log_file, "a", encoding="utf-8") as f:
                f.write(formatted_message + "\n")
        except Exception:
            pass  # Don't let logging errors crash the program

    # Update error counters for terminal UI
    if "terminal_ui" in globals() and terminal_ui and error_type:
        if error_type in terminal_ui.stats:
            terminal_ui.stats[error_type] += 1
            terminal_ui.stats["total_errors"] += 1


def check_content_relevance(chunk_text: str, model_name: str) -> Dict:
    """
    Check if the chunk content is relevant to the paper's core content.
    Returns relevance score and reasoning.
    """
    system_message = (
        "You are an expert content evaluator who determines if text content is relevant "
        "to the core scientific/technical content of a paper versus non-relevant material "
        "like copyright notices, licensing information, references, acknowledgments, or metadata."
    )

    user_message = (
        f"Evaluate the following text chunk and determine if it contains core scientific/technical content "
        f"that would be appropriate for generating educational questions.\n\n"
        f"TEXT CHUNK:\n{chunk_text}\n\n"
        f"EVALUATION CRITERIA:\n"
        f"- CORE CONTENT (High relevance): Scientific concepts, research findings, technical explanations, "
        f"methodology, data analysis, theories, experimental results, clinical information, etc.\n"
        f"- NON-CORE CONTENT (Low relevance): Copyright notices, licensing text, reference lists, "
        f"acknowledgments, author information, publication metadata, figure/table captions only, "
        f"page headers/footers, disclaimers, etc.\n\n"
        f"SCORING:\n"
        f"- Score 8-10: Rich core content ideal for question generation\n"
        f"- Score 5-7: Some core content but mixed with non-relevant material\n"
        f"- Score 1-4: Primarily non-relevant content (references, metadata, etc.)\n\n"
        f"Provide your response in this format:\n"
        f"RELEVANCE_SCORE: <numeric score between 1-10>\n"
        f"REASONING: <brief explanation of why this content is or isn't relevant for question generation>\n"
        f"CONTENT_TYPE: <primary type of content: 'core_scientific', 'mixed', 'references', 'metadata', 'copyright', etc.>\n"
    )

    try:
        response = batched_openai_completion(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
        )

        output = response["choices"][0]["message"]["content"].strip()

        # Extract relevance score
        score_match = re.search(r"RELEVANCE_SCORE:\s*(\d+(?:\.\d+)?)", output)
        relevance_score = int(float(score_match.group(1))) if score_match else 5

        # Extract reasoning
        reasoning_match = re.search(r"REASONING:\s*(.*?)(?:\n|$)", output, re.DOTALL)
        reasoning = (
            reasoning_match.group(1).strip()
            if reasoning_match
            else "No reasoning provided"
        )

        # Extract content type
        content_type_match = re.search(r"CONTENT_TYPE:\s*(.*?)(?:\n|$)", output)
        content_type = (
            content_type_match.group(1).strip() if content_type_match else "unknown"
        )

        return {
            "relevance_score": relevance_score,
            "reasoning": reasoning,
            "content_type": content_type,
            "is_relevant": relevance_score >= 6,  # Threshold for relevance
            "raw_output": output,
        }

    except Exception as e:
        log_message(
            f"Error checking content relevance: {e}",
            log_level="ERROR",
            error_type="error_other",
        )
        return {
            "relevance_score": 5,  # Default to medium relevance on error
            "reasoning": f"Error during relevance check: {str(e)}",
            "content_type": "unknown",
            "is_relevant": True,  # Default to relevant on error to avoid losing content
            "raw_output": "",
        }


def generate_multiple_choice_qa_pairs(
    chunk_id: str,
    chunk_text: str,
    model_name: str,
    path: str,
    num_answers: int = 7,
    min_score: int = 7,
) -> Dict:
    """
    Generate a multiple-choice Q/A pair from a chunk.
    Returns a dictionary with question, answer, and other metadata.
    """
    global _exit_requested

    # Start timing the processing
    start_time = time.time()

    # Check for exit request
    if _exit_requested:
        return {
            "chunk_id": chunk_id,
            "path": path,
            "status": "cancelled",
            "processing_time": 0,
            "message": "Cancelled due to shutdown request",
        }

    # --------------------------------------------------------------------
    # Step 0: Check content relevance
    # --------------------------------------------------------------------
    relevance_check = check_content_relevance(chunk_text, model_name)

    # Skip non-relevant content
    if not relevance_check["is_relevant"]:
        log_message(
            f"Chunk {chunk_id} skipped - not relevant to core content: {relevance_check['reasoning']}",
            log_level="INFO",
            error_type="filtered_non_relevant",
        )
        return {
            "chunk_id": chunk_id,
            "path": path,
            "status": "filtered_non_relevant",
            "processing_time": time.time() - start_time,
            "relevance_check": relevance_check,
            "message": f"Skipped non-relevant content: {relevance_check['content_type']}",
        }

    # --------------------------------------------------------------------
    # Step 1: Summarize & expand the chunk => augmented_chunk
    # --------------------------------------------------------------------
    system_message = (
        "You are a helpful assistant that summarizes text in bullet points "
        "and expands on them using your broader knowledge. "
        "Name this result 'augmented_chunk'."
    )
    user_message = (
        f"Given the following chunk of text, please:\n\n"
        f"1. Summarize the text in bullet points.\n"
        f"2. Expand on the summary using your parametric knowledge.\n\n"
        f"Chunk:\n{chunk_text}\n\n"
        f"Return the result as plain text labeled 'augmented_chunk:' at the start."
    )

    try:
        step1_start_time = time.time()
        response_1 = batched_openai_completion(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
        )
        step1_output = response_1["choices"][0]["message"]["content"].strip()
        step1_time = time.time() - step1_start_time

        # We'll assume the model starts with "augmented_chunk:"
        augmented_chunk = step1_output
        if "augmented_chunk:" in step1_output.lower():
            augmented_chunk = re.split(
                r"augmented_chunk\s*:\s*",
                step1_output,
                flags=re.IGNORECASE,
                maxsplit=1,
            )[-1].strip()

    except Exception as e:
        log_message(
            f"Error summarizing chunk {chunk_id}: {e}",
            log_level="ERROR",
            error_type="error_summarizing",
        )
        return {
            "chunk_id": chunk_id,
            "path": path,
            "error": f"Error in step 1: {str(e)}",
            "status": "error",
            "processing_time": time.time() - start_time,
        }

    # --------------------------------------------------------------------
    # Step 2: Generate a MULTIPLE-CHOICE question with n answers
    # --------------------------------------------------------------------
    system_message_2 = (
        "You are a helpful assistant that generates high-quality multiple-choice questions "
        "based on text provided by the user. Each question should be challenging but fair, "
        "with one clearly correct answer and plausible but incorrect distractors."
    )

    user_message_2 = (
        f"Generate ONE well-formed multiple-choice question "
        f"with exactly {num_answers} answer choices labeled 1 through {num_answers}.\n\n"
        f"Text:\n{augmented_chunk}\n\n"
        f"Requirements:\n"
        f"1. Begin with 1-2 sentences of contextual information that establishes the domain/topic without referencing source materials.\n"
        f"2. Create a challenging question that tests deep understanding.\n"
        f"3. Ensure there is EXACTLY ONE clearly correct answer.\n"
        f"4. Make the other choices plausible but clearly incorrect.\n"
        f"5. The question should focus on a concept or fact that is clearly stated or strongly implied in the text.\n"
        f"6. Number your answer choices from 1 to {num_answers}.\n"
        f"7. Finally, indicate which number (1-{num_answers}) is the correct answer.\n"
        f"8. DO NOT provide explanations for why each answer is correct or incorrect.\n"
        f"9. CRITICAL: Both context and question must be completely self-contained. DO NOT reference any external materials including:\n"
        f"   - 'the text', 'the passage', 'the document', 'the paper', 'the study'\n"
        f"   - 'the author states', 'according to the text', 'as mentioned', 'as described'\n"
        f"   - 'Appendix', 'Figure', 'Table', 'Section', 'Chapter', 'above', 'below'\n"
        f"   - Any other references to source materials or external content\n"
        f"10. The context and question should read as if testing general knowledge on the topic, not comprehension of a specific text.\n"
        f"11. Answer choices should contain only direct technical information without meta-references to content or sources.\n\n"
        f"Your response must follow this format precisely: \n"
        f"CONTEXT: <1-2 sentences establishing domain/topic context>\n"
        f"QUESTION: <the question>\n"
        f"1: <first answer choice>\n"
        f"2: <second answer choice>\n"
        f"...\n"
        f"CORRECT ANSWER: <number of correct answer>\n"
    )

    try:
        step2_start_time = time.time()
        response_2 = batched_openai_completion(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message_2},
                {"role": "user", "content": user_message_2},
            ],
            temperature=0.8,
        )
        step2_output = response_2["choices"][0]["message"]["content"].strip()
        step2_time = time.time() - step2_start_time
    except Exception as e:
        log_message(
            f"Error generating question for chunk {chunk_id}: {e}",
            log_level="ERROR",
            error_type="error_question_gen",
        )
        return {
            "chunk_id": chunk_id,
            "path": path,
            "error": f"Error in step 2: {str(e)}",
            "status": "error",
            "processing_time": time.time() - start_time,
        }

    # --------------------------------------------------------------------
    # Step 3: Self-evaluate the generated question
    # --------------------------------------------------------------------
    system_message_3 = (
        "You are an expert teacher evaluating the quality of a multiple choice question. "
        "Your role is to ensure questions are clear, fair, and educationally valuable."
    )

    user_message_3 = (
        f"Evaluate the following multiple-choice question on a scale from 1-10, "
        f"where 10 is a perfect question.\n\n"
        f"CONTENT:\n{chunk_text}\n\n"
        f"QUESTION:\n{step2_output}\n\n"
        f"CONTENT RELEVANCE INFO:\n"
        f"- Relevance Score: {relevance_check['relevance_score']}/10\n"
        f"- Content Type: {relevance_check['content_type']}\n"
        f"- Relevance Reasoning: {relevance_check['reasoning']}\n\n"
        f"Rate the question based on these criteria:\n"
        f"- Clarity: Is the question clear and unambiguous?\n"
        f"- Accuracy: Is the content factually correct and aligned with the source material?\n"
        f"- Difficulty: Is the difficulty appropriate (challenging but fair)?\n"
        f"- Distractors: Are the incorrect options plausible but clearly wrong?\n"
        f"- Educational value: Does answering this question demonstrate understanding?\n"
        f"- Self-contained: CRITICAL - Does the question stand alone without ANY references to external materials?\n"
        f"- Content relevance: IMPORTANT - Questions based on low-relevance content (references, metadata, etc.) should receive lower scores\n\n"
        f"AUTOMATIC DISQUALIFIERS (score must be 1-3 if ANY are present):\n"
        f"- References to 'the text', 'the passage', 'the document', 'the paper', 'the study'\n"
        f"- References to 'the author', 'according to', 'as mentioned', 'as described'\n"
        f"- References to 'Appendix', 'Figure', 'Table', 'Section', 'Chapter'\n"
        f"- References to 'above', 'below', 'previously mentioned', 'following'\n"
        f"- Any other references that assume the reader has access to external materials\n"
        f"- Content based primarily on references, copyright notices, or metadata (should score 1-4)\n\n"
        f"SCORING ADJUSTMENT FOR CONTENT RELEVANCE:\n"
        f"- If content relevance score is 1-4: Maximum question score should be 4\n"
        f"- If content relevance score is 5-7: Maximum question score should be 7\n"
        f"- If content relevance score is 8-10: Normal scoring applies\n\n"
        f"A truly self-contained question should read like a general knowledge question on the topic.\n\n"
        f"Provide your response in this format:\n"
        f"SCORE: <numeric score between 1-10>\n"
        f"CRITIQUE: <brief explanation of score>\n"
    )

    try:
        step3_start_time = time.time()
        response_3 = batched_openai_completion(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message_3},
                {"role": "user", "content": user_message_3},
            ],
            temperature=0.3,
        )
        step3_output = response_3["choices"][0]["message"]["content"].strip()
        step3_time = time.time() - step3_start_time

        # Extract score from evaluation
        score_match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", step3_output)
        score = int(float(score_match.group(1))) if score_match else 0

        # Extract critique
        critique_match = re.search(r"CRITIQUE:(.*?)(?:\n\n|$)", step3_output, re.DOTALL)
        critique = (
            critique_match.group(1).strip()
            if critique_match
            else "No critique provided"
        )

    except Exception as e:
        log_message(
            f"Error evaluating question for chunk {chunk_id}: {e}",
            log_level="ERROR",
            error_type="error_question_eval",
        )
        return {
            "chunk_id": chunk_id,
            "path": path,
            "error": f"Error in step 3: {str(e)}",
            "status": "error",
            "processing_time": time.time() - start_time,
        }

    # Calculate total processing time
    processing_time = time.time() - start_time

    # If the score is below the threshold, don't return the question
    if score < min_score:
        log_message(
            f"Question for chunk {chunk_id} scored too low ({score}). Skipping.",
            log_level="WARNING",
            error_type="low_score_questions",
        )
        return {
            "chunk_id": chunk_id,
            "path": path,
            "score": score,
            "critique": critique,
            "status": "low_score",
            "processing_time": processing_time,
        }

    # Extract context, question and answers from the model output
    context_match = re.search(r"CONTEXT:\s*(.*?)(?:\n|$)", step2_output, re.DOTALL)
    raw_context = context_match.group(1).strip() if context_match else ""
    context = clean_answer_content(raw_context)

    question_match = re.search(r"QUESTION:\s*(.*?)(?:\n|$)", step2_output)
    raw_question = (
        question_match.group(1).strip() if question_match else "No question found"
    )
    question = clean_answer_content(raw_question)

    # Extract answer choices
    answer_pattern = r"(\d+):\s*(.*?)(?=\n\d+:|$)"
    answer_matches = re.findall(answer_pattern, step2_output, re.DOTALL)

    choices = []
    for num, text in answer_matches:
        clean_text = clean_answer_content(text)
        if clean_text:
            choices.append(clean_text)

    # Extract correct answer
    correct_answer_match = re.search(r"CORRECT ANSWER:\s*(\d+)", step2_output)
    correct_answer_num = (
        int(correct_answer_match.group(1)) if correct_answer_match else 1
    )

    # Build the final question text (with embedded choices)
    full_question = f"{context}\n\n{question}\n\n"
    for i, choice in enumerate(choices):
        full_question += f"{i + 1}) {choice}  \n"

    # Build the answer text
    if 1 <= correct_answer_num <= len(choices):
        answer_text = f"The correct answer is {correct_answer_num}) {choices[correct_answer_num - 1]}."
    else:
        answer_text = f"1) {choices[0]}" if choices else "No valid answer found"

    return {
        "chunk_id": chunk_id,
        "question": full_question.strip(),
        "answer": answer_text,
        "text": chunk_text,
        "type": "multiple-choice",
        "path": path,
        "choices": choices,
        "correct_answer_num": correct_answer_num,
        "processing_time": processing_time,
        "status": "success",
        "relevance_check": relevance_check,  # Store relevance information
        "quality_check": {
            "score": score,
            "critique": critique,
            "raw_output": step3_output,
        },
        "step_times": {
            "summarize": step1_time,
            "generate": step2_time,
            "evaluate": step3_time,
        },
    }


def generate_chunk_id(dataset_index: int, path: str) -> str:
    """
    Generate a chunk ID from dataset index and file path.
    Format: {file_id}_{chunk_index:04d}
    """
    # Create file_id from path
    file_id = hashlib.sha256(path.encode()).hexdigest()[:16]
    chunk_index = dataset_index
    return f"{file_id}_{chunk_index:04d}"


def reverse_chunk_id(chunk_id: str) -> Tuple[str, int]:
    """
    Reverse engineer a chunk_id to get file_id and chunk_index.

    Args:
        chunk_id: The chunk ID in format {file_id}_{chunk_index:04d}

    Returns:
        Tuple[str, int]: (file_id, chunk_index)
    """
    parts = chunk_id.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid chunk_id format: {chunk_id}")

    file_id = parts[0]
    try:
        chunk_index = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid chunk index in chunk_id: {chunk_id}")

    return file_id, chunk_index


def get_original_path_from_chunk_id(
    chunk_id: str, path_mapping: Dict[str, str]
) -> Optional[str]:
    """
    Get the original file path from a chunk_id using the path mapping.

    Args:
        chunk_id: The chunk ID to look up
        path_mapping: Dictionary mapping file_id to original_path

    Returns:
        Optional[str]: The original file path, or None if not found
    """
    try:
        file_id, _ = reverse_chunk_id(chunk_id)
        return path_mapping.get(file_id)
    except ValueError:
        return None


class CheckpointManager:
    """
    Checkpoint manager for tracking processed chunks and questions.
    """

    def __init__(self, checkpoint_file: str, force_restart: bool = False):
        self.checkpoint_file = checkpoint_file
        self.data = {
            "processed_chunks": {},
            "questions": [],
            "path_mapping": {},  # file_id -> original_path mapping
            "counters": {
                "total_chunks": 0,
                "processed_chunks": 0,
                "questions_generated": 0,
            },
            "error_stats": {
                "error_chunk_reading": 0,
                "error_summarizing": 0,
                "error_question_gen": 0,
                "error_question_eval": 0,
                "error_api": 0,
                "error_other": 0,
                "low_score_questions": 0,
                "filtered_non_relevant": 0,
                "total_errors": 0,
            },
        }

        if not force_restart:
            self._load_checkpoint()

    def _load_checkpoint(self):
        """Load checkpoint data from file."""
        try:
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                log_message(f"Loaded checkpoint from {self.checkpoint_file}")
            else:
                log_message("No existing checkpoint found, starting fresh")
        except Exception as e:
            log_message(f"Error loading checkpoint: {e}", log_level="ERROR")

    def save(self):
        """Save checkpoint data to file."""
        try:
            with open(self.checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log_message(f"Error saving checkpoint: {e}", log_level="ERROR")

    def is_chunk_processed(self, chunk_id: str) -> bool:
        """Check if a chunk has been processed."""
        return chunk_id in self.data["processed_chunks"]

    def update_processed_chunk(self, chunk_id: str, chunk_data: dict):
        """Update a processed chunk in the checkpoint."""
        self.data["processed_chunks"][chunk_id] = chunk_data
        self.save()

    def add_question(self, question_data: dict):
        """Add a generated question to the checkpoint."""
        self.data["questions"].append(question_data)
        self.data["counters"]["questions_generated"] = len(self.data["questions"])
        self.save()

    def get_questions(self) -> list:
        """Get all generated questions."""
        return self.data["questions"]

    def get_processed_chunks(self) -> dict:
        """Get all processed chunks."""
        return self.data["processed_chunks"]

    def get_counters(self) -> dict:
        """Get counter statistics."""
        return self.data["counters"]

    def get_error_stats(self) -> dict:
        """Get error statistics."""
        return self.data["error_stats"]

    def update_counters(self, **kwargs):
        """Update counter statistics."""
        for key, value in kwargs.items():
            if key in self.data["counters"]:
                self.data["counters"][key] = value
        self.save()

    def add_path_mapping(self, file_id: str, original_path: str):
        """Add a file_id to original_path mapping."""
        self.data["path_mapping"][file_id] = original_path
        self.save()

    def get_path_mapping(self) -> dict:
        """Get the path mapping dictionary."""
        return self.data.get("path_mapping", {})

    def get_original_path(self, chunk_id: str) -> Optional[str]:
        """Get the original file path from a chunk_id."""
        return get_original_path_from_chunk_id(chunk_id, self.get_path_mapping())


def configure_apis(model_name: str, config_file: str = "model_servers.yaml") -> str:
    """
    Configure the OpenAI client based on the model configuration.
    Returns the actual model name to use for API calls.
    """
    global _openai_client

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        log_message(
            f"Model configuration file '{config_file}' not found.",
            log_level="ERROR",
        )
        sys.exit(1)
    except yaml.YAMLError as e:
        log_message(f"Error parsing YAML file '{config_file}': {e}", log_level="ERROR")
        sys.exit(1)

    # Find the model configuration
    model_config = None
    for server in config.get("servers", []):
        if server.get("shortname") == model_name:
            model_config = server
            break

    if not model_config:
        log_message(
            f"Model shortname '{model_name}' not found in '{config_file}'.",
            log_level="ERROR",
        )
        sys.exit(1)

    # Get API configuration
    openai_api_key = model_config.get("openai_api_key")
    openai_api_base = model_config.get("openai_api_base")
    openai_model = model_config.get("openai_model")

    if not all([openai_api_key, openai_api_base, openai_model]):
        log_message(
            f"Missing API configuration for model '{model_name}'.", log_level="ERROR"
        )
        sys.exit(1)

    # Handle environment variable substitution
    if openai_api_key.startswith("${") and openai_api_key.endswith("}"):
        env_var = openai_api_key[2:-1]
        openai_api_key = os.environ.get(env_var)
        if not openai_api_key:
            log_message(
                f"Environment variable '{env_var}' not set for API key.",
                log_level="ERROR",
            )
            sys.exit(1)

    # Initialize OpenAI client
    try:
        _openai_client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
        log_message(f"Configured API client for model: {openai_model}")
        return openai_model
    except Exception as e:
        log_message(f"Error initializing OpenAI client: {e}", log_level="ERROR")
        sys.exit(1)


def process_dataset(
    dataset_path: str,
    output_file: str,
    model_name: str,
    question_type: QuestionType,
    num_answers: int,
    min_score: int,
    checkpoint_file: str,
    force_restart: bool = False,
    workers: int = None,
    max_chunks: int = None,
):
    """
    Process a Hugging Face dataset to generate questions.
    """
    global _total_chunks, _processed_chunks, terminal_ui

    log_message("Starting dataset processing...")
    log_message(f"Dataset path: {dataset_path}")
    log_message(f"Output file: {output_file}")
    log_message(f"Model: {model_name}")
    log_message(f"Question type: {question_type}")
    log_message(f"Workers: {workers}")

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(checkpoint_file, force_restart)

    # Initialize terminal UI
    terminal_ui = init_terminal_ui()

    try:
        # Load dataset
        log_message("Loading Hugging Face dataset...")
        dataset = datasets.load_from_disk(dataset_path)
        log_message(f"Loaded dataset with {len(dataset)} entries")

        # Limit chunks if specified
        if max_chunks and max_chunks < len(dataset):
            dataset = dataset.select(range(max_chunks))
            log_message(f"Limited to {max_chunks} chunks for processing")

        _total_chunks = len(dataset)

        # Update terminal UI
        if terminal_ui:
            terminal_ui.update_stats(
                total_chunks=_total_chunks,
                max_workers=workers or multiprocessing.cpu_count() - 1,
                status_message="Processing dataset chunks...",
            )

        # Process chunks
        successful_questions = 0

        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all chunks for processing
            future_to_chunk = {}

            for idx, row in enumerate(dataset):
                chunk_text = row["text"]
                path = row["path"]
                chunk_id = generate_chunk_id(idx, path)

                # Add path mapping
                file_id = hashlib.sha256(path.encode()).hexdigest()[:16]
                checkpoint_manager.add_path_mapping(file_id, path)

                # Skip if already processed
                if checkpoint_manager.is_chunk_processed(chunk_id):
                    _processed_chunks += 1
                    continue

                # Submit for processing
                future = executor.submit(
                    generate_multiple_choice_qa_pairs,
                    chunk_id,
                    chunk_text,
                    model_name,
                    path,
                    num_answers,
                    min_score,
                )
                future_to_chunk[future] = chunk_id

            # Process results
            for future in tqdm(future_to_chunk, desc="Processing chunks"):
                if _exit_requested:
                    break

                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    chunk_id = future_to_chunk[future]

                    # Update checkpoint
                    checkpoint_manager.update_processed_chunk(chunk_id, result)

                    # Add successful questions to output
                    if result["status"] == "success":
                        # Create cleaning_metadata structure
                        cleaning_metadata = {
                            "relevance_check": result["relevance_check"],
                            "quality_check": result["quality_check"],
                            "cleaned_at": time.time(),
                            "cleaning_version": "1.0",
                        }

                        question_data = {
                            "question": result["question"],
                            "answer": result["answer"],
                            "text": result["text"],
                            "type": result["type"],
                            "chunk_id": result["chunk_id"],
                            "path": result["path"],
                            "cleaning_metadata": cleaning_metadata,
                        }
                        checkpoint_manager.add_question(question_data)
                        successful_questions += 1

                    # Update counters
                    _processed_chunks += 1

                    # Update UI
                    if terminal_ui and _use_split_screen:
                        success_rate = (
                            successful_questions / max(1, _processed_chunks)
                        ) * 100
                        terminal_ui.update_stats(
                            chunks_processed=_processed_chunks,
                            questions_generated=successful_questions,
                            success_rate=success_rate,
                            status_message=f"Processed chunk {chunk_id}",
                        )

                except Exception as e:
                    log_message(
                        f"Error processing chunk result: {e}",
                        log_level="ERROR",
                        error_type="error_other",
                    )

        # Write final output
        log_message(f"Writing final output to {output_file}")
        questions = checkpoint_manager.get_questions()

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)

        log_message(f"Successfully generated {len(questions)} questions")

    except Exception as e:
        log_message(f"Error in main processing: {e}", log_level="ERROR")
        raise

    finally:
        cleanup_ui()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate multiple-choice questions from Hugging Face dataset"
    )

    parser.add_argument(
        "dataset_path", help="Path to the Hugging Face dataset directory"
    )

    parser.add_argument(
        "--output",
        default="questions_v2.json",
        help="Output JSON file (default: questions_v2.json)",
    )

    parser.add_argument(
        "--model",
        default="gpt4",
        help="Model shortname from model_servers.yaml (default: gpt4)",
    )

    parser.add_argument(
        "--type",
        type=QuestionType,
        choices=list(QuestionType),
        default=QuestionType.MULTIPLE_CHOICE,
        help="Type of questions to generate (default: mc)",
    )

    parser.add_argument(
        "--num-answers",
        type=int,
        default=7,
        help="Number of answer choices for multiple choice questions (default: 7)",
    )

    parser.add_argument(
        "--min-score",
        type=int,
        default=7,
        help="Minimum quality score to keep a question (default: 7)",
    )

    parser.add_argument(
        "--config",
        default="model_servers.yaml",
        help="Path to model configuration file (default: model_servers.yaml)",
    )

    parser.add_argument(
        "--checkpoint",
        default="mcqa_checkpoint_v2.json",
        help="Checkpoint file for resumable processing (default: mcqa_checkpoint_v2.json)",
    )

    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Force restart from beginning, ignoring existing checkpoint",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, multiprocessing.cpu_count() - 1),
        help="Number of worker processes (default: CPU count - 1)",
    )

    parser.add_argument(
        "--max-chunks",
        type=int,
        help="Maximum number of chunks to process (for testing)",
    )

    parser.add_argument(
        "--no-split-screen",
        action="store_true",
        help="Disable split-screen terminal UI",
    )

    parser.add_argument(
        "--error-threshold",
        type=int,
        default=200,
        help="Maximum number of errors before stopping (default: 200)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    global _use_split_screen, _max_error_threshold

    args = parse_arguments()

    _max_error_threshold = args.error_threshold
    _use_split_screen = not args.no_split_screen

    # Configure APIs
    actual_model_name = configure_apis(args.model, args.config)

    # Determine checkpoint file name based on output file
    checkpoint_file = args.checkpoint
    if (
        args.output != "questions_v2.json"
        and checkpoint_file == "mcqa_checkpoint_v2.json"
    ):
        output_base = os.path.splitext(args.output)[0]
        checkpoint_file = f"{output_base}_checkpoint.json"
        log_message(f"Using checkpoint file: {checkpoint_file}")

    # Process the dataset
    process_dataset(
        dataset_path=args.dataset_path,
        output_file=args.output,
        model_name=actual_model_name,
        question_type=args.type,
        num_answers=args.num_answers,
        min_score=args.min_score,
        checkpoint_file=checkpoint_file,
        force_restart=args.force_restart,
        workers=args.workers,
        max_chunks=args.max_chunks,
    )


if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown
    signal.signal(
        signal.SIGINT,
        lambda s, f: setattr(sys.modules[__name__], "_exit_requested", True),
    )
    signal.signal(
        signal.SIGTERM,
        lambda s, f: setattr(sys.modules[__name__], "_exit_requested", True),
    )

    try:
        main()
    except KeyboardInterrupt:
        log_message("Process interrupted by user", log_level="WARNING")
        sys.exit(1)
    except Exception as e:
        log_message(f"Fatal error: {e}", log_level="ERROR")
        sys.exit(1)
    finally:
        cleanup_ui()
