#!/usr/bin/env python3
"""
mcqa_generation_distllm.py - Multiple Choice Question Generation from Hugging Face Dataset

This script generates high-quality multiple-choice questions from a Hugging Face dataset
containing pre-processed text chunks. It bypasses PDF parsing and text chunking,
working directly with the provided dataset.

Usage:
    python mcqa_generation_distllm.py <dataset_path> --output <output.json> --model <model_name>

Dataset Requirements:
    - "text" column: Text chunks for question generation
    - "embedding" column: Embeddings (not used, but can be present)
    - "path" column: Source file path for traceability

Features:
    - Parallel processing with configurable workers
    - Checkpoint system for resumable processing
    - Terminal UI with progress tracking
    - Comprehensive error handling and logging
    - Quality evaluation and filtering
    - Same 3-step question generation as make_v21.py
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
            "total_errors": 0,
            "model_name": "Unknown",
        }

    def start(self):
        """Start the terminal UI in a separate thread."""
        global _use_split_screen
        if not sys.stdout.isatty() or os.environ.get("TERM") == "dumb":
            self.curses_enabled = False
            _use_split_screen = False
            return False

        try:
            self.curses_enabled = True
            self.running = True
            self.ui_thread = threading.Thread(target=self._run_ui, daemon=True)
            self.ui_thread.start()
            return True
        except Exception as e:
            print(f"Error starting terminal UI: {e}")
            self.curses_enabled = False
            _use_split_screen = False
            return False

    def stop(self):
        """Stop the terminal UI thread."""
        self.running = False
        if self.ui_thread:
            try:
                self.ui_thread.join(timeout=1.0)
            except:
                pass

    def _run_ui(self):
        """Main loop for the UI thread."""
        global _use_split_screen
        try:
            curses.wrapper(self._curses_main)
        except Exception as e:
            self.curses_enabled = False
            _use_split_screen = False
            print(f"Error in terminal UI: {e}")

    def _curses_main(self, stdscr):
        """Main curses function that sets up the UI."""
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_RED, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_CYAN, -1)
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)

        curses.curs_set(0)
        self.screen = stdscr

        height, width = self.screen.getmaxyx()

        self.top_pane = curses.newwin(self.top_height, width, 0, 0)
        self.bottom_pane = curses.newwin(
            height - self.top_height - 1, width, self.top_height + 1, 0
        )

        divider = curses.newwin(1, width, self.top_height, 0)
        divider.bkgd("-", curses.color_pair(5))
        divider.refresh()

        self.bottom_pane.scrollok(True)
        self.bottom_pane.idlok(True)

        while self.running:
            self._process_logs()
            self._update_display()
            time.sleep(0.1)

    def _process_logs(self):
        """Process any new log messages from the queue."""
        try:
            while not _log_queue.empty():
                try:
                    log_msg = _log_queue.get_nowait()
                    timestamp = time.strftime("%H:%M:%S", time.localtime())

                    if isinstance(log_msg, tuple) and len(log_msg) == 2:
                        msg_text, level = log_msg
                        if level == "ERROR":
                            formatted_msg = f"[{timestamp}] [ERROR] {msg_text}"
                        elif level == "WARNING":
                            formatted_msg = f"[{timestamp}] [WARNING] {msg_text}"
                        else:
                            formatted_msg = f"[{timestamp}] {msg_text}"
                    else:
                        formatted_msg = f"[{timestamp}] {log_msg}"

                    self.log_lines.append(formatted_msg)

                    if len(self.log_lines) > self.max_log_lines:
                        self.log_lines = self.log_lines[-self.max_log_lines :]

                    if (
                        self.log_position == len(self.log_lines) - 2
                        or self.log_position >= len(self.log_lines) - 1
                    ):
                        self.log_position = len(self.log_lines) - 1

                    _log_queue.task_done()
                except Exception as e:
                    try:
                        self.log_lines.append(
                            f"[{time.strftime('%H:%M:%S', time.localtime())}] [ERROR] Log processing error: {str(e)}"
                        )
                        _log_queue.task_done()
                    except:
                        pass
        except queue.Empty:
            pass

    def _update_display(self):
        """Update both panes with the latest information."""
        if not self.screen:
            return

        try:
            height, width = self.screen.getmaxyx()

            if height < 10 or width < 20:
                try:
                    self.screen.clear()
                    self.screen.addstr(0, 0, "Term too small")
                    self.screen.refresh()
                except:
                    pass
                return

            self._update_stats_pane()
            self._update_log_pane()

        except Exception as e:
            pass

    def _update_stats_pane(self):
        """Update the top pane with statistics."""
        if not self.top_pane:
            return

        try:
            self.top_pane.clear()

            # Title
            title = "MCQA Generation from Hugging Face Dataset"
            self.top_pane.addstr(0, 0, title, curses.color_pair(5))

            # Progress information
            elapsed = time.time() - _start_time
            self.stats["elapsed_time"] = elapsed

            # Calculate progress
            if self.stats["total_chunks"] > 0:
                progress = min(
                    100.0,
                    (self.stats["chunks_processed"] / self.stats["total_chunks"]) * 100,
                )
                self.stats["completion_percentage"] = progress

                # Calculate ETA
                if self.stats["chunks_processed"] > 0:
                    avg_time = elapsed / self.stats["chunks_processed"]
                    remaining = (
                        self.stats["total_chunks"] - self.stats["chunks_processed"]
                    )
                    eta_seconds = remaining * avg_time
                    self.stats["eta"] = self._format_time(eta_seconds)
                    self.stats["avg_chunk_time"] = f"{avg_time:.2f}s"

            # Progress bar
            progress_width = 50
            if self.stats["total_chunks"] > 0:
                progress_bar = self._generate_progress_bar(
                    self.stats["completion_percentage"] / 100.0, progress_width
                )
                self.top_pane.addstr(
                    2,
                    0,
                    f"Progress: {progress_bar} {self.stats['completion_percentage']:.1f}%",
                )

            # Statistics
            self.top_pane.addstr(
                4,
                0,
                f"Chunks: {self.stats['chunks_processed']}/{self.stats['total_chunks']}",
            )
            self.top_pane.addstr(
                5, 0, f"Questions Generated: {self.stats['questions_generated']}"
            )
            self.top_pane.addstr(
                6, 0, f"Success Rate: {self.stats['success_rate']:.1f}%"
            )
            self.top_pane.addstr(7, 0, f"Elapsed Time: {self._format_time(elapsed)}")
            self.top_pane.addstr(8, 0, f"ETA: {self.stats['eta']}")
            self.top_pane.addstr(
                9, 0, f"Avg Time/Chunk: {self.stats['avg_chunk_time']}"
            )

            # Worker information
            self.top_pane.addstr(
                11,
                0,
                f"Workers: {self.stats['active_workers']}/{self.stats['max_workers']}",
            )
            self.top_pane.addstr(12, 0, f"Model: {self.stats['model_name']}")

            # Error statistics
            self.top_pane.addstr(14, 0, f"Errors - Total: {self.stats['total_errors']}")
            self.top_pane.addstr(
                15, 0, f"  Summarizing: {self.stats['error_summarizing']}"
            )
            self.top_pane.addstr(
                16, 0, f"  Question Gen: {self.stats['error_question_gen']}"
            )
            self.top_pane.addstr(
                17, 0, f"  Question Eval: {self.stats['error_question_eval']}"
            )
            self.top_pane.addstr(
                18, 0, f"  Low Score: {self.stats['low_score_questions']}"
            )

            # Current status
            status = (
                self.stats["status_message"][:70] + "..."
                if len(self.stats["status_message"]) > 70
                else self.stats["status_message"]
            )
            self.top_pane.addstr(19, 0, f"Status: {status}")

            self.top_pane.refresh()

        except Exception as e:
            pass

    def _update_log_pane(self):
        """Update the bottom pane with log messages."""
        if not self.bottom_pane:
            return

        try:
            self.bottom_pane.clear()

            height, width = self.bottom_pane.getmaxyx()

            # Show recent log messages
            start_idx = max(0, len(self.log_lines) - height + 1)
            for i, line in enumerate(self.log_lines[start_idx:]):
                if i >= height - 1:
                    break

                # Truncate line if too long
                display_line = line[: width - 1] if len(line) >= width else line

                # Color coding
                color = curses.color_pair(0)
                if "[ERROR]" in line:
                    color = curses.color_pair(2)
                elif "[WARNING]" in line:
                    color = curses.color_pair(3)

                try:
                    self.bottom_pane.addstr(i, 0, display_line, color)
                except:
                    pass

            self.bottom_pane.refresh()

        except Exception as e:
            pass

    def _generate_progress_bar(self, fraction, width):
        """Generate a text progress bar."""
        filled = int(width * fraction)
        return "[" + "=" * filled + " " * (width - filled) + "]"

    def _format_time(self, seconds):
        """Format seconds into human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"

    def update_stats(self, **kwargs):
        """Update statistics dictionary."""
        for key, value in kwargs.items():
            if key in self.stats:
                self.stats[key] = value

    def log(self, message):
        """Add a log message to the queue."""
        if not _log_queue.full():
            _log_queue.put(message)


# Global terminal UI instance
terminal_ui = None


def init_terminal_ui():
    """Initialize the terminal UI."""
    global terminal_ui
    terminal_ui = TerminalUI()
    return terminal_ui.start()


def cleanup_ui():
    """Clean up the terminal UI."""
    global terminal_ui
    if terminal_ui:
        terminal_ui.stop()


def log_message(message, log_level="INFO", error_type=None):
    """Log a message with timestamp and level."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # Console output
    if _use_split_screen and terminal_ui and terminal_ui.curses_enabled:
        terminal_ui.log((message, log_level))
    else:
        print(f"[{timestamp}] [{log_level}] {message}")

    # Error log file
    if _error_log_file:
        try:
            with open(_error_log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] [{log_level}] {message}\n")
        except Exception as e:
            print(f"Error writing to log file: {e}")

    # Update error counters
    if error_type and terminal_ui:
        current_count = terminal_ui.stats.get(f"error_{error_type}", 0)
        terminal_ui.update_stats(**{f"error_{error_type}": current_count + 1})

        # Update total errors
        total_errors = sum(
            v for k, v in terminal_ui.stats.items() if k.startswith("error_")
        )
        terminal_ui.update_stats(total_errors=total_errors)

        # Check error threshold
        if current_count + 1 >= _max_error_threshold:
            log_message(
                f"Error threshold reached for {error_type} ({current_count + 1} >= {_max_error_threshold}). Stopping processing.",
                log_level="ERROR",
            )
            signal_handler(signal.SIGTERM, None)


def batched_openai_completion(model: str, messages: list, **kwargs):
    """
    Make an OpenAI API call with the global client.
    """
    global _openai_client

    if not _openai_client:
        raise ValueError("OpenAI client not initialized")

    # Check if we need to skip temperature (for reasoning models)
    skip_temperature = any(
        name in model.lower() for name in ["o3", "o4-mini", "o4mini"]
    )

    # Prepare parameters
    params = {"model": model, "messages": messages, **kwargs}

    # Remove temperature for reasoning models
    if skip_temperature and "temperature" in params:
        del params["temperature"]

    try:
        response = _openai_client.chat.completions.create(**params)
        # Convert to dict format for compatibility
        return {
            "choices": [{"message": {"content": response.choices[0].message.content}}]
        }
    except Exception as e:
        log_message(f"OpenAI API error: {e}", log_level="ERROR", error_type="api")
        raise


def clean_answer_content(answer_text: str) -> str:
    """
    Clean up answer content by removing extra whitespace and formatting.
    """
    if not answer_text:
        return ""

    # Remove excessive whitespace
    cleaned = re.sub(r"\s+", " ", answer_text.strip())

    # Remove common formatting artifacts
    cleaned = re.sub(r"^\s*[-*•]\s*", "", cleaned)
    cleaned = re.sub(r"\s*\n\s*", " ", cleaned)

    return cleaned.strip()


def clean_answer_choices(choices: List[str]) -> List[str]:
    """
    Clean up answer choices by removing numbering and extra whitespace.
    """
    cleaned_choices = []
    for choice in choices:
        # Remove leading numbers/letters and clean whitespace
        cleaned = re.sub(r"^\s*\d+[.):]\s*", "", choice)
        cleaned = re.sub(r"^\s*[A-Za-z][.):]\s*", "", cleaned)
        cleaned = clean_answer_content(cleaned)
        if cleaned:
            cleaned_choices.append(cleaned)
    return cleaned_choices


def generate_chunk_id(path: str, chunk_index: int) -> str:
    """
    Generate a unique chunk ID based on the file path and chunk index.
    """
    # Create a file ID from the path
    file_info = f"{path}_{os.path.getmtime(path) if os.path.exists(path) else 0}"
    file_id = hashlib.sha256(file_info.encode()).hexdigest()[:16]

    # Create chunk ID
    return f"{file_id}_{chunk_index:08d}"


def reverse_chunk_id(chunk_id: str) -> tuple:
    """
    Extract file_id and chunk_index from a chunk_id.

    Args:
        chunk_id: The chunk ID to reverse (format: file_id_chunk_index)

    Returns:
        tuple: (file_id, chunk_index) or (None, None) if invalid format
    """
    try:
        # Split chunk_id into file_id and chunk_index
        parts = chunk_id.rsplit("_", 1)
        if len(parts) != 2:
            return None, None

        file_id = parts[0]
        chunk_index = int(parts[1])

        return file_id, chunk_index
    except Exception:
        return None, None


def get_original_path_from_chunk_id(chunk_id: str, path_mapping: dict) -> Optional[str]:
    """
    Get the original file path from a chunk_id using the path mapping.

    Args:
        chunk_id: The chunk ID to resolve
        path_mapping: Dictionary mapping file_id to original path

    Returns:
        Optional[str]: Original file path or None if not found
    """
    file_id, chunk_index = reverse_chunk_id(chunk_id)
    if file_id is None:
        return None

    return path_mapping.get(file_id)


def example_usage():
    """
    Example of how to use the chunk ID reverse functionality.
    """
    # Example after processing a dataset
    checkpoint_manager = CheckpointManager("checkpoint.json")

    # Get a chunk ID from generated questions
    questions = checkpoint_manager.get_questions()
    if questions:
        chunk_id = questions[0]["metadata"]["chunk_id"]

        # Get the original file path (including extension)
        original_path = checkpoint_manager.get_original_path(chunk_id)
        print(f"Chunk {chunk_id} came from file: {original_path}")

        # Get file extension
        if original_path:
            file_extension = os.path.splitext(original_path)[1]
            print(f"File extension: {file_extension}")

        # Parse chunk ID components
        file_id, chunk_index = reverse_chunk_id(chunk_id)
        print(f"File ID: {file_id}, Chunk Index: {chunk_index}")


def generate_multiple_choice_qa_pairs(
    chunk_id: str,
    chunk_text: str,
    model_name: str,
    num_answers: int = 7,
    min_score: int = 7,
) -> Dict:
    """
    Generate a multiple-choice Q/A pair from a chunk.
    Returns a dictionary with question, answer, and other metadata.
    """
    global _exit_requested

    start_time = time.time()

    if _exit_requested:
        return {
            "chunk_id": chunk_id,
            "status": "cancelled",
            "processing_time": 0,
            "message": "Cancelled due to shutdown request",
        }

    # Step 1: Summarize & expand the chunk
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

        augmented_chunk = step1_output
        if "augmented_chunk:" in step1_output.lower():
            augmented_chunk = re.split(
                r"augmented_chunk\s*:\s*", step1_output, flags=re.IGNORECASE, maxsplit=1
            )[-1].strip()

    except Exception as e:
        log_message(
            f"Error summarizing chunk {chunk_id}: {e}",
            log_level="ERROR",
            error_type="summarizing",
        )
        return {
            "chunk_id": chunk_id,
            "error": f"Error in step 1: {str(e)}",
            "status": "error",
            "processing_time": time.time() - start_time,
        }

    # Step 2: Generate a multiple-choice question
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
            error_type="question_gen",
        )
        return {
            "chunk_id": chunk_id,
            "error": f"Error in step 2: {str(e)}",
            "status": "error",
            "processing_time": time.time() - start_time,
        }

    # Step 3: Self-evaluate the generated question
    system_message_3 = (
        "You are an expert teacher evaluating the quality of a multiple choice question. "
        "Your role is to ensure questions are clear, fair, and educationally valuable."
    )

    user_message_3 = (
        f"Evaluate the following multiple-choice question on a scale from 1-10, "
        f"where 10 is a perfect question.\n\n"
        f"CONTENT:\n{chunk_text}\n\n"
        f"QUESTION:\n{step2_output}\n\n"
        f"Rate the question based on these criteria:\n"
        f"- Clarity: Is the question clear and unambiguous?\n"
        f"- Accuracy: Is the content factually correct and aligned with the source material?\n"
        f"- Difficulty: Is the difficulty appropriate (challenging but fair)?\n"
        f"- Distractors: Are the incorrect options plausible but clearly wrong?\n"
        f"- Educational value: Does answering this question demonstrate understanding?\n"
        f"- Self-contained: CRITICAL - Does the question stand alone without ANY references to external materials?\n\n"
        f"AUTOMATIC DISQUALIFIERS (score must be 1-3 if ANY are present):\n"
        f"- References to 'the text', 'the passage', 'the document', 'the paper', 'the study'\n"
        f"- References to 'the author', 'according to', 'as mentioned', 'as described'\n"
        f"- References to 'Appendix', 'Figure', 'Table', 'Section', 'Chapter'\n"
        f"- References to 'above', 'below', 'previously mentioned', 'following'\n"
        f"- Any other references that assume the reader has access to external materials\n\n"
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
            error_type="question_eval",
        )
        return {
            "chunk_id": chunk_id,
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
            error_type="low_score",
        )
        return {
            "chunk_id": chunk_id,
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

    # Build the final question text
    full_question = f"{context}\n\n{question}"

    # Build the answer text
    if 1 <= correct_answer_num <= len(choices):
        answer_text = f"{correct_answer_num}. {choices[correct_answer_num - 1]}"
    else:
        answer_text = f"1. {choices[0]}" if choices else "No valid answer found"

    return {
        "chunk_id": chunk_id,
        "question": full_question,
        "answer": answer_text,
        "choices": choices,
        "correct_answer_num": correct_answer_num,
        "score": score,
        "critique": critique,
        "processing_time": processing_time,
        "status": "success",
    }


def worker_process_chunk(
    chunk_id, chunk_text, model_name, question_type, num_answers, min_score
):
    """
    Worker function to process a single chunk.
    """
    try:
        if question_type == QuestionType.MULTIPLE_CHOICE:
            return generate_multiple_choice_qa_pairs(
                chunk_id, chunk_text, model_name, num_answers, min_score
            )
        else:
            # For free-form QA, we'd implement a similar function
            # For now, just return an error
            return {
                "chunk_id": chunk_id,
                "error": "Free-form QA not implemented yet",
                "status": "error",
                "processing_time": 0,
            }
    except Exception as e:
        return {
            "chunk_id": chunk_id,
            "error": f"Worker error: {str(e)}",
            "status": "error",
            "processing_time": 0,
        }


def signal_handler(signum, frame):
    """Handle interrupt signals for graceful shutdown."""
    global _exit_requested
    _exit_requested = True
    log_message(
        "Interrupt signal received. Shutting down gracefully...", log_level="WARNING"
    )


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
    Configure the OpenAI API based on model selection.
    """
    global _openai_client

    try:
        with open(config_file, "r") as f:
            servers_config = yaml.safe_load(f)
    except Exception as e:
        log_message(f"Error loading {config_file}: {e}", log_level="ERROR")
        sys.exit(1)

    # Find the selected model's configuration
    selected_server = None
    for server in servers_config["servers"]:
        if server["shortname"] == model_name:
            selected_server = server
            break

    if not selected_server:
        log_message(
            f"Error: Model '{model_name}' not found in {config_file}", log_level="ERROR"
        )
        log_message(
            f"Available models: {', '.join(s['shortname'] for s in servers_config['servers'])}",
            log_level="INFO",
        )
        sys.exit(1)

    # Get API key
    api_key = selected_server.get("openai_api_key", "dummy_key_not_used")
    if api_key.startswith("${") and api_key.endswith("}"):
        env_var = api_key[2:-1]
        api_key = os.environ.get(env_var, "")
        if not api_key:
            log_message(
                f"Error: Environment variable {env_var} not set", log_level="ERROR"
            )
            sys.exit(1)

    # Initialize the OpenAI client
    client_config = {"api_key": api_key}

    if "openai_api_base" in selected_server:
        client_config["base_url"] = selected_server.get("openai_api_base")

    if "org_id" in selected_server:
        client_config["organization"] = selected_server["org_id"]

    _openai_client = OpenAI(**client_config)

    actual_model_name = selected_server.get("openai_model")

    base_url = selected_server.get("openai_api_base", "https://api.openai.com/v1")
    log_message(f"Configured OpenAI API with base URL: {base_url}")
    log_message(f"Using model shortname: {model_name}")
    log_message(f"Actual model identifier: {actual_model_name}")

    return actual_model_name


def process_dataset(
    dataset_path: str,
    output_file: str,
    model_name: str,
    question_type: QuestionType,
    num_answers: int,
    min_score: int,
    checkpoint_file: str,
    force_restart: bool = False,
    workers: Optional[int] = None,
    max_chunks: Optional[int] = None,
):
    """
    Main processing function for the Hugging Face dataset.
    """
    global _total_chunks, _processed_chunks, _start_time, _max_workers, _error_log_file

    # Set max workers
    _max_workers = workers if workers else max(1, multiprocessing.cpu_count() - 1)
    _start_time = time.time()

    # Initialize error log
    output_dir = os.path.dirname(output_file)
    output_name = os.path.splitext(os.path.basename(output_file))[0]
    _error_log_file = os.path.join(output_dir, f"{output_name}_errors.log")

    with open(_error_log_file, "w", encoding="utf-8") as f:
        f.write(
            f"Error log for MCQA generation started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(f"Error threshold set to: {_max_error_threshold}\n")
        f.write(f"Dataset path: {dataset_path}\n")
        f.write(f"Output file: {output_file}\n")
        f.write(f"Checkpoint file: {checkpoint_file}\n")
        f.write("=" * 80 + "\n\n")

    # Initialize terminal UI
    ui_initialized = init_terminal_ui()
    if ui_initialized:
        log_message("Terminal UI initialized successfully")
    else:
        log_message(
            "Terminal UI initialization failed, falling back to standard output",
            log_level="WARNING",
        )

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(checkpoint_file, force_restart)

    try:
        # Load dataset
        log_message(f"Loading dataset from {dataset_path}")
        if terminal_ui and _use_split_screen:
            terminal_ui.update_stats(
                status_message="Loading dataset...",
                max_workers=_max_workers,
                model_name=model_name,
            )

        # Load the dataset
        dataset = datasets.load_from_disk(dataset_path)

        # If it's a DatasetDict, get the first split
        if isinstance(dataset, datasets.DatasetDict):
            dataset = dataset[list(dataset.keys())[0]]

        # Limit dataset size if specified
        if max_chunks and len(dataset) > max_chunks:
            dataset = dataset.select(range(max_chunks))

        _total_chunks = len(dataset)
        log_message(f"Loaded dataset with {_total_chunks} chunks")

        # Update UI
        if terminal_ui and _use_split_screen:
            terminal_ui.update_stats(
                total_chunks=_total_chunks, status_message="Processing chunks..."
            )

        # Get already processed chunks
        processed_chunks = checkpoint_manager.get_processed_chunks()

        # Create list of chunks to process and build path mapping
        chunks_to_process = []
        for i, row in enumerate(dataset):
            chunk_id = generate_chunk_id(row["path"], i)

            # Add to path mapping
            file_id, _ = reverse_chunk_id(chunk_id)
            if file_id:
                checkpoint_manager.add_path_mapping(file_id, row["path"])

            if chunk_id not in processed_chunks:
                chunks_to_process.append((chunk_id, row["text"]))

        log_message(f"Found {len(chunks_to_process)} chunks to process")

        # Process chunks in parallel
        if chunks_to_process:
            with ProcessPoolExecutor(max_workers=_max_workers) as executor:
                # Submit all tasks
                futures = []
                for chunk_id, chunk_text in chunks_to_process:
                    if _exit_requested:
                        break
                    future = executor.submit(
                        worker_process_chunk,
                        chunk_id,
                        chunk_text,
                        model_name,
                        question_type,
                        num_answers,
                        min_score,
                    )
                    futures.append(future)

                # Process results
                for future in tqdm(
                    futures, desc="Processing chunks", disable=_use_split_screen
                ):
                    if _exit_requested:
                        break

                    try:
                        result = future.result(timeout=300)  # 5 minute timeout

                        # Update checkpoint
                        checkpoint_manager.update_processed_chunk(
                            result["chunk_id"], result
                        )

                        # Add successful questions to output
                        if result["status"] == "success":
                            question_data = {
                                "question": result["question"],
                                "answer": result["answer"],
                                "choices": result["choices"],
                                "metadata": {
                                    "chunk_id": result["chunk_id"],
                                    "score": result["score"],
                                    "processing_time": result["processing_time"],
                                },
                            }
                            checkpoint_manager.add_question(question_data)

                        # Update counters
                        _processed_chunks += 1

                        # Update UI
                        if terminal_ui and _use_split_screen:
                            success_rate = (
                                len(checkpoint_manager.get_questions())
                                / max(1, _processed_chunks)
                            ) * 100
                            terminal_ui.update_stats(
                                chunks_processed=_processed_chunks,
                                questions_generated=len(
                                    checkpoint_manager.get_questions()
                                ),
                                success_rate=success_rate,
                                status_message=f"Processed chunk {result['chunk_id']}",
                            )

                    except Exception as e:
                        log_message(
                            f"Error processing chunk result: {e}",
                            log_level="ERROR",
                            error_type="other",
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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate multiple-choice questions from a Hugging Face dataset."
    )
    parser.add_argument(
        "dataset_path", help="Path to the Hugging Face dataset directory"
    )
    parser.add_argument(
        "--output",
        default="questions.json",
        help="Output JSON file (default: questions.json)",
    )
    parser.add_argument(
        "--model", default="gpt4", help="Model shortname from model configuration file"
    )
    parser.add_argument(
        "--config",
        default="model_servers.yaml",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--type",
        type=QuestionType,
        choices=list(QuestionType),
        default=QuestionType.MULTIPLE_CHOICE,
        help="Type of questions to generate",
    )
    parser.add_argument(
        "--num-answers",
        type=int,
        default=7,
        help="Number of answer choices (default: 7)",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=7,
        help="Minimum quality score (1-10) (default: 7)",
    )
    parser.add_argument(
        "--checkpoint",
        default="mcqa_checkpoint.json",
        help="Checkpoint file for resumable processing",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Force restart, ignoring checkpoint",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count - 1)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum number of chunks to process (for testing)",
    )
    parser.add_argument(
        "--no-split-screen", action="store_true", help="Disable split-screen UI"
    )
    parser.add_argument(
        "--error-threshold",
        type=int,
        default=200,
        help="Maximum errors before stopping (default: 200)",
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
    if args.output != "questions.json" and checkpoint_file == "mcqa_checkpoint.json":
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
    main()
