#!/usr/bin/env python3
"""
reasoning_traces_multimode_v1.py - Multi-mode reasoning trace generator

This script generates reasoning traces for multiple-choice questions using three different
reasoning modes (detailed, focused, efficient) without exposing correct answers to the model.
The output includes all three reasoning traces for each question in a structured format
suitable for RAG distillation.

Key features:
- Processes questions with detailed, focused, and efficient reasoning modes
- Does not expose correct answers to the reasoning model
- Preserves input order with question_id indexing
- Includes original text field and question content
- Parallel processing support
- Robust error handling for partial failures
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import yaml
from openai import OpenAI
from tqdm import tqdm

# Global variables
_start_time = time.time()
_total_questions = 0
_processed_questions = 0
_current_model_name = None


def log_message(message, log_level="INFO"):
    """Log a message with timestamp and log level."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{log_level}] {message}")


def parse_arguments():
    """Parse command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate multi-mode reasoning traces for multiple choice questions without exposing correct answers."
    )
    parser.add_argument(
        "input_file",
        help="JSON file containing multiple choice questions",
    )
    parser.add_argument(
        "--output",
        default="reasoning_traces_multimode.json",
        help="Output JSON file (default: reasoning_traces_multimode.json)",
    )
    parser.add_argument(
        "--model",
        default="gpt4",
        help="Model shortname from model_servers.yaml to use",
    )
    parser.add_argument(
        "--config",
        default="model_servers.yaml",
        help="Path to model configuration file (default: model_servers.yaml)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions to process (default: all)",
    )
    parser.add_argument(
        "--specialty",
        default="expert",
        help='Specialty persona to adopt (e.g., "microbiologist", "quantum physicist", "historian") (default: expert)',
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save partial results after processing this many questions (default: 10)",
    )
    parser.add_argument(
        "--continue-from",
        default=None,
        help="Continue from a previously saved output file",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for processing questions (default: 1)",
    )
    parser.add_argument(
        "--parallel-modes",
        action="store_true",
        help="Process reasoning modes in parallel (faster but uses more API calls simultaneously)",
    )

    return parser.parse_args()


def configure_apis(
    model_name: str, config_file: str = "model_servers.yaml"
) -> tuple[OpenAI, str]:
    """
    Configure the necessary APIs based on model selection.

    Args:
        model_name: The model shortname to use
        config_file: Path to the model configuration file

    Returns:
        Tuple of (OpenAI client, actual model name to use with the API)
    """
    # Load the servers configuration
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

    # Configure OpenAI API with server details
    log_message(
        f"Using model '{selected_server['openai_model']}' from {selected_server['server']}"
    )

    # Set OpenAI API parameters
    openai_api_key = selected_server["openai_api_key"]
    # Handle environment variables in the API key
    if openai_api_key.startswith("${") and openai_api_key.endswith("}"):
        env_var = openai_api_key[2:-1]
        openai_api_key = os.environ.get(env_var, "")
        if not openai_api_key:
            log_message(
                f"Error: Environment variable {env_var} is not set or empty",
                log_level="ERROR",
            )
            sys.exit(1)

    # Create OpenAI client with the new API
    client = OpenAI(api_key=openai_api_key, base_url=selected_server["openai_api_base"])

    # Return the client and actual model name to use with the API
    return client, selected_server["openai_model"]


def get_expert_persona(specialty: str) -> str:
    """
    Generate a detailed persona description for the selected specialty.

    Args:
        specialty: The expert specialty (e.g., microbiologist, quantum physicist, historian)

    Returns:
        A detailed persona description tailored to the specialty
    """
    # Pre-defined personas for common specialties
    predefined_personas = {
        "microbiologist": """I am a microbiologist with over 20 years of experience studying antimicrobial resistance and bacterial pathogenesis. 
I've spent countless hours in the lab isolating bacterial strains, conducting susceptibility tests, and analyzing genomic data. 
When I approach a scientific question, I consider the molecular mechanisms at play, evolutionary pressures, and ecological contexts. 
I'm particularly meticulous about methodology and constantly thinking about experimental design, controls, and statistical significance. 
I tend to connect new information to established principles in bacterial physiology, genetics, and ecology. 
I'm familiar with current literature on antimicrobial agents, resistance mechanisms, biofilms, and emerging therapeutic approaches.""",
        "physicist": """I am a physicist with over 20 years of experience in theoretical and computational physics. 
I've worked extensively on quantum mechanics, statistical mechanics, and particle physics. 
When I approach a physics problem, I consider the underlying physical principles, mathematical formulations, and experimental evidence.
I'm particularly attentive to mathematical rigor, dimensional analysis, and the implications of symmetries.
I tend to connect new information to established theories and look for consistency with fundamental laws.
I'm familiar with current research on quantum field theory, cosmology, condensed matter physics, and computational methods.""",
        "quantum physicist": """I am a quantum physicist with extensive experience in quantum mechanics, quantum field theory, and quantum computing. 
My research focuses on understanding the fundamental principles of quantum systems and their applications in technology. 
When approaching problems, I instinctively think about wave functions, quantum states, superposition, entanglement, and quantum measurement theory. 
I consider both the mathematical formalism and the conceptual interpretations of quantum phenomena. 
My approach is rigorous, often using advanced mathematical tools to analyze quantum systems and their behavior. 
I'm familiar with current research in quantum technologies, quantum information processing, and quantum foundations.""",
        "historian": """I am a historian with decades of experience in analyzing historical documents, events, and trends. 
My expertise involves critically examining primary and secondary sources, contextualizing events within their broader historical context. 
When analyzing historical questions, I consider multiple perspectives, sociopolitical factors, economic conditions, and cultural influences. 
I'm particularly attentive to the biases in historical accounts and the importance of evaluating the reliability of sources. 
My approach involves connecting specific events to larger historical patterns and understanding how past developments influence present conditions. 
I'm well-versed in historiography and the evolution of historical interpretations over time.""",
    }

    # Check if the specialty is in our predefined list
    if specialty.lower() in predefined_personas:
        return predefined_personas[specialty.lower()]

    # For unknown specialties, generate a generic expert persona based on the specialty name
    specialty_words = specialty.split()
    specialty_base = specialty_words[-1] if len(specialty_words) > 0 else specialty

    # Is it a scientific field?
    scientific_fields = [
        "biologist",
        "physicist",
        "chemist",
        "geologist",
        "astronomer",
        "mathematician",
        "engineer",
        "scientist",
        "researcher",
    ]

    is_scientific = any(field in specialty_base.lower() for field in scientific_fields)

    if is_scientific:
        return f"""I am a {specialty} with extensive expertise in my field. 
My work involves analyzing complex scientific problems using rigorous methodologies and detailed knowledge of {specialty} principles.
When approaching questions in my field, I think systematically about the underlying mechanisms, relevant theories, and empirical evidence.
I pay particular attention to scientific accuracy, methodological considerations, and the current state of research in {specialty}.
My approach combines theoretical understanding with practical knowledge of experimental techniques and data analysis.
I'm well-versed in the latest research and ongoing debates in the field of {specialty}."""
    else:
        # Generic expert persona for non-scientific fields
        return f"""I am a {specialty} with extensive expertise and experience in my field.
My work involves analyzing complex problems through the specialized lens of a {specialty}.
When approaching questions in my field, I consider multiple factors, theoretical frameworks, and practical implications.
I'm particularly attentive to the nuances, contexts, and specialized knowledge that inform {specialty} analysis.
My approach combines theoretical understanding with practical insights gained through years of experience.
I'm well-versed in the foundational principles, current developments, and ongoing debates in my field."""


def generate_reasoning_prompt(
    specialty: str,
    persona: str,
    question_text: str,
    options: List[str],
    reasoning_mode: str,
) -> str:
    """
    Generate reasoning prompt based on the selected reasoning mode.

    Args:
        specialty: Expert specialty
        persona: Expert persona description
        question_text: The question text
        options: List of answer options
        reasoning_mode: "detailed", "focused", or "efficient"

    Returns:
        Complete prompt string
    """
    # Check if scientific field
    is_scientific = any(
        field in specialty.lower()
        for field in [
            "scientist",
            "biologist",
            "physicist",
            "chemist",
            "geologist",
            "astronomer",
            "mathematician",
            "engineer",
        ]
    )

    # Split the question text to get just the question part (without options)
    question_parts = question_text.split("\n\n", 1)
    question_only = question_parts[0] if len(question_parts) > 0 else question_text

    # Base prompt structure
    base_prompt = f"""You are a {specialty} reasoning through a multiple-choice question. Your persona: {persona}

QUESTION:
{question_only}

ANSWER OPTIONS:
"""

    # Add options
    for i, option in enumerate(options):
        base_prompt += f"{i + 1}. {option}\n"

    # Add reasoning instructions based on mode
    if reasoning_mode == "detailed":
        return base_prompt + generate_detailed_instructions(specialty, is_scientific)
    elif reasoning_mode == "focused":
        return base_prompt + generate_focused_instructions(specialty, is_scientific)
    elif reasoning_mode == "efficient":
        return base_prompt + generate_efficient_instructions(specialty, is_scientific)
    else:
        # Default to detailed
        return base_prompt + generate_detailed_instructions(specialty, is_scientific)


def generate_detailed_instructions(specialty: str, is_scientific: bool) -> str:
    """Generate detailed reasoning instructions with prediction."""
    if is_scientific:
        return f"""
TASK:
Please provide an extremely detailed analysis as a {specialty} thinking through this problem. For each answer option:
1. Treat each option as a hypothesis that you're carefully considering
2. Use specialized terminology and concepts from your field in your reasoning
3. Consider relevant mechanisms, processes, theoretical frameworks, and evidence
4. Reason through the implications and logical consequences of each option
5. Reference relevant principles, theories, or frameworks from your field
6. Consider edge cases, exceptions, and nuances for each option
7. Express uncertainty and weigh evidence where appropriate

Structure your response as an expert's analytical process:
- Analyze each option thoroughly in sequential order (Option 1, then Option 2, etc.)
- For each option, begin with "Analyzing option X..."
- After analyzing ALL options, make your prediction based on your analysis
- Provide your reasoning for the prediction and confidence level

Output your reasoning in JSON format with the following structure:
{{
  "thought_process": {{
    "option_1": "Detailed reasoning about option 1 as a hypothesis",
    "option_2": "Detailed reasoning about option 2 as a hypothesis",
    ... (all options in numerical order)
  }},
  "reasoning_summary": "Overall synthesized analysis",
  "prediction": {{
    "predicted_answer": "The option number you predict is correct (e.g., 3)",
    "prediction_reasoning": "Brief explanation of why you predict this answer",
    "confidence_level": "high/medium/low",
    "confidence_explanation": "Why you have this confidence level"
  }}
}}

IMPORTANT: Your response must be a valid, parseable JSON object. For each option, include detailed reasoning of at least 150-200 words. You MUST make a prediction and specify ONLY the option number (e.g., '3', NOT 'Option 3').
"""
    else:
        return f"""
TASK:
Please provide an extremely detailed analysis as a {specialty} thinking through this problem. For each answer option:
1. Treat each option as a possibility that you're carefully considering
2. Use specialized terminology and concepts from your field in your reasoning
3. Consider relevant frameworks, methodologies, contexts, and evidence
4. Reason through the implications and logical consequences of each option
5. Reference relevant principles, theories, or frameworks from your field
6. Consider edge cases, exceptions, and nuances for each option
7. Express uncertainty and weigh evidence where appropriate

Structure your response as an expert's analytical process:
- Analyze each option thoroughly in sequential order (Option 1, then Option 2, etc.)
- For each option, begin with "Analyzing option X..."
- After analyzing ALL options, make your prediction based on your analysis
- Provide your reasoning for the prediction and confidence level

Output your reasoning in JSON format with the following structure:
{{
  "thought_process": {{
    "option_1": "Detailed reasoning about option 1 as a possibility",
    "option_2": "Detailed reasoning about option 2 as a possibility",
    ... (all options in numerical order)
  }},
  "reasoning_summary": "Overall synthesized analysis",
  "prediction": {{
    "predicted_answer": "The option number you predict is correct (e.g., 3)",
    "prediction_reasoning": "Brief explanation of why you predict this answer",
    "confidence_level": "high/medium/low",
    "confidence_explanation": "Why you have this confidence level"
  }}
}}

IMPORTANT: Your response must be a valid, parseable JSON object. For each option, include detailed reasoning of at least 150-200 words. You MUST make a prediction and specify ONLY the option number (e.g., '3', NOT 'Option 3').
"""


def generate_focused_instructions(specialty: str, is_scientific: bool) -> str:
    """Generate focused reasoning instructions with prediction."""
    if is_scientific:
        return f"""
TASK:
As a {specialty}, analyze this question efficiently but thoroughly. Focus on the key scientific principles that differentiate the options:

1. Identify the core scientific concept being tested
2. Analyze the plausibility of each option based on scientific principles
3. Focus detailed analysis on the most scientifically interesting aspects
4. Use your scientific knowledge to identify the key distinguishing factors
5. Make a prediction based on your focused analysis

Structure your response efficiently:
- Briefly explain the key scientific principle at stake
- Analyze each option's scientific plausibility (3-4 sentences each)
- Make your prediction with reasoning and confidence level

Output in JSON format:
{{
  "key_principle": "The main scientific concept being tested",
  "option_analysis": {{
    "option_1": "Scientific analysis of option 1's plausibility",
    "option_2": "Scientific analysis of option 2's plausibility",
    ... (all options)
  }},
  "scientific_assessment": "Overall scientific analysis",
  "prediction": {{
    "predicted_answer": "The option number you predict is correct (e.g., 3)",
    "prediction_reasoning": "Brief scientific reasoning for your prediction",
    "confidence_level": "high/medium/low",
    "confidence_explanation": "Why you have this confidence level"
  }}
}}

IMPORTANT: Be focused but thorough. You MUST make a prediction and specify ONLY the option number (e.g., '3', NOT 'Option 3').
"""
    else:
        return f"""
TASK:
As a {specialty}, analyze this question efficiently but thoroughly. Focus on the key factors that differentiate the options:

1. Identify the core concept or principle being tested
2. Analyze the plausibility of each option based on your expertise
3. Focus detailed analysis on the most relevant aspects
4. Use your expertise to identify the key distinguishing factors
5. Make a prediction based on your focused analysis

Structure your response efficiently:
- Briefly explain the key principle at stake
- Analyze each option's plausibility (3-4 sentences each)
- Make your prediction with reasoning and confidence level

Output in JSON format:
{{
  "key_principle": "The main concept being tested",
  "option_analysis": {{
    "option_1": "Analysis of option 1's plausibility",
    "option_2": "Analysis of option 2's plausibility",
    ... (all options)
  }},
  "overall_assessment": "Overall analysis",
  "prediction": {{
    "predicted_answer": "The option number you predict is correct (e.g., 3)",
    "prediction_reasoning": "Brief reasoning for your prediction",
    "confidence_level": "high/medium/low",
    "confidence_explanation": "Why you have this confidence level"
  }}
}}

IMPORTANT: Be focused but thorough. You MUST make a prediction and specify ONLY the option number (e.g., '3', NOT 'Option 3').
"""


def generate_efficient_instructions(specialty: str, is_scientific: bool) -> str:
    """Generate efficient reasoning instructions with prediction."""
    return f"""
TASK:
As a {specialty}, provide a streamlined analysis of this question:

1. Identify what the question is really asking
2. Apply your expertise to analyze each option's validity
3. Provide concise reasoning for each option's strengths and weaknesses
4. Make a quick prediction based on your analysis

Be direct and focused - trust your expertise and make a confident choice.

Output in JSON format:
{{
  "question_focus": "Brief explanation of what the question tests",
  "option_analysis": {{
    "option_1": "Concise analysis of option 1 (2-3 sentences)",
    "option_2": "Concise analysis of option 2 (2-3 sentences)",
    ... (all options)
  }},
  "key_insights": "Main insights from the analysis",
  "prediction": {{
    "predicted_answer": "The option number you predict is correct (e.g., 3)",
    "prediction_reasoning": "Brief reasoning for your choice (1-2 sentences)",
    "confidence_level": "high/medium/low"
  }}
}}

IMPORTANT: Be concise and direct. Trust your instincts and make a prediction. Specify ONLY the option number (e.g., '3', NOT 'Option 3').
"""


def extract_mc_options(question_text: str) -> List[str]:
    """
    Extract multiple choice options from the question text.

    Args:
        question_text: The full multiple choice question text

    Returns:
        List of extracted options without their numbers/letters
    """
    # Split the question into the actual question and the options
    parts = question_text.split("\n\n", 1)
    if len(parts) < 2:
        # Handle case where there's no clear separation
        return []

    options_text = parts[1]

    # Match different option formats like "1.", "1)", "A.", "A)", etc.
    options = re.findall(
        r"(?:^|\n)(?:\d+|\w)[.)] (.*?)(?=(?:\n(?:\d+|\w)[.)])|$)",
        options_text,
        re.DOTALL,
    )

    # Clean up the options (remove asterisks marking correct answers, etc.)
    cleaned_options = [
        re.sub(r"\s*\(\*\)\s*$", "", option.strip()) for option in options
    ]

    return cleaned_options


def extract_thought_process_from_text(text: str, option_count: int) -> Dict[str, str]:
    """
    Extract thought process for each option from raw text when JSON parsing fails.

    Args:
        text: The raw text from the model
        option_count: The number of options in the question

    Returns:
        Dictionary with thought process for each option
    """
    thought_process = {}

    # Look for patterns like "Option 1:" or "Analyzing option 1"
    option_patterns = [
        r"(?:Option|OPTION)\s+(\d+)[\s:]+(.*?)(?=(?:Option|OPTION)\s+\d+[\s:]|$)",
        r"(?:Analyzing|Considering|Examining)\s+(?:option|Option)\s+(\d+)[\s.:]+(.*?)(?=(?:Analyzing|Considering|Examining)\s+(?:option|Option)|$)",
        r"(?:^|\n)(\d+)[.]:?\s+(.*?)(?=(?:^|\n)\d+[.:]|$)",
    ]

    for pattern in option_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for opt_num, content in matches:
            if not content.strip():  # Skip empty content
                continue
            try:
                opt_idx = int(opt_num)
                if 1 <= opt_idx <= option_count:  # Ensure it's a valid option number
                    thought_process[f"option_{opt_idx}"] = content.strip()
            except ValueError:
                continue

    return thought_process


def extract_prediction_from_text(text: str) -> Dict[str, str]:
    """
    Extract prediction information from raw text when JSON parsing fails.

    Args:
        text: The raw text from the model

    Returns:
        Dictionary with prediction information
    """
    prediction = {
        "predicted_answer": "Could not determine",
        "prediction_reasoning": "",
        "confidence_level": "unknown",
        "confidence_explanation": "",
    }

    # Try to find the predicted option number
    predict_patterns = [
        r"I\s+predict\s+(?:that\s+)?(?:option|answer)\s*(?:number|#)?\s*(\d+)",
        r"(?:My prediction|My answer|I believe|I think)\s+(?:is|would be)\s+(?:option|answer)?\s*(?:number|#)?\s*(\d+)",
        r"(?:option|answer)\s+(\d+)\s+(?:is|seems|appears to be)\s+(?:the\s+)?correct",
        r"(?:based on|after)\s+(?:my|this)\s+analysis,\s+(?:option|answer)\s+(\d+)",
        r"(?:therefore|thus|hence),\s+(?:option|answer)\s+(\d+)",
        r"(?:I would|I am going to|I will)\s+(?:choose|select|pick|go with)\s+(?:option|answer)\s+(\d+)",
        r"(?:option|answer)\s+(\d+)[\s.:,]",
        r"(?:the\s+)?correct\s+(?:option|answer)\s+(?:is|would be)\s+(\d+)",
        r"I\s+(?:choose|select|pick)\s+(?:option|answer)\s+(\d+)",
        r"(?:option|answer)\s+(\d+)\s+is\s+(?:the\s+)?(?:most\s+)?(?:correct|accurate|appropriate)",
        r"predicted_answer.*?(\d+)",
    ]

    for pattern in predict_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            prediction["predicted_answer"] = match.group(1)
            break

    # Extract reasoning
    reason_patterns = [
        r"(?:prediction_reasoning|reasoning for prediction|why.*predict)[\s:\"]*([^\"]*?)(?:\"|confidence|$)",
        r"(?:My reasoning|Reasoning|Here\'s my reasoning|Reason for prediction)(?:for this prediction)?(?:is|:)(.*?)(?:Confidence|In conclusion|To summarize|In summary|$)",
        r"(?:I predict|I believe|I think).*?because(.*?)(?:Confidence|In conclusion|To summarize|In summary|$)",
        r"(?:This option|Option \d+) is correct because(.*?)(?:Confidence|In conclusion|To summarize|In summary|$)",
    ]

    for pattern in reason_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match and match.group(1).strip():
            prediction["prediction_reasoning"] = (
                match.group(1).strip().replace('\\"', '"')
            )
            break

    # Extract confidence level
    confidence_patterns = [
        r"(?:confidence_level|confidence)[\s:\"]*([^\"]*?)(?:\"|$)",
        r"(?:My confidence|Confidence level|I am)(?:is|:)?\s+(high|medium|low)",
        r"I have\s+(high|medium|low)(?:\s+level of)?\s+confidence",
        r"(?:high|medium|low) confidence in (?:this|my) (?:prediction|answer|conclusion)",
    ]

    for pattern in confidence_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            prediction["confidence_level"] = (
                match.group(1).lower().strip().replace('"', "")
            )
            break

    # Extract confidence explanation
    explanation_patterns = [
        r"(?:confidence_explanation|Confidence explanation|Reason for confidence|Why I\'m confident)[\s:\"]*([^\"]*?)(?:\"|$)",
        r"(?:I\'m|I am) (?:highly|moderately|somewhat) confident because(.*?)(?:In conclusion|To summarize|In summary|$)",
        r"My confidence is (high|medium|low) because(.*?)(?:In conclusion|To summarize|In summary|$)",
    ]

    for pattern in explanation_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match and len(match.groups()) > 0:
            # Get the last group if there are multiple
            last_group = match.group(len(match.groups()))
            if last_group and last_group.strip():
                prediction["confidence_explanation"] = last_group.strip().replace(
                    '\\"', '"'
                )
                break

    return prediction


def generate_reasoning_trace(
    question_data: Dict[str, Any],
    client: OpenAI,
    model_name: str,
    specialty: str,
    reasoning_mode: str = "detailed",
) -> Dict[str, Any]:
    """
    Generate a reasoning trace for a multiple choice question using the specified mode.

    Args:
        question_data: Dictionary containing the question data
        client: OpenAI client
        model_name: The model name to use for generating the reasoning
        specialty: The scientific specialty persona to adopt
        reasoning_mode: The reasoning mode ("detailed", "focused", or "efficient")

    Returns:
        Dictionary with the reasoning trace
    """
    # Extract question components (without exposing correct answer)
    question_text = question_data.get("question", "")

    # Extract options from the question text
    options = extract_mc_options(question_text)

    if not options:
        return {
            "error": "Could not extract options from question",
            "reasoning_mode": reasoning_mode,
            "raw_question": question_text,
        }

    # Get the expert persona
    persona = get_expert_persona(specialty)

    # Generate the reasoning prompt based on the selected reasoning mode
    prompt = generate_reasoning_prompt(
        specialty, persona, question_text, options, reasoning_mode
    )

    try:
        # Create the completion request with timeout
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert {specialty} with deep knowledge in your field. You meticulously analyze questions using detailed reasoning and technical terminology appropriate to your domain. You express your thought process as a rich internal analysis, considering multiple angles, frameworks, and implications. After your analysis, you make a prediction about which answer is correct based on your expertise. IMPORTANT: Provide thorough reasoning for each option AND make a final prediction with your reasoning and confidence level.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,  # Lower temperature for more deterministic output
            max_tokens=4000,
            timeout=120,  # 2 minute timeout
        )

        # Extract the response
        response_text = response.choices[0].message.content.strip()

        # First try to parse as JSON directly
        try:
            json_content = json.loads(response_text)
            log_message(
                f"Successfully parsed {reasoning_mode} response as valid JSON",
                log_level="DEBUG",
            )
            json_content["reasoning_mode"] = reasoning_mode
            return json_content
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            code_block_match = re.search(
                r"```(?:json)?\s*(.*?)\s*```", response_text, re.DOTALL
            )
            if code_block_match:
                try:
                    json_content = json.loads(code_block_match.group(1))
                    log_message(
                        f"Successfully parsed {reasoning_mode} JSON from code block",
                        log_level="DEBUG",
                    )
                    json_content["reasoning_mode"] = reasoning_mode
                    return json_content
                except json.JSONDecodeError:
                    # Fallback to structured extraction from raw text
                    log_message(
                        f"JSON parsing failed for {reasoning_mode}, extracting structured data from raw text",
                        log_level="INFO",
                    )

                    # Extract thought process for each option
                    thought_process = extract_thought_process_from_text(
                        response_text, len(options)
                    )

                    # Extract prediction from text
                    prediction = extract_prediction_from_text(response_text)

                    # Create structured JSON based on reasoning mode
                    if reasoning_mode == "efficient":
                        json_content = {
                            "question_focus": "",
                            "option_analysis": thought_process,
                            "key_insights": "",
                            "prediction": prediction,
                            "extracted_from_text": True,
                            "reasoning_mode": reasoning_mode,
                        }
                    elif reasoning_mode == "focused":
                        json_content = {
                            "key_principle": "",
                            "option_analysis": thought_process,
                            "scientific_assessment": "",
                            "prediction": prediction,
                            "extracted_from_text": True,
                            "reasoning_mode": reasoning_mode,
                        }
                    else:  # detailed mode
                        json_content = {
                            "thought_process": thought_process,
                            "reasoning_summary": "",
                            "prediction": prediction,
                            "extracted_from_text": True,
                            "reasoning_mode": reasoning_mode,
                        }

                    return json_content
            else:
                # No code block found, extract structured data directly from raw text
                log_message(
                    f"No JSON code block found for {reasoning_mode}, extracting directly from text",
                    log_level="INFO",
                )

                # Extract thought process for each option
                thought_process = extract_thought_process_from_text(
                    response_text, len(options)
                )

                # Extract prediction from text
                prediction = extract_prediction_from_text(response_text)

                # Create structured JSON based on reasoning mode
                if reasoning_mode == "efficient":
                    json_content = {
                        "question_focus": "",
                        "option_analysis": thought_process,
                        "key_insights": "",
                        "prediction": prediction,
                        "extracted_from_text": True,
                        "reasoning_mode": reasoning_mode,
                    }
                elif reasoning_mode == "focused":
                    json_content = {
                        "key_principle": "",
                        "option_analysis": thought_process,
                        "scientific_assessment": "",
                        "prediction": prediction,
                        "extracted_from_text": True,
                        "reasoning_mode": reasoning_mode,
                    }
                else:  # detailed mode
                    json_content = {
                        "thought_process": thought_process,
                        "reasoning_summary": "",
                        "prediction": prediction,
                        "extracted_from_text": True,
                        "reasoning_mode": reasoning_mode,
                    }

                # If we couldn't extract meaningful structured data, save the raw text
                if not thought_process:
                    log_message(
                        f"Structured extraction failed for {reasoning_mode}, using raw text",
                        log_level="WARNING",
                    )
                    json_content = {
                        "thought_process": {},
                        "reasoning_summary": response_text,
                        "raw_text": response_text,
                        "extraction_failed": True,
                        "reasoning_mode": reasoning_mode,
                    }

                return json_content

    except Exception as e:
        error_msg = str(e)
        log_message(
            f"Error generating {reasoning_mode} reasoning trace: {error_msg}",
            log_level="ERROR",
        )

        return {
            "error": error_msg,
            "error_type": type(e).__name__,
            "reasoning_mode": reasoning_mode,
            "model_used": model_name,
            "query_successful": False,
        }


def process_question_multimode(
    question_data: Dict[str, Any],
    question_id: int,
    client: OpenAI,
    model_name: str,
    specialty: str,
    parallel_modes: bool = False,
) -> Dict[str, Any]:
    """
    Process a single question with all three reasoning modes.

    Args:
        question_data: Dictionary containing the question data
        question_id: Index of the question in the input file
        client: OpenAI client
        model_name: Model name to use
        specialty: Expert specialty
        parallel_modes: Whether to process modes in parallel

    Returns:
        Dictionary with question data and all reasoning traces
    """
    global _processed_questions

    start_time = time.time()

    # Extract basic question info
    question_text = question_data.get("question", "")
    text_field = question_data.get("text", "")
    question_type = question_data.get("type", "multiple-choice")
    correct_answer = question_data.get("answer", "")

    result = {
        "question_id": question_id,
        "question": question_text,
        "text": text_field,
        "type": question_type,
        "correct_answer": correct_answer,
        "reasoning_traces": {},
        "predictions": {},
        "processing_metadata": {
            "specialty_used": specialty,
            "modes_attempted": ["detailed", "focused", "efficient"],
            "modes_successful": [],
            "processing_errors": [],
            "processing_time_seconds": 0,
        },
    }

    modes = ["detailed", "focused", "efficient"]

    if parallel_modes:
        # Process modes in parallel
        log_message(
            f"Processing question {question_id} with parallel modes", log_level="DEBUG"
        )

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_mode = {
                executor.submit(
                    generate_reasoning_trace,
                    question_data,
                    client,
                    model_name,
                    specialty,
                    mode,
                ): mode
                for mode in modes
            }

            for future in as_completed(future_to_mode):
                mode = future_to_mode[future]
                try:
                    trace_result = future.result()

                    if not trace_result.get("error"):
                        # Extract prediction and store separately
                        prediction = trace_result.get("prediction", {})
                        if prediction:
                            result["predictions"][mode] = prediction

                        # Store reasoning trace without prediction
                        reasoning_trace = {
                            k: v for k, v in trace_result.items() if k != "prediction"
                        }
                        result["reasoning_traces"][mode] = reasoning_trace

                        result["processing_metadata"]["modes_successful"].append(mode)
                    else:
                        result["reasoning_traces"][mode] = trace_result
                        error_msg = f"{mode} mode failed: {trace_result.get('error', 'Unknown error')}"
                        result["processing_metadata"]["processing_errors"].append(
                            error_msg
                        )
                        log_message(error_msg, log_level="WARNING")

                except Exception as e:
                    error_msg = f"{mode} mode exception: {str(e)}"
                    result["processing_metadata"]["processing_errors"].append(error_msg)
                    result["reasoning_traces"][mode] = {
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "reasoning_mode": mode,
                    }
                    log_message(error_msg, log_level="ERROR")
    else:
        # Process modes sequentially
        log_message(
            f"Processing question {question_id} with sequential modes",
            log_level="DEBUG",
        )

        for mode in modes:
            try:
                trace_result = generate_reasoning_trace(
                    question_data, client, model_name, specialty, mode
                )

                if not trace_result.get("error"):
                    # Extract prediction and store separately
                    prediction = trace_result.get("prediction", {})
                    if prediction:
                        result["predictions"][mode] = prediction

                    # Store reasoning trace without prediction
                    reasoning_trace = {
                        k: v for k, v in trace_result.items() if k != "prediction"
                    }
                    result["reasoning_traces"][mode] = reasoning_trace

                    result["processing_metadata"]["modes_successful"].append(mode)
                    log_message(
                        f"Successfully completed {mode} mode for question {question_id}",
                        log_level="DEBUG",
                    )
                else:
                    result["reasoning_traces"][mode] = trace_result
                    error_msg = f"{mode} mode failed: {trace_result.get('error', 'Unknown error')}"
                    result["processing_metadata"]["processing_errors"].append(error_msg)
                    log_message(error_msg, log_level="WARNING")

            except Exception as e:
                error_msg = f"{mode} mode exception: {str(e)}"
                result["processing_metadata"]["processing_errors"].append(error_msg)
                result["reasoning_traces"][mode] = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "reasoning_mode": mode,
                }
                log_message(error_msg, log_level="ERROR")

    # Update processing metadata
    processing_time = time.time() - start_time
    result["processing_metadata"]["processing_time_seconds"] = processing_time

    # Update global progress
    _processed_questions += 1
    completion_percentage = (_processed_questions / _total_questions) * 100
    elapsed_time = time.time() - _start_time
    avg_time_per_question = (
        elapsed_time / _processed_questions if _processed_questions > 0 else 0
    )

    estimated_remaining = avg_time_per_question * (
        _total_questions - _processed_questions
    )
    if estimated_remaining < 60:
        eta = f"{estimated_remaining:.0f} seconds"
    elif estimated_remaining < 3600:
        eta = f"{estimated_remaining / 60:.1f} minutes"
    else:
        eta = f"{estimated_remaining / 3600:.1f} hours"

    successful_modes = len(result["processing_metadata"]["modes_successful"])
    log_message(
        f"Processed question {question_id}/{_total_questions} ({completion_percentage:.1f}%) - "
        f"{successful_modes}/3 modes successful - ETA: {eta}"
    )

    return result


def process_questions_multimode(
    questions: List[Dict[str, Any]],
    client: OpenAI,
    model_name: str,
    specialty: str,
    parallel_modes: bool = False,
    num_workers: int = 1,
    save_interval: int = 10,
    output_path: str = "",
) -> List[Dict[str, Any]]:
    """
    Process questions with multi-mode reasoning traces.

    Args:
        questions: List of question dictionaries
        client: OpenAI client
        model_name: Model name
        specialty: Specialty persona
        parallel_modes: Whether to process modes in parallel
        num_workers: Number of parallel workers for questions

    Returns:
        List of processed results
    """
    if num_workers == 1:
        # Sequential processing
        results = []
        for i, question in enumerate(
            tqdm(
                questions, desc=f"Generating {specialty}'s multi-mode reasoning traces"
            )
        ):
            result = process_question_multimode(
                question, i, client, model_name, specialty, parallel_modes
            )
            results.append(result)
        return results

    # Parallel processing of questions
    log_message(f"Using {num_workers} parallel workers for question processing")

    # Prepare arguments for parallel processing
    args_list = [
        (question, i, client, model_name, specialty, parallel_modes)
        for i, question in enumerate(questions)
    ]

    results = []
    failed_count = 0

    def process_question_wrapper(args_tuple):
        """Wrapper function for parallel processing."""
        try:
            question, question_id, client, model_name, specialty, parallel_modes = (
                args_tuple
            )
            result = process_question_multimode(
                question, question_id, client, model_name, specialty, parallel_modes
            )
            return {"success": True, "result": result, "index": question_id}
        except Exception as e:
            log_message(
                f"Error processing question {args_tuple[1]}: {e}", log_level="ERROR"
            )
            return {"success": False, "error": str(e), "index": args_tuple[1]}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_question_wrapper, args): i
            for i, args in enumerate(args_list)
        }

        # Collect results with progress tracking
        with tqdm(
            total=len(questions),
            desc=f"Generating {specialty}'s multi-mode reasoning traces (parallel)",
        ) as pbar:
            for future in as_completed(future_to_index):
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per question
                    if result["success"]:
                        results.append((result["index"], result["result"]))
                    else:
                        failed_count += 1
                        log_message(
                            f"Failed to process question {result['index']}: {result['error']}",
                            log_level="ERROR",
                        )
                        # Add empty result to maintain order
                        results.append((result["index"], None))
                except Exception as e:
                    failed_count += 1
                    log_message(
                        f"Exception processing question: {e}", log_level="ERROR"
                    )
                    results.append((len(results), None))

                pbar.update(1)
                pbar.set_postfix({"Failed": failed_count})

                # Save checkpoint every save_interval questions
                if len(results) % save_interval == 0:
                    # Sort current results and filter out None values for checkpoint
                    sorted_results = sorted(results, key=lambda x: x[0])
                    valid_results = [
                        result for _, result in sorted_results if result is not None
                    ]

                    # Only save if we have actual results
                    if valid_results:
                        try:
                            checkpoint_path = (
                                f"{os.path.splitext(output_path)[0]}_checkpoint.json"
                            )
                            with open(checkpoint_path, "w", encoding="utf-8") as f:
                                json.dump(valid_results, f, indent=2)
                            log_message(
                                f"Saved checkpoint with {len(valid_results)} questions to {checkpoint_path}",
                                log_level="INFO",
                            )
                        except Exception as e:
                            log_message(
                                f"Error saving checkpoint: {e}", log_level="ERROR"
                            )

    # Sort results by original order and filter out None values
    results.sort(key=lambda x: x[0])
    final_results = [result for _, result in results if result is not None]

    if failed_count > 0:
        log_message(
            f"Failed to process {failed_count} out of {len(questions)} questions",
            log_level="WARNING",
        )

    log_message(
        f"Successfully processed {len(final_results)} out of {len(questions)} questions using {num_workers} workers"
    )
    return final_results


def main():
    """Main entry point function."""
    global _total_questions, _processed_questions

    # Parse command-line arguments
    args = parse_arguments()

    # Configure the OpenAI API for the selected model
    client, model_name = configure_apis(args.model, args.config)

    # Initialize results list
    results = []

    # Check if we're continuing from a previous run
    starting_index = 0
    if args.continue_from and os.path.exists(args.continue_from):
        try:
            with open(args.continue_from, "r", encoding="utf-8") as f:
                results = json.load(f)
                starting_index = len(results)
                log_message(
                    f"Continuing from previous run - {starting_index} questions already processed",
                    log_level="INFO",
                )
        except Exception as e:
            log_message(f"Error reading continue-from file: {e}", log_level="ERROR")
            log_message("Starting from scratch", log_level="INFO")

    # Read the input file
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            questions_data = json.load(f)
    except Exception as e:
        log_message(f"Error reading input file: {e}", log_level="ERROR")
        sys.exit(1)

    # Process all questions (don't filter by type)
    questions = questions_data

    if not questions:
        log_message("No questions found in the input file.", log_level="ERROR")
        sys.exit(1)

    # Apply maximum questions limit if specified
    if args.max_questions is not None and args.max_questions > 0:
        questions = questions[: args.max_questions]

    # Skip questions we've already processed if continuing
    if starting_index > 0:
        questions = questions[starting_index:]

    _total_questions = len(questions) + starting_index
    _processed_questions = starting_index

    log_message(f"Found {len(questions)} questions to process.")
    log_message(f"Using {args.specialty} persona for reasoning")
    log_message("Processing modes: detailed, focused, efficient")
    if args.parallel_modes:
        log_message("Using parallel mode processing (faster but more API calls)")
    else:
        log_message("Using sequential mode processing")

    # Generate multi-mode reasoning traces
    new_results = process_questions_multimode(
        questions,
        client,
        model_name,
        args.specialty,
        args.parallel_modes,
        args.workers,
        args.save_interval,
        args.output,
    )

    # Add new results to existing results (if continuing)
    if starting_index > 0:
        results.extend(new_results)
    else:
        results = new_results

    # Save results periodically during processing
    for i, result in enumerate(new_results):
        current_index = starting_index + i + 1
        if (current_index % args.save_interval == 0) or (i == len(new_results) - 1):
            try:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                log_message(
                    f"Saved intermediate results to {args.output} after processing {current_index} questions",
                    log_level="INFO",
                )
            except Exception as e:
                log_message(
                    f"Error saving intermediate results: {e}", log_level="ERROR"
                )

    # Save final results
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        log_message(f"Successfully wrote multi-mode reasoning traces to {args.output}")
    except Exception as e:
        log_message(f"Error writing output file: {e}", log_level="ERROR")

    # Calculate and display summary statistics
    elapsed_time = time.time() - _start_time
    if elapsed_time < 60:
        time_str = f"{elapsed_time:.1f} seconds"
    elif elapsed_time < 3600:
        time_str = f"{elapsed_time / 60:.1f} minutes"
    else:
        time_str = f"{elapsed_time / 3600:.1f} hours"

    log_message(
        f"Processing complete! Generated {args.specialty}'s multi-mode reasoning traces for {_processed_questions} questions in {time_str}."
    )
    log_message(f"Results saved to {args.output}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("📈 MULTI-MODE PROCESSING SUMMARY")
    print("=" * 80)

    total_questions = len(results)
    modes = ["detailed", "focused", "efficient"]
    mode_stats = {mode: {"success": 0, "failed": 0} for mode in modes}
    questions_with_all_modes = 0
    questions_with_partial_modes = 0
    questions_with_no_modes = 0

    for result in results:
        successful_modes = result.get("processing_metadata", {}).get(
            "modes_successful", []
        )

        if len(successful_modes) == 3:
            questions_with_all_modes += 1
        elif len(successful_modes) > 0:
            questions_with_partial_modes += 1
        else:
            questions_with_no_modes += 1

        for mode in modes:
            if mode in successful_modes:
                mode_stats[mode]["success"] += 1
            else:
                mode_stats[mode]["failed"] += 1

    print(f"Total questions processed: {total_questions}")
    print(
        f"Questions with all 3 modes successful: {questions_with_all_modes} ({(questions_with_all_modes / total_questions) * 100:.1f}%)"
    )
    print(f"Questions with partial success: {questions_with_partial_modes}")
    print(f"Questions with no successful modes: {questions_with_no_modes}")

    print("\nMode-specific success rates:")
    for mode in modes:
        success_rate = (mode_stats[mode]["success"] / total_questions) * 100
        print(
            f"• {mode.capitalize()}: {mode_stats[mode]['success']}/{total_questions} successful ({success_rate:.1f}%)"
        )

    # Calculate prediction accuracy statistics
    print("\n📊 PREDICTION ACCURACY BY REASONING MODE:")
    print("=" * 80)

    for mode in modes:
        correct_predictions = 0
        total_predictions = 0

        for result in results:
            if mode in result.get("predictions", {}):
                total_predictions += 1
                prediction = result["predictions"][mode]
                predicted_answer = prediction.get("predicted_answer", "")
                correct_answer = result.get("correct_answer", "")

                # Simple accuracy check - extract number from both answers
                pred_match = re.search(r"\b(\d+)\b", str(predicted_answer))
                correct_match = re.search(r"\b(\d+)\b", str(correct_answer))

                if pred_match and correct_match:
                    if pred_match.group(1) == correct_match.group(1):
                        correct_predictions += 1

        if total_predictions > 0:
            accuracy = (correct_predictions / total_predictions) * 100
            print(
                f"• {mode.capitalize()}: {correct_predictions}/{total_predictions} correct ({accuracy:.1f}%)"
            )
        else:
            print(f"• {mode.capitalize()}: No valid predictions")

    print(f"\nSpecialty used: {args.specialty}")
    print(f"Model used: {args.model}")
    print(f"Parallel modes: {'Enabled' if args.parallel_modes else 'Disabled'}")
    print(f"Workers: {args.workers}")
    print("=" * 80)


if __name__ == "__main__":
    main()
