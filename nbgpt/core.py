import logging
import sys
import os
from dotenv import load_dotenv
from openai import OpenAI
import nbformat
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()


def get_logger(name=None, log_to_console=False, log_file="nbgpt.log"):
    """
    Configures and returns a logger instance.

    Args:
        name (str, optional): Name for the logger. Defaults to the module's __name__ if not provided.
        log_to_console (bool): Whether to enable console logging. Defaults to False.
        log_file (str): Path to the log file. Defaults to "app.log".

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Use module name or the provided name for the logger
    logger = logging.getLogger(name or __name__)

    # Avoid duplicate handlers
    if logger.hasHandlers():
        return logger

    # Determine log level based on DEBUG environment variable
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Default: File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Optional: Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


# Initialize logger
logger = get_logger()


def check_required_vars():
    """
    Checks if the required variables are set.
    """
    required_vars = ["OPENAI_API_KEY"]
    for var in required_vars:
        if not os.getenv(var):
            raise EnvironmentError(f"Required environment variable {var} is missing.")


# Check required environment variables
check_required_vars()


def get_openai_client():
    """
    Returns an instance of the OpenAI client.
    """
    return OpenAI()


# Initialize OpenAI client
client = get_openai_client()


def get_nbcell_list(
    notebook_path: str,
    cell_types=["code", "markdown"],
    include_output: bool = False,
):
    """
    Reads a Jupyter notebook and returns a list of cells of specified types.
    """
    logger.info(f"Reading notebook from '{notebook_path}'")
    cells_list = []
    try:
        with open(notebook_path, "r", encoding="utf-8") as file:
            notebook = nbformat.read(file, as_version=4)

        filtered_cells = [
            cell for cell in notebook.cells if cell.cell_type in cell_types
        ]

        for cell in filtered_cells:
            cell_data = {}
            cell_data["type"] = cell.cell_type
            cell_data["source"] = cell["source"]
            if include_output and cell.cell_type == "code":
                cell_data["outputs"] = cell.get("outputs", [])
            cells_list.append(cell_data)

        if not filtered_cells:
            logger.info("No cells of specified types found.")

    except FileNotFoundError:
        logger.error(f"The file '{notebook_path}' was not found.")
    except nbformat.reader.NotJSONError:
        logger.error(f"The file '{notebook_path}' is not a valid JSON notebook.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

    return cells_list


def call_openai_llm(system_prompt, user_prompt, message, model="gpt-4o"):
    """
    Calls the OpenAI language model with the provided user prompt.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join([user_prompt, message])},
    ]
    try:
        completion = client.chat.completions.create(
            model=model, messages=messages, max_tokens=1500
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"An error occurred while calling OpenAI LLM: {e}")
        raise


def generate_analysis(system_prompt, user_prompt, nbcell_list):
    """
    Generates an analysis of the notebook cells using OpenAI LLM.
    """
    logger.info("Generating analysis for notebook cells.")
    try:
        message = "\n".join([cell["source"] for cell in nbcell_list])
        analysis = call_openai_llm(system_prompt, user_prompt, message)
        logger.info(f"Analysis generated successfully.\n{analysis}\n")
        return analysis
    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}")
        raise


def generate_improved_nbcell(
    system_prompt, user_prompt, nbcell_list, llm_analysis, nbcell
):
    """
    Generates an improved version of a notebook cell based on LLM analysis.
    """
    try:
        message = "\n".join(
            [
                f"ORIGINAL_NOTEBOOK: {nbcell_list}",
                f"ANALYSIS: {llm_analysis}",
                f"CELL: {nbcell}",
            ]
        )
        improved_cell = call_openai_llm(system_prompt, user_prompt, message)
        return improved_cell
    except Exception as e:
        logger.error(f"An error occurred during improvement: {e}")
        raise


def generate_translated_nbcell(
    system_prompt, user_prompt, nbcell_list, language, nbcell
):
    """
    Translates a notebook cell to the specified language using OpenAI LLM.
    """
    try:
        message = "\n".join(
            [
                f"ORIGINAL: {nbcell_list}",
                f"TARGET_LANGUAGE: {language}",
                f"CELL: {nbcell}",
            ]
        )
        translated_cell = call_openai_llm(system_prompt, user_prompt, message)
        return translated_cell
    except Exception as e:
        logger.error(f"An error occurred during translation: {e}")
        raise


def generate_improved_nb(
    system_prompt, analysis_prompt, improvement_prompt, nb_path, cell_types=["markdown"]
):
    """
    Generates an improved version of the notebook by iterating through its cells.
    """
    logger.info(f"Generating improved notebook for '{nb_path}'")
    nbcell_list = get_nbcell_list(nb_path)
    llm_analysis = generate_analysis(system_prompt, analysis_prompt, nbcell_list)
    nbcell_list_improved = []
    for cell in tqdm(nbcell_list, desc="Generating iterative improvements"):
        if cell["type"] not in cell_types:
            nbcell_list_improved.append(cell)
            continue
        improved_cell = generate_improved_nbcell(
            system_prompt, improvement_prompt, nbcell_list, llm_analysis, cell["source"]
        )
        new_cell = cell.copy()
        new_cell["source"] = improved_cell
        nbcell_list_improved.append(new_cell)
    logger.info("Improved notebook cells generated successfully.")
    return nbcell_list_improved


def generate_translated_nb(
    system_prompt,
    translation_prompt,
    nb_path,
    language="ro",
    cell_types=["markdown", "code"],
):
    """
    Generates a translated version of the notebook by iterating through its cells.
    """
    logger.info(f"Generating translated notebook for '{nb_path}'")
    nbcell_list = get_nbcell_list(nb_path)
    nbcell_list_translated = []
    for cell in tqdm(nbcell_list, desc="Generating iterative translations"):
        if cell["type"] not in cell_types:
            nbcell_list_translated.append(cell)
            continue
        translated_cell = generate_translated_nbcell(
            system_prompt, translation_prompt, nbcell_list, language, cell["source"]
        )
        new_cell = cell.copy()
        new_cell["source"] = translated_cell
        nbcell_list_translated.append(new_cell)
    logger.info("Translated notebook cells generated successfully.")
    return nbcell_list_translated


def save_new_nb(nbcell_list, output_path):
    """
    Saves the modified notebook cells to a new notebook file.
    """
    logger.info(f"Saving new notebook to '{output_path}'")
    try:
        new_nb = nbformat.v4.new_notebook()
        for cell in nbcell_list:
            new_cell = nbformat.v4.new_code_cell(cell["source"])
            new_cell.cell_type = cell["type"]
            new_nb.cells.append(new_cell)
        with open(output_path, "w", encoding="utf-8") as file:
            nbformat.write(new_nb, file)
        logger.info(f"New notebook saved to '{output_path}'")
    except Exception as e:
        logger.error(f"An error occurred while saving the new notebook: {e}")
        raise
