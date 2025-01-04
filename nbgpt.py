import nbformat
import sys
from openai import OpenAI
import argparse
from tqdm import tqdm
import logging

# Initialize OpenAI client
client = OpenAI()


def configure_logging():
    """
    Configures the logging module to display INFO level messages.
    """
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


configure_logging()

logger = logging.getLogger(__name__)


def call_openai_llm(*llm_user_prompt):
    model = "gpt-4o"
    llm_prompt_system = """
    You are a world class, academic peer reviewer that acts as a helpful assistant for improving Jupyter notebooks developed for young students, containing both code and markdown cells.
    The resulting notebook should be more readable, maintainable, and efficient. You are expected to provide constructive feedback and suggestions for improvement.
    You can also suggest additional code snippets, explanations, or visualizations to enhance the notebook. Please ensure that the notebook is still functional after your changes.
    """
    messages = [
        {"role": "system", "content": llm_prompt_system},
        {"role": "user", "content": "\n".join(llm_user_prompt)},
    ]
    try:
        completion = client.chat.completions.create(
            model=model, messages=messages, max_tokens=1500
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"An error occurred while calling OpenAI LLM: {e}")
        raise


def get_nbcell_list(
    notebook_path: str,
    cell_types=["code", "markdown"],
    include_output: bool = False,
):
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


def generate_analysis(nbcell_list):
    logger.info("Generating analysis for notebook cells.")
    llm_prompt = "Analyze the following Python notebook and suggest improvements that would take it to the next level:"
    try:
        combined_content = "\n".join(cell["source"] for cell in nbcell_list)
        analysis = call_openai_llm(llm_prompt, combined_content)
        logger.info("Analysis generated successfully.")
        return analysis
    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}")
        raise


def generate_improved_nbcell(nbcell_list, llm_analysis, nbcell):
    llm_prompt = """
    Significantly adjust the following Python notebook CELL, part of the ORIGINAL_NOTEBOOK, following the improvements suggested by the ANALYSIS.
    Make sure that you strictly reply with the adjusted content only, *without any leading markers*.
    When using LaTeX, you must only use single dollar signs ($) to wrap the LaTeX content.
    """
    try:
        improved_cell = call_openai_llm(
            llm_prompt,
            f"ORIGINAL_NOTEBOOK: {nbcell_list}",
            f"ANALYSIS: {llm_analysis}",
            f"CELL: {nbcell}",
        )
        return improved_cell
    except Exception as e:
        logger.error(f"An error occurred during improvement: {e}")
        raise


def generate_translated_nbcell(nbcell_list, language, nbcell):
    llm_prompt = """
    Translate the following Python notebook CELL to the specified TARGET_LANGUAGE.
    Make sure that you strictly reply with the translated content only, *without any leading markers* (such as ```markdown or ```python).
    When using LaTeX, you must only use single dollar signs ($) to wrap the LaTeX content.
    If the cell contains code, stricly translate the comments only, without modifying the code itself.
    """
    try:
        translated_cell = call_openai_llm(
            llm_prompt,
            f"ORIGINAL: {nbcell_list}",
            f"TARGET_LANGUAGE: {language}",
            f"CELL: {nbcell}",
        )
        return translated_cell
    except Exception as e:
        logger.error(f"An error occurred during translation: {e}")
        raise


def generate_improved_nb(nb_path, cell_types=["markdown"]):
    logger.info(f"Generating improved notebook for '{nb_path}'")
    nbcell_list = get_nbcell_list(nb_path)
    llm_analysis = generate_analysis(nbcell_list)
    nbcell_list_improved = []
    for cell in tqdm(nbcell_list, desc="Generating iterative improvements"):
        if cell["type"] not in cell_types:
            nbcell_list_improved.append(cell)
            continue
        improved_cell = generate_improved_nbcell(
            nbcell_list, llm_analysis, cell["source"]
        )
        new_cell = cell.copy()
        new_cell["source"] = improved_cell
        nbcell_list_improved.append(new_cell)
    logger.info("Improved notebook cells generated successfully.")
    return nbcell_list_improved


def generate_translated_nb(nb_path, language="ro", cell_types=["markdown", "code"]):
    logger.info(f"Generating translated notebook for '{nb_path}'")
    nbcell_list = get_nbcell_list(nb_path)
    nbcell_list_translated = []
    for cell in tqdm(nbcell_list, desc="Generating iterative translations"):
        if cell["type"] not in cell_types:
            nbcell_list_translated.append(cell)
            continue
        translated_cell = generate_translated_nbcell(
            nbcell_list, language, cell["source"]
        )
        new_cell = cell.copy()
        new_cell["source"] = translated_cell
        nbcell_list_translated.append(new_cell)
    logger.info("Translated notebook cells generated successfully.")
    return nbcell_list_translated


def save_new_nb(nbcell_list, output_path):
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


nb_path = "perceptron.ipynb"
nb_improved_path = nb_path.replace(".ipynb", "_improved.ipynb")
nb_translated_path = nb_path.replace(".ipynb", "_translated.ipynb")
nbcell_list_improved = generate_improved_nb(nb_path)
save_new_nb(nbcell_list_improved, nb_improved_path)
nbcell_list_translated = generate_translated_nb(nb_improved_path)
save_new_nb(nbcell_list_translated, nb_translated_path)
