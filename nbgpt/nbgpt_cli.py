from core import (
    get_logger,
    generate_improved_nb,
    generate_translated_nb,
    save_new_nb,
    load_config,
    get_config_value,
)
import argparse
import sys
from datetime import datetime

# Initialize logger
logger = get_logger()


# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="CLI for nbgpt tasks")
    parser.add_argument(
        "--config", default="config.yml", help="Path to the configuration file"
    )
    parser.add_argument(
        "--task",
        choices=["improve", "translate"],
        default="improve",
        help="Command to execute",
    )
    parser.add_argument(
        "--language", default="romanian", help="Target language for translation"
    )
    parser.add_argument("nb_path", help="Path to the notebook file to process")
    return parser.parse_args()


# Generate a new notebook path with a suffix and timestamp
def get_new_nb_path(nb_path, suffix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return nb_path.replace(".ipynb", f"_{suffix}_{timestamp}.ipynb")


# Create an improved version of the notebook
def make_improved_nb(config, nb_path):
    print(f"Improving notebook '{nb_path}' ...")
    system_prompt = get_config_value(config, "system_prompt")
    analysis_prompt = get_config_value(config, "improve", "prompts", "analysis")
    improvement_prompt = get_config_value(config, "improve", "prompts", "improvement")
    improved_nb = generate_improved_nb(
        system_prompt, analysis_prompt, improvement_prompt, nb_path
    )
    new_nb_path = get_new_nb_path(nb_path, "improved")
    save_new_nb(improved_nb, new_nb_path)
    logger_msg = f"Saved improved notebook to '{new_nb_path}'"
    logger.info(logger_msg)
    print(logger_msg)


# Create a translated version of the notebook
def make_translated_nb(config, language, nb_path):
    print(f"Translating notebook '{nb_path}' to '{language}' ...")
    system_prompt = get_config_value(config, "system_prompt")
    translation_prompt = get_config_value(config, "translate", "prompt")
    translated_nb = generate_translated_nb(
        system_prompt, translation_prompt, nb_path, language
    )
    new_nb_path = get_new_nb_path(nb_path, f"translated_{language.lower()}")
    save_new_nb(translated_nb, new_nb_path)
    logger_msg = f"Saved translated notebook to '{new_nb_path}'"
    logger.info(logger_msg)
    print(logger_msg)


# Main function to execute the appropriate task based on arguments
def main():
    args = parse_arguments()
    config = load_config(args.config)  # uses selected config file
    if args.task == "improve":
        make_improved_nb(config, args.nb_path)
    elif args.task == "translate":
        language = args.language.capitalize()
        make_translated_nb(config, language, args.nb_path)


# Entry point of the script
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}.")
        sys.exit(1)
