from core import get_logger, generate_improved_nb, generate_translated_nb, save_new_nb
import yaml
import argparse
import sys
from datetime import datetime

# Initialize logger
logger = get_logger()


# Load configuration from a YAML file
def load_config(file_path="config.yml"):
    try:
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from file: {file_path}")
            logger.debug(f"Configuration: {config}")
            return config
    except FileNotFoundError:
        logger.error("Configuration file not found. Using default values.")
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}. Using default values.")
        raise


# Retrieve a value from a nested dictionary using a list of keys
def get_config_value(config, *keys):
    """
    Retrieve a value from a nested dictionary using a list of keys.

    :param config: The configuration dictionary.
    :param keys: A list of keys representing the path to the desired value.
    :return: The value from the configuration dictionary.
    :raises KeyError: If any key in the path is not found.
    """
    try:
        value = config
        for key in keys:
            value = value[key]
        return value
    except KeyError as e:
        logger.error(f"Key not found in configuration: {e}")
        raise


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
    system_prompt = get_config_value(config, "system_prompt")
    analysis_prompt = get_config_value(config, "improve", "prompts", "analysis")
    improvement_prompt = get_config_value(config, "improve", "prompts", "improvement")
    improved_nb = generate_improved_nb(
        system_prompt, analysis_prompt, improvement_prompt, nb_path
    )
    new_nb_path = get_new_nb_path(nb_path, "improved")
    save_new_nb(improved_nb, new_nb_path)
    logger.info(f"Saving improved notebook to '{new_nb_path}'")


# Create a translated version of the notebook
def make_translated_nb(config, language, nb_path):
    system_prompt = get_config_value(config, "system_prompt")
    translation_prompt = get_config_value(config, "translate", "prompt")
    translated_nb = generate_translated_nb(
        system_prompt, translation_prompt, nb_path, language
    )
    new_nb_path = get_new_nb_path(nb_path, f"translated_{language.lower()}")
    save_new_nb(translated_nb, new_nb_path)
    logger.info(f"Saving translated notebook to '{new_nb_path}'")


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
