from core import get_logger
import yaml
import argparse

logger = get_logger()


def get_default_config():
    llm_system_prompt = """
        The following is a list of the most common programming languages:
        """
    llm_user_prompt = """
        List the most common programming languages.
        """

    return {
        "llm_system_prompt": llm_system_prompt,
        "llm_user_prompt": llm_user_prompt,
    }


def load_config(file_path="config.yml"):
    try:
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
            logger.info("Configuration loaded from file:")
            return config
    except FileNotFoundError:
        logger.error("Configuration file not found. Using default values.")
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}. Using default values.")
    return get_default_config()


def parse_arguments():
    parser = argparse.ArgumentParser(description="CLI for nbgpt tasks")
    parser.add_argument(
        "--config", default="config.yml", help="Path to the configuration file"
    )
    parser.add_argument(
        "--command", choices=["improve", "translate"], help="Command to execute"
    )
    parser.add_argument(
        "--language", default="ro", help="Target language for translation"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = load_config(args.config)  # uses selected config file
    if args.command == "improve":
        print("Running improve command...")
    elif args.command == "translate":
        print(f"Running translate command in {args.language}...")


if __name__ == "__main__":
    main()
