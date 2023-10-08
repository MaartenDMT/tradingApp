import glob
import logging

import pandas as pd


def next_available_filename(base_filename, extension, directory='data/'):
    directory = f"{directory}{extension}"
    existing_files = glob.glob(f"{directory}/{base_filename}*.{extension}")
    max_number = 0
    for file in existing_files:
        try:
            file_number = int(file.split(base_filename)[
                              1].split(f".{extension}")[0])
            if file_number > max_number:
                max_number = file_number
        except ValueError:
            print(f"Error in the value for {extension}")

    next_number = max_number + 1
    next_filename = f"{directory}/{base_filename}{next_number}.{extension}"
    return next_filename


# Set up root logger
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create separate loggers for Environment and Agent
env_logger = logging.getLogger('Environment')
agent_logger = logging.getLogger('Agent')
main_logger = logging.getLogger('Main')


def env_logging():

    # Optionally, you can customize them further, e.g., set different log levels
    env_logger.setLevel(logging.DEBUG)

    # Optionally, set up file handlers to log to separate files
    env_handler = logging.FileHandler('data/logs/environment.log')

    # Format the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    env_handler.setFormatter(formatter)

    # Add the handlers
    env_logger.addHandler(env_handler)

    return env_logger


def agent_logging():

    # Optionally, you can customize them further, e.g., set different log levels
    agent_logger.setLevel(logging.DEBUG)

    # Optionally, set up file handlers to log to separate files
    agent_handler = logging.FileHandler('data/logs/agent.log')

    # Format the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    agent_handler.setFormatter(formatter)

    # Add the handlers
    agent_logger .addHandler(agent_handler)

    return agent_logger


def main_logging():

    # Optionally, you can customize them further, e.g., set different log levels
    main_logger.setLevel(logging.INFO)

    # Optionally, set up file handlers to log to separate files
    main_handler = logging.FileHandler('data/logs/main.log')

    # Format the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main_handler.setFormatter(formatter)

    # Add the handlers
    main_logger.addHandler(main_handler)

    return main_logger


def best_csv(b_reward, acc) -> bool:
    filename = "data/best_model/best_model.csv"
    if glob.glob(filename):
        df = pd.read_csv(filename)
        csv_reward = df['reward']
        csv_accuracy = df["accuracy"]
        if b_reward > csv_reward and acc > csv_accuracy:
            dict_ar = {'accuracy': acc, 'reward': b_reward}
            # save the model and overwrite the csv
            df = pd.DataFrame(dict_ar)
            df.to_csv(filename)
            return True
        return False
    return True
