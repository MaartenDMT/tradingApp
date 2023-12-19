import logging
from logging.handlers import RotatingFileHandler


def setup_loggers():
    log_configurations = [
        {'name': 'autobot', 'file': 'data/logs/autobot.log'},
        {'name': 'tradex', 'file': 'data/logs/tradex.log'},
        {'name': 'presenter', 'file': 'data/logs/presenter.log'},
        {'name': 'model', 'file': 'data/logs/model.log'},
        {'name': 'view', 'file': 'data/logs/view.log'},
        {'name': 'rl_trading', 'file': 'data/logs/rl_trading.log'},
        {'name': 'app', 'file': 'data/logs/app.log'},
        {'name': 'rl', 'file': 'data/logs/rl.log'},
        {'name': 'env', 'file': 'data/logs/env.log'},
        {'name': 'agent', 'file': 'data/logs/agent.log'},
        {'name': 'manual', 'file': 'data/logs/manual.log'},
    ]

    loggers = {}

    for config in log_configurations:
        logger = setup_individual_logger(config['name'], config['file'])
        loggers[config['name']] = logger

    return loggers  # Now, you can get any logger instance by its name from this dictionary


def setup_individual_logger(logger_name, log_file):
    logger = logging.getLogger(logger_name)

    if not logger.hasHandlers():  # To ensure that handlers don't get added every time we call this function
        logger.setLevel(logging.DEBUG)

        file_handler = RotatingFileHandler(
            log_file, maxBytes=2_000_000_000, backupCount=10)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger
