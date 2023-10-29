import os

import pandas as pd

MODEL_PATH = 'data/saved_model/dql_model.keras'
CSV_PATH = "data/best_model/best_model.csv"


def is_new_record(b_reward, acc):
    """Check if the given reward and accuracy are a new record."""
    if os.path.isfile(CSV_PATH):
        return acc > 0.55
    return True  # If file doesn't exist, then it's a new record by default


def save_to_csv(b_reward, acc, model_params):
    """Append the new record to the CSV, then prune if necessary."""
    new_record_data = {
        'accuracy': [acc],
        'reward': [b_reward]
    }
    # Add model parameters to the record data
    for param_name, param_value in model_params.items():
        new_record_data[param_name] = [param_value]

    new_record = pd.DataFrame(new_record_data)

    if os.path.isfile(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df = pd.concat([df, new_record], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)
    else:
        new_record.to_csv(CSV_PATH, index=False)


def save_model(self):
    """Save the current model."""
    self.model.save(MODEL_PATH)
