import os
from typing import List, Optional

import pandas as pd
import tensorflow as tf

MODEL_PATH = 'data/saved_model'
CSV_PATH = "data/best_model/best_model.csv"


def read_csv_file(path: str) -> pd.DataFrame:
    """Read a CSV file and return its content as a DataFrame."""
    return pd.read_csv(path)


def get_known_models() -> List[str]:
    """Return a list of known models."""
    return ['Standard_Model', 'Dense_Model', 'LSTM_Model', 'CONV1D_LSTM_Model']


def get_max_accuracy(df: pd.DataFrame, model_name: str) -> Optional[float]:
    """Return the maximum accuracy for a given model from the DataFrame."""
    if 'model_name' not in df.columns:
        raise ValueError(
            f"Expected 'model_name' column in CSV but it was not found! {df.columns.tolist()}")

    # Return None if the model name is not one of the known models.
    if model_name not in get_known_models():
        return None

    filtered_df = df[df['model_name'] == model_name]

    # Return None if there are no records for the specified model_name
    if filtered_df.empty:
        return None

    return filtered_df['accuracy'].max()


def is_new_record(b_reward, acc, modelname) -> bool:
    """Check if the given reward is a new record for the specified model."""
    if not os.path.isfile(CSV_PATH):
        return True  # If file doesn't exist, then it's a new record by default

    df = read_csv_file(CSV_PATH)
    max_acc = get_max_accuracy(df, modelname)

    if max_acc is None:
        return True  # Treat None as if no record exists for the model or the modelname is unknown

    return acc > max_acc


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
    self.model.save(f"{MODEL_PATH}/{self.modelname}_model.keras")


#####

def multi_head_attention(d_model, num_heads):
    depth = d_model // num_heads
    attention_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=depth)

    def attention(x):
        q = tf.keras.layers.Dense(d_model)(x)
        k = tf.keras.layers.Dense(d_model)(x)
        v = tf.keras.layers.Dense(d_model)(x)

        output = attention_layer(query=q, key=k, value=v)
        return output

    return attention


def transformer_block(d_model, num_heads, dropout):
    inputs = tf.keras.layers.Input(shape=(None, d_model))

    attention_output = multi_head_attention(
        d_model, num_heads)(inputs)
    attention_output = tf.keras.layers.Dropout(dropout)(attention_output)
    out1 = tf.keras.layers.LayerNormalization()(inputs + attention_output)

    ffn_output = tf.keras.layers.Dense(d_model, activation='relu')(out1)
    ffn_output = tf.keras.layers.Dense(d_model)(ffn_output)
    out2 = tf.keras.layers.LayerNormalization()(out1 + ffn_output)

    return tf.keras.Model(inputs=inputs, outputs=out2)
