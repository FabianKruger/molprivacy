import pandas as pd
from moreno.config import Config


def save_csv(data: pd.DataFrame, filename: str):
    """Saving a pandas dataframe to a csv file in the data directory.

    Args:
        data (pd.DataFrame): data to save
        filename (str): name of the created file
    """
    data_dir = Config.get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    file_path = data_dir / filename

    data.to_csv(file_path, index=False)

    print(f"File saved to {file_path}")
