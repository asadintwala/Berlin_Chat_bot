import pandas as pd

def load_data(file_path):
    """Loads data from Excel or CSV."""
    if file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use .xlsx or .csv.")