# files handlers
import pandas as pd
import os 
def read_csv(path, words_column=None, definitions_column=None, **kwargs):
    """
    Load a CSV file, validate its existence, and rearrange columns to match "words" and "definitions" format.

    Args:
        path (str): The path to the CSV file.
        words_column (str): The name of the column containing words.
        definitions_column (str): The name of the column containing definitions.
        **kwargs: Additional keyword arguments to pass to the `pandas.read_csv` function.

    Returns:
        pd.DataFrame: The loaded DataFrame. #change this to our class
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"The file '{path}' does not exist.")
    
    data = pd.read_csv(path, **kwargs)

    # Check if both columns are passed
    if words_column is not None and definitions_column is not None:
        if words_column not in data.columns or definitions_column not in data.columns:
            raise ValueError(f"Both of the given columns ('{words_column}' and '{definitions_column}' ) must be present in the CSV file.")
        else:
            # Rearrange columns to match "words" and "definitions" format
            return data[[words_column, definitions_column]].rename(columns={words_column: 'Words', definitions_column: 'Definitions'})
    else:
        # Ensure the file has at least two columns
        if len(data.columns) < 2:
            raise ValueError("The file must have at least two columns to infer words and definitions.")
        
        # Take a random sample of 5 rows
        sample = data.sample(n=5)
        
        # Compute the average text length for each column
        avg_lengths = sample.applymap(str).apply(lambda col: col.str.len().mean())
        
        # Assume the column with the longer average length is the definition column
        sorted_columns = avg_lengths.sort_values(ascending=False).index
        inferred_definitions_col = sorted_columns[0]
        inferred_words_col = sorted_columns[1]
        
        return data[[inferred_words_col, inferred_definitions_col]].rename(
            columns={inferred_words_col: 'Words', inferred_definitions_col: 'Definitions'}
        )
