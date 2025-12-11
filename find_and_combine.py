import os
import pandas as pd
import re

def find_and_combine_csv_files(root_dir, file_pattern):
    """
    Search for CSV files matching file_pattern in subfolders of root_dir,
    combine them into a single CSV file sorted by time(unix).

    Args:
        root_dir (str): Root directory containing subfolders like Kraken_OHLCVT_Q[1-4]_[2023-2025]
        file_pattern (str): Pattern for CSV files, e.g., 'ETHUSD_1'
    """
    # Define the regex pattern for subfolders (Kraken_OHLCVT_Q[1-4]_[2023-2025])
    folder_pattern = re.compile(r'Kraken_OHLCVT_Q[1-4]_(202[3-5])$')

    # Expected columns
    expected_columns = ['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']

    # List to store DataFrames
    all_data = []

    # Walk through the directory
    for root, dirs, files in os.walk(root_dir):
        # Check if the current directory matches the folder pattern
        if folder_pattern.search(os.path.basename(root)):
            # Look for the specific CSV file
            for file in files:
                if file == f"{file_pattern}.csv":  # Exact match for file name
                    file_path = os.path.join(root, file)
                    print(f"Found file: {file_path}")
                    # Read the CSV file
                    try:
                        df = pd.read_csv(file_path, header=None, names=expected_columns)
                        # Verify the number of columns
                        if df.shape[1] == len(expected_columns):
                            all_data.append(df)
                        else:
                            print(f"Warning: {file_path} has unexpected number of columns: {df.shape[1]}")
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

    if not all_data:
        print(f"No files found matching {file_pattern}.csv in the specified directories.")
        return

    # Combine all DataFrames
    combined_df = pd.concat(all_data, ignore_index=True)

    # Sort by time(unix)
    combined_df.sort_values(by='time', inplace=True)

    # Remove duplicates based on time(unix), keeping the first occurrence
    combined_df.drop_duplicates(subset='time', keep='first', inplace=True)

    # Define output file path
    output_file = os.path.join(root_dir, f"{file_pattern}.csv")

    # Save to CSV without index, ensuring correct columns
    combined_df.to_csv(output_file, index=False, header=True)
    print(f"Combined data saved to {output_file}")
    print(f"Total rows: {len(combined_df)}")

def main():
    # Specify the root directory
    root_directory = input("Enter the root directory path: ")

    # Specify the file pattern (e.g., 'ETHUSD_1')
    file_pattern = input("Enter the file pattern (e.g., ETHUSD_1): ")

    # Validate root directory
    if not os.path.isdir(root_directory):
        print(f"Error: {root_directory} is not a valid directory.")
        return

    # Call the function to find and combine CSV files
    find_and_combine_csv_files(root_directory, file_pattern)

if __name__ == "__main__":
    main()
