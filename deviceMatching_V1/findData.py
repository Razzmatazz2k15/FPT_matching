import os

def find_file(filename, data_directory):
    for root, _, files in os.walk(data_directory):
        if filename in files:
           
            return os.path.join(root, filename)
    return None

def find_parquet_file(filename, data_directory):
    """
    Searches for a Parquet "file" (directory or file) in a given directory.

    Args:
        filename (str): The name of the Parquet "file" (directory).
        data_directory (str): The root directory to search in.

    Returns:
        str: The full path to the Parquet directory or None if not found.
    """
    filename = filename.lower().strip()  # Normalize the filename
    for root, dirs, files in os.walk(data_directory):  # Include directories
        # Check if any directory matches the filename
        for dir_name in dirs:
            if dir_name.lower() == filename:
                print(f"Found Parquet directory: {dir_name}")
                return os.path.join(root, dir_name)

        # Additionally, check if there are .parquet files (optional)
        for file_name in files:
            if file_name.lower() == filename:
                print(f"Found Parquet file: {file_name}")
                return os.path.join(root, file_name)

    # No match found
    return None

if __name__ == "__main__":
    # Testing find_file
    print("Testing find_file...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a dummy file in the temporary directory
        test_file_name = "test.txt"
        test_file_path = os.path.join(tmp_dir, test_file_name)
        with open(test_file_path, "w") as f:
            f.write("dummy content")
        
        # Test: find the file that exists
        found_path = find_file(test_file_name, tmp_dir)
        print(f"find_file: Expected to find {test_file_name}, got: {found_path}")

        # Test: try to find a file that does not exist
        not_found = find_file("nonexistent.txt", tmp_dir)
        print(f"find_file: Expected None for nonexistent.txt, got: {not_found}")

    # Testing find_parquet_file
    print("\nTesting find_parquet_file...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a directory named "data.parquet"
        parquet_dir_name = "data.parquet"
        parquet_dir_path = os.path.join(tmp_dir, parquet_dir_name)
        os.mkdir(parquet_dir_path)

        # Create a subdirectory and a dummy parquet file inside it
        sub_dir = os.path.join(tmp_dir, "subfolder")
        os.mkdir(sub_dir)
        parquet_file_name = "datafile.parquet"
        parquet_file_path = os.path.join(sub_dir, parquet_file_name)
        with open(parquet_file_path, "w") as f:
            f.write("dummy parquet content")

        # Test: find the Parquet directory by name
        found_dir = find_parquet_file(parquet_dir_name, tmp_dir)
        print(f"find_parquet_file: Expected to find directory {parquet_dir_name}, got: {found_dir}")

        # Test: find the Parquet file by name
        found_file = find_parquet_file(parquet_file_name, tmp_dir)
        print(f"find_parquet_file: Expected to find file {parquet_file_name}, got: {found_file}")

        # Test: search for a non-existent parquet name
        not_found_parquet = find_parquet_file("nonexistent.parquet", tmp_dir)
        print(f"find_parquet_file: Expected None for nonexistent.parquet, got: {not_found_parquet}")

    print("\nAll tests completed successfully!")