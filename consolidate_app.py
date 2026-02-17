import subprocess
import os

# Function to extract a file from a feature branch without switching branches

def extract_file_from_branch(branch_name, file_path):
    # Ensure the file path is absolute
    if not os.path.isabs(file_path):
        raise ValueError("The file path must be absolute.")

    try:
        # Use git show to extract the file from the specified branch
        command = ["git", "show", f'{branch_name}:{file_path}']
        with open(os.path.basename(file_path), 'wb') as f:
            subprocess.run(command, stdout=f, check=True)
        print(f"Successfully extracted {file_path} from {branch_name}.")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting file: {e}")

# Example usage:
# extract_file_from_branch('feature/my-feature', '/path/to/my_file.txt')
