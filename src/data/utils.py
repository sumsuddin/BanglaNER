import os
import shutil

def delete_files_and_directories(directory_path):
    try:
        for root, _, files in os.walk(directory_path, topdown=False):
            # Delete all files in the current directory
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)

            # Delete all directories in the current directory
            for directory in os.listdir(root):
                directory_path = os.path.join(root, directory)
                if os.path.isdir(directory_path):
                    shutil.rmtree(directory_path)

        print("All files and directories have been deleted successfully.")
    except Exception as e:
        print(f"Error deleting files and directories: {e}")