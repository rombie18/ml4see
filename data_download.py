import json
import inquirer
import os

DATA_FOLDER_PATH = "data_"
DATA_SUMMARY_PATH = "data_summary.json"


def format_file_size(size_in_bytes, decimal_places=2):
    '''
    Convert a size in bytes to a human-readable string with optional decimal places.

    Args:
        size_in_bytes (int): Size in bytes to be converted.
        decimal_places (int, optional): Number of decimal places (default is 2).

    Returns:
        str: Human-readable size with appropriate units (B, KB, MB, GB, etc.).
    '''
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
    size = size_in_bytes
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.{decimal_places}f} {unit}"
        size /= 1024


def main():
    """
    Download selected runs of data based on user input.

    This function reads data summary information from a JSON file,
    prompts the user to select runs for download, calculates total size that will be used,
    and then uses 'curl' to download the selected runs of data in parallel.

    Args:
        None

    Returns:
        None
    """
    runs = []
    with open(DATA_SUMMARY_PATH, 'r', encoding="utf-8") as file:
        runs = json.load(file)

    choices = [
        f"{run['name']} ({format_file_size(int(run['size']))})" for run in runs]
    answers = inquirer.prompt([
        inquirer.Checkbox(
            'select_runs', message="What runs would you like to download?", choices=choices),
    ])

    selected_run_names = [selected_run.split()[0]
                          for selected_run in answers["select_runs"]]
    selected_runs = [run for run in runs if run["name"] in selected_run_names]
    total_selected_size = sum(int(run["size"]) for run in selected_runs)

    confirmation_message = f"You will download {format_file_size(total_selected_size)} of data. Do you want to proceed?"
    answers = inquirer.prompt(
        [inquirer.Confirm('confirm_size', message=confirmation_message)])

    if not answers["confirm_size"]:
        print("Download cancelled.")
        return

    command = "curl -Z --create-dirs " + \
        " ".join(
            [f"{selected_run['url']} -o {DATA_FOLDER_PATH}/{selected_run['name']}.tar" for selected_run in selected_runs])
    os.system(command)


main()
