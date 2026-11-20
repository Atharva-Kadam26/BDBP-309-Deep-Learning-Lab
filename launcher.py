
import os
from pathlib import Path
import pyperclip

# Main project directory
base_folder = Path(__file__).resolve().parent

# Find all Python files in existing lab folders
python_files = []

for root, dirs, files in os.walk(base_folder):

    # Skip unnecessary directories
    dirs[:] = [
        d for d in dirs
        if d not in [".venv", "__pycache__", ".git", "data"]
    ]

    for file in files:
        if file.endswith(".py") and file != "launcher.py":
            python_files.append(Path(root) / file)

# Sort files alphabetically
python_files.sort()

print("\n========== DEEP LEARNING LAB LAUNCHER ==========")

# Search for a lab or program
search = input("\nEnter lab name or keyword: ").lower().strip()

# Find matching files
matches = []

for file in python_files:
    relative_path = str(file.relative_to(base_folder)).lower()

    if search in relative_path:
        matches.append(file)

# Display matching files
if len(matches) == 0:

    print("\nNo matching Python files found!")

else:

    print("\nMatching programs:\n")

    for i, file in enumerate(matches):
        print(str(i + 1) + ". " +
              str(file.relative_to(base_folder)))

    # Select the required program
    choice = int(input("\nEnter program number: "))

    if choice < 1 or choice > len(matches):
        print("Invalid selection!")

    else:

        selected_file = matches[choice - 1]

        # Read complete source code
        with open(selected_file, "r", encoding="utf-8") as f:
            code = f.read()

        # Copy complete code to clipboard
        pyperclip.copy(code)

        print("\n========================================")
        print("SUCCESS! Complete code copied.")
        print("File:", selected_file.name)
        print("========================================")
        print("\nOpen your answer.py and press CTRL + V.")
