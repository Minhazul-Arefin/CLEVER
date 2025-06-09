import os
import requests

# Define folder paths
fortran_folder_path = "/home/minhaz/Desktop/Code/GraphRAG/Working F90"
txt_output_path = "math.txt"

# LM Studio API endpoint
API_URL = "http://localhost:1235/v1/chat/completions"

# Function to read Fortran files
def read_fortran_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()

# Function to send Fortran code to LLaMA
def chat_with_llama(user_message):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "local-model",
        "messages": [
            {
                "role": "system",
                "content": "You are an AI that extracts variables, constants, and equations from Fortran code."
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        "max_tokens": 10000
    }

    response = requests.post(API_URL, headers=headers, json=data)

    try:
        response_json = response.json()
        return response_json["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {e}\nResponse Text: {response.text}"

# Collect all .f90 and .F90 files from the folder (including subdirectories)
fortran_files = [
    os.path.join(root, file)
    for root, _, files in os.walk(fortran_folder_path)
    for file in files if file.endswith(('.f90', '.F90'))
]

print(f"Total Fortran files found: {len(fortran_files)}")
processed_files = 0

# Open the text file for writing
with open(txt_output_path, "w", encoding="utf-8") as txt_file:
    txt_file.write("===== Extracted Global/Local Variables, Constants, and Equations =====\n\n")

    for file_path in fortran_files:
        file_name = os.path.basename(file_path)
        processed_files += 1
        print(f"Processing {processed_files}: {file_name}")

        # Read file content
        fortran_code = read_fortran_file(file_path)

        # Final Prompt
        prompt = (
            f"From the Fortran file `{file_name}`, extract the following in a clear and structured way:\n\n"
            "1. Global Variables: variables defined at module or program level\n"
            "2. Local Variables: variables defined inside subroutines or functions\n"
            "3. Constants: named constants used in equations\n"
            "4. Equations: write the full mathematical expression in human-readable format\n\n"
            "Use this exact output format:\n\n"
            "### Global Variables:\n"
            "VariableName (symbol)\n"
            "...\n\n"
            "### Local Variables:\n"
            "VariableName (symbol)\n"
            "...\n\n"
            "### Constants:\n"
            "ConstantName (symbol)\n"
            "...\n\n"
            "### Equations:\n"
            "EquationName (equation)\n"
            "...\n\n"
            "Here is the Fortran code:\n\n"
            f"{fortran_code}"
        )

        # Call the LLM
        response = chat_with_llama(prompt)

        if not response.strip():
            print(f" No data extracted for: {file_name}")
            continue

        print(f"Extracted data for: {file_name}")

        # Write to file
        txt_file.write(f"===== File: {file_name} =====\n\n")
        txt_file.write(response)
        txt_file.write("\n" + "="*80 + "\n\n")

print(f"\nProcessed {processed_files} files.")
print(f"TXT file successfully created: {txt_output_path}")

