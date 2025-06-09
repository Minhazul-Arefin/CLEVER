# CLEVER
A Python toolkit to parse E3SM Fortran code into a Neo4j knowledge graph of files, equations, variables, and constants, with built-in LLM-powered querying and function approximation.




## Files in this Repository:

* `README.md`: This file.
* `math.txt`: (Expected output) A text file containing generated mathematical expressions.
* `math_txt_generator.py`: A Python script to generate mathematical expressions and save them to `math.txt`.
* `sci-kg_constructor.py`: A Python script intended to construct a scientific knowledge graph. (Further details on its input/output would go here if known, e.g., "This script takes CSV files as input and loads them into a Neo4j database.")
* `symbolic_math_to_csv.py`: A Python script to parse the `math.txt` file and convert its contents into a CSV format suitable for a graph database like Neo4j.

## How to Run Sequentially:

Follow these steps to run the scripts in the correct order:

1.  **Generate Mathematical Expressions:**
    * First, you need to create the `math.txt` file which will contain the mathematical expressions.
    * Run the `math_txt_generator.py` script:
        ```bash
        python math_txt_generator.py
        ```
    * This will generate (or overwrite) the `math.txt` file in your directory.

2.  **Convert Mathematical Expressions to CSV:**
    * Next, use the `symbolic_math_to_csv.py` script to parse the `math.txt` and prepare it for your graph database.
    * Run the script:
        ```bash
        python symbolic_math_to_csv.py
        ```
    * This script is expected to generate one or more `.csv` files (e.g., `nodes.csv`, `relationships.csv`) which will be used for the knowledge graph.

3.  **Construct the Scientific Knowledge Graph:**
    * Finally, run the `sci-kg_constructor.py` script to take the generated CSV files and load them into your scientific knowledge graph (e.g., a Neo4j database).
    * Run the script:
        ```bash
        python sci-kg_constructor.py
        ```
    * **Note:** Ensure your Neo4j database (or the graph database you are using) is running and accessible before proceeding to this step. Refer to the comments or documentation within `sci-kg_constructor.py` for specific details on database connections or requirements.
