from neo4j import GraphDatabase
import pandas as pd

# Neo4j connection details
uri = "bolt://localhost:7687"
user = "neo4j"
password = "lFz_YZwb8Ht8D65WJT_lFExW9HDzBGPeNgqQovarK5k"

# Load your CSV file
csv_path = "E3SM.sif default node.csv"
df = pd.read_csv(csv_path).fillna('')  # Replace NaNs with empty string

# Connect to Neo4j
driver = GraphDatabase.driver(uri, auth=(user, password))

def upload_to_neo4j(tx, system, file, equation, variable, constant):
    # Create system node
    if system:
        tx.run("MERGE (s:System {name: $system})", system=system)

    # Create file node and relationship
    if file:
        tx.run("""
            MERGE (s:System {name: $system})
            MERGE (f:File {name: $file})
            MERGE (s)-[:hasFile]->(f)
        """, system=system, file=file)

    # Create equation node and relationship
    if equation:
        tx.run("""
            MERGE (f:File {name: $file})
            MERGE (e:Equation {name: $equation})
            MERGE (f)-[:hasEquation]->(e)
        """, file=file, equation=equation)

    # Create variable node and relationship
    if variable:
        tx.run("""
            MERGE (e:Equation {name: $equation})
            MERGE (v:Variable {name: $variable})
            MERGE (e)-[:hasVariable]->(v)
        """, equation=equation, variable=variable)

    # Create constant node and relationship
    if constant:
        tx.run("""
            MERGE (e:Equation {name: $equation})
            MERGE (c:Constant {name: $constant})
            MERGE (e)-[:hasConstant]->(c)
        """, equation=equation, constant=constant)

# Ingest data
with driver.session() as session:
    for _, row in df.iterrows():
        session.write_transaction(
            upload_to_neo4j,
            row['System'],
            row['File'],
            row['Equation'],
            row['Variable'],
            row['Constant']
        )

driver.close()

