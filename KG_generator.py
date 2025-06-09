import re
import csv

input_path = "math.txt"
output_path = "cytoscape_kg_edges.csv"

with open(input_path, "r", encoding="utf-8") as f:
    content = f.read()

file_blocks = re.split(r"===== File: (.*?) =====", content)[1:]
edges = []
all_variables = {}   # symbol -> (name, source_file)
all_constants = {}   # symbol -> (name, source_file)
all_equations = []   # (equation_text, source_file)

total_files = len(file_blocks) // 2
print(f"Detected {total_files} Fortran files for processing.")

# First pass: extract entities
for i in range(0, len(file_blocks), 2):
    filename = file_blocks[i].strip()
    block = file_blocks[i + 1]
    print(f"Processing file: {filename} ({i // 2 + 1}/{total_files})")

    # Extract variables
    var_section = re.search(r"### Variables:(.*?)(?=###|={5,})", block, re.DOTALL)
    if var_section:
        lines = [v.strip() for v in var_section.group(1).splitlines() if v.strip()]
        for line in lines:
            match = re.match(r"(.+?)\s*\((.+?)\)", line)
            if match:
                name, symbol = match.groups()
                symbol = symbol.strip()
                all_variables[symbol] = (name.strip(), filename)
                edges.append((filename, symbol, "contains"))

    # Extract constants (explicit)
    const_section = re.search(r"### Constants:(.*?)(?=###|={5,})", block, re.DOTALL)
    if const_section:
        lines = [c.strip() for c in const_section.group(1).splitlines() if c.strip()]
        for line in lines:
            match = re.match(r"(.+?)\s*\((.+?)\)", line)
            if match:
                name, symbol = match.groups()
                symbol = symbol.strip()
                all_constants[symbol] = (name.strip(), filename)
                edges.append((filename, symbol, "contains"))

    # Extract equations
    eqn_section = re.search(r"### Equations:(.*?)(?=={5,})", block, re.DOTALL)
    if eqn_section:
        lines = [e.strip() for e in eqn_section.group(1).splitlines() if e.strip()]
        for eqn in lines:
            if "=" in eqn:
                eqn_clean = eqn.strip()
                all_equations.append((eqn_clean, filename))
                edges.append((filename, eqn_clean, "encodes"))

print(f"Finished entity extraction.")
print(f"  Variables: {len(all_variables)}")
print(f"  Constants: {len(all_constants)}")
print(f"  Equations: {len(all_equations)}")

# Infer constants from equations
for eqn_text, _ in all_equations:
    inferred = re.findall(r"\b([A-ZΔΩΓθμρσλφψζτνπ]{1,5})\b", eqn_text)
    for symbol in inferred:
        if symbol not in all_variables and symbol not in all_constants:
            all_constants[symbol] = ("(inferred)", "inferred_file")
            edges.append(("inferred_file", symbol, "contains"))

# Link variables/constants to equations (cross-file)
for idx, (eqn_text, eqn_file) in enumerate(all_equations):
    if idx % max(1, len(all_equations) // 10) == 0:
        pct = (idx / len(all_equations)) * 100
        print(f"Linking progress: {pct:.1f}%")

    for symbol, (name, origin_file) in all_variables.items():
        if re.search(rf"\b{re.escape(symbol)}\b", eqn_text):
            edges.append((origin_file, symbol, "contains"))
            edges.append((symbol, eqn_text, "hasVariable"))

    for symbol, (name, origin_file) in all_constants.items():
        if re.search(rf"\b{re.escape(symbol)}\b", eqn_text):
            edges.append((origin_file, symbol, "contains"))
            edges.append((symbol, eqn_text, "hasConstant"))

# Add E3SM root node → all .F90 files
file_nodes = set()
for src, tgt, rel in edges:
    if src.endswith(".F90"):
        file_nodes.add(src)
    if tgt.endswith(".F90"):
        file_nodes.add(tgt)

for file_node in sorted(file_nodes):
    edges.append(("E3SM", file_node, "has"))

# Write final edge list
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Source", "Target", "Relationship"])
    for src, tgt, rel in edges:
        writer.writerow([src, tgt, rel])

print(f"Completed. Final edge list saved to: {output_path}")
