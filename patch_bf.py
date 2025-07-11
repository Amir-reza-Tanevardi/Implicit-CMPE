import os
import importlib.util

##### Fix to bayesflow helper_functions.py file #####
spec = importlib.util.find_spec("bayesflow")
if spec and spec.origin:
    bayesflow_root = os.path.dirname(spec.origin)  # path to bayesflow package
    target_file = os.path.join(bayesflow_root, "helper_functions.py")  # adjust as needed

    # Example: patch the file
    with open(target_file, "r") as f:
        lines = f.readlines()

    # Replace deprecated code or make other fixes
    lines = [line.replace("optimizer.lr", "optimizer.learning_rate") for line in lines]  # example fix

    with open(target_file, "w") as f:
        f.writelines(lines)

    print(f"Patched {target_file}")
else:
    print("Could not locate bayesflow.")
#####################################################