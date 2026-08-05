#!/usr/bin/env python3
"""
Script to execute the fault diagnosis demo notebook
"""

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

def run_notebook(notebook_path, output_path=None):
    """
    Execute a Jupyter notebook

    Args:
        notebook_path: Path to input notebook
        output_path: Path to save executed notebook (optional)
    """
    print(f"Loading notebook: {notebook_path}")

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    print("Executing notebook...")
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    try:
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
        print("✓ Notebook executed successfully")

        if output_path:
            with open(output_path, 'w') as f:
                nbformat.write(nb, f)
            print(f"✓ Executed notebook saved to: {output_path}")

        return True

    except Exception as e:
        print(f"✗ Error executing notebook: {e}")
        return False

if __name__ == "__main__":
    notebook_file = "fault_diagnosis_demo.ipynb"
    output_file = "fault_diagnosis_demo_executed.ipynb"

    success = run_notebook(notebook_file, output_file)

    if success:
        print("\n" + "="*70)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nGenerated files:")
        print("  - fault_diagnosis_demo_executed.ipynb (executed notebook)")
        print("  - demo1_actuator_fault.pdf")
        print("  - demo2_sensor_fault.pdf")
        print("  - demo3_combined_faults.pdf")
        print("  - demo_summary.pdf")
        print("\nTo view the notebook:")
        print("  jupyter notebook fault_diagnosis_demo_executed.ipynb")
    else:
        print("\n✗ Demo failed")
