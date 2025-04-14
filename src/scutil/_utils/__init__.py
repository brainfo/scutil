"""
Utility functions for eyeball-sc
"""
import os

def check_workdir(workdir):
    ## check if current path is the working directory
    if os.getcwd() != workdir:
        print(f"The current path is not the working directory {workdir}.")
        os.chdir(workdir)
        print(f"Changed the current path to {workdir}.")
    return workdir


