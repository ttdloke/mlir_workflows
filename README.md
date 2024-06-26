# mlir_workflows

Repo for scripts, notebooks and anything else related to generating mlir files that will be used downstream by Tenstorrent's MLIR project

Note: These workflows can also be seen as different potential entrypoints into the Tenstorrent ecosystem and can be polished up and published for users

# Structure

mlir_files subdirectory is where all generated mlir files live. All mlir files end in .mlir regardless of dialect so for the ease of finding what dialect file you want and being able to compare the same model in different dialects each subdirectory in mlir_files is a dialect name and the files inside are of that dialect
