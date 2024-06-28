#!/bin/zsh

read "F?Enter 1 for running simple_ops.py or 2 for running torch_nn_ops.py: "

# Add in support for different dialects ... 

# read "DIALECT?Which MLIR dialect would you like to generate (stablehlo, tosa, linalg) ? "

if [[ $F == "1" ]]; then
    FILE=simple_ops.py
else
    FILE=torch_nn_ops.py
fi

python $FILE

read "ANSWER?Save output and generate python file?(y/n) "

if [[ $ANSWER == "y" ]]; then
    read "NAME?Enter name of file: "

    MLIR_FILE_LOC=../mlir_files/stablehlo
    python $FILE > $MLIR_FILE_LOC/$NAME.mlir

    cat $FILE > torch_scripts/$NAME.py

    echo "Generated mlir file at $MLIR_FILE_LOC/$NAME.mlir"

    echo "Generated copy of python script at torch_scripts/$NAME.py"
fi

