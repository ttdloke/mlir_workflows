#!/bin/zsh

read "F?Enter 1 for running simple_ops.py or 2 for running torch_nn_ops.py: "

if [[ $F == "1" ]]; then
    FILE=simple_ops.py
else
    FILE=torch_nn_ops.py
fi

python $FILE

read "ANSWER?Save output and generate python file?(y/n) "

# if [[ $ANSWER == "y" ]]; then
#     read "NAME?Enter name of file: "

#     python $FILE > /Users/dloke/Documents/tt/tt-mlir/test/ttmlir/$NAME.mlir

#     cat $FILE > $NAME.py

#     echo "Generated mlir file at /Users/dloke/Documents/tt/tt-mlir/test/ttmlir/$NAME.mlir"

#     echo "Generated copy of python script at $PWD/$NAME.py"
# fi

