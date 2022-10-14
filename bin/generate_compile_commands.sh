#!/bin/bash

BASEDIR=$(dirname "$0")

# Command to generate the compilation database file.
cmake -S $BASEDIR/../ -B $BASEDIR/../build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Location of the compilation database file.
outfile="$BASEDIR/../build/compile_commands.json"

# TODO: do we still need this with cmake 
# Command to replace the marker for exec_root in the file.
#execroot=$(bazel info execution_root)
#sed -i.bak "s@__EXEC_ROOT__@${execroot}@" "${outfile}"
#sed -i '' -e 's/-fno-canonical-system-headers/ /g' "${outfile}"

rm -f "$BASEDIR/../compile_commands.json"

# Trick clang-tidy into thinking anything all includes are system headers
# This is kinda ugly but it works, maybe theres a better way in cmake, see below,
# but it is kind of non standard with pybind11.
# See https://stackoverflow.com/questions/46638293/ignore-system-headers-in-clang-tidy
sed 's/-I/-isystem /g' ${outfile} > "$BASEDIR/../compile_commands.json" 

