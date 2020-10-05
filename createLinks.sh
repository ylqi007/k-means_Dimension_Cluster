#!/usr/bin/env bash

# Destination directory of datasets
DATA=/home/ylqi007/work/PyCharm_Projects/DATA

# ln -s [OPTIONS] FILE LINK
ln -s ${DATA} .


# ln -s [OPTIONS] FILE LINK
# `ln` is a command-line utility for creating links between files.
# By default, the ln command creates hard links.
# To create a symbolic link, use the `-s` (`--symbolic`) option.
#
# The `ln` command syntax for creating symbolic links is as follows:
# ln -s [OPTIONS] FILE LINK
# 1. If both the `FILE` and `LINK` are given, `ln` will create a link from the file specified as the first argument (`FILE`)
#  to the file specified as the second argument (`LINK`).
# 2. If only one file is given as an argument or the second argument is a dot (`.`), ln will create a link to that file in
#  the current working directory. The name of the symlink will be the same as the name of the file it points to.
# 3. By default, on success, ln doesn’t produce any output and returns zero.
