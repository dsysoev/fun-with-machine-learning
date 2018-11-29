#!/usr/bin/env bash

if [ -d data-cifar10 ]; then
    echo "data-cifar10 directory already present, exiting"
    exit 1
fi

mkdir data-cifar10
wget --directory-prefix=data-cifar10 https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
cd data-cifar10
tar -xvzf cifar-10-python.tar.gz
