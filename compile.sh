#!/bin/bash

# build cuda-api-wrappers

if [ ! -d cuda-api-wrappers ]
then
	git clone https://github.com/eyalroz/cuda-api-wrappers
	cp -r cuda-api-wrappers/scripts .
fi

# build cuda-api-wrappers

if [ ! -d "cuda-api-wrappers/lib" ] 
then
	cd cuda-api-wrappers
	cmake .
	make
	cd ..
fi

# build project 
rm -R build
mkdir build
cd build
cmake ..
make