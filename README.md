# Python repository to reproduce figures in  Badman+ "Prediction and Verification of Parker Solar Probe Solar Wind Sources at 13.3 $𝑅_\odot$"

This repository contains :

* README.md - this readme
* SourcesPaperFigures.ipynb - a jupyter notebook which runs through each figure, downloading the data it needs along the way. "Fig#.png" files will appear in the directory.
* helpers.py - useful python utility functions used in the main notebook
* CSV/ - a directory containing archived CSV files from the PSP footpoint prediction campaigns which are the subject of this work
* submodules 
    * kent_distirbution - forked from https://github.com/edfraenkel/kent_distribution and updated to work in python 3.0. This module can fit a Kent distribution (https://en.wikipedia.org/wiki/Kent_distribution) to an input set of points on the unit sphere.
    * solarsynoptic - cloned from https://github.com/dstansby/solarsynoptic. This module downloads EUV full disk images of the Sun, reprojects them into the Carrington frame (longitude vs latitude bins) and can combine multiple days of images into a full-sun map.
