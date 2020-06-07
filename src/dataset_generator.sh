#!/bin/bash

#VARIABLES
ASSETS_FOLDER="../assets"
MANEUVER=$1
MAIN=$(find . -name main.py)

#for all maneuver in dataset folder
for i in $(ls -d $ASSETS_FOLDER/$MANEUVER*);
do
	echo "----------EXEC - $i----------"
	VAR="$MAIN --parse --maneuver $MANEUVER --video-path-list /$i/front.mp4;/$i/left.mp4;/$i/rear.mp4;/$i/right.mp4"
	python $VAR
done
