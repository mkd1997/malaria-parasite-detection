#! /bin/bash

for f in dir*;
do
	python3 get_hist.py $f;	
	echo "$f done"
done
