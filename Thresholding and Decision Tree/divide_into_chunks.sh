#! /bin/bash
cd ./Parasitized/;
i=0; 
for f in *; 
do 
    d=dir_$(printf %03d $((i/100+1))); 
    mkdir -p $d; 
    mv "$f" $d; 
    i=$[$i+1]; 
done
