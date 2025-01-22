#!/bin/bash

data="lra-news1"

for i in $data
do
	for j in {3000..5000..1000}
	do
		echo "python main_learnable.py --mode train --task $i --random 1001 --name learnable --dsteps $j"
		python main_learnable.py --mode train --task $i --random 1001 --name learnable --dsteps $j
	done
done
