#!/bin/bash

#data="lra-news1 lra-yelp1 lra-image lra-nature lra-listops lra-retrieval"
data="lra-nature"
for i in $data
do
	# for j in {1..10..1}
	for j in 1.3 1.5 1.9 2.0 2.5 3.0
	do
		echo "python main_learnable_skewness.py --mode train --task $i --random 7 --name learnable --sk $j"
		python main_learnable_skewness.py --mode train --task $i --random 7 --name learnable --sk $j
	done
done
