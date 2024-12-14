#!/usr/bin/env python3
import os
import warnings
warnings.filterwarnings("ignore")

hidden = 		[64]
num_layers = 	[1]
partitions = 	[1500]

bitwidth = 2

dataset = [
        ( 'Proteins'                 , 29     , 2) ,   
		( 'artist'                 	 , 100	  , 12),
		( 'soc-BlogCatalog'	     	 , 128	  , 39),    
]

os.system("touch res_QGTC_batched_GIN_{}bit.log".format(bitwidth))


for n_Layer in num_layers:
	for hid in hidden:
		for data, d, c in dataset:
			for p in partitions:
				command = "python main_qgtc.py \
    						--gpu 0 \
							--dataset {} \
       						--dim {} \
                            --n-hidden {} \
                            --n-classes {} \
							--psize {}\
							--use_QGTC \
           					--bit_width {}\
							--run_GIN >> res_QGTC_batched_GIN_{}bit.log".\
							format(data, d, c, hid, p, bitwidth, bitwidth)		
				os.system(command)
				print()
 
os.system("python main_qgtc.py --gpu 0 --dataset ppi --use_QGTC --run_GIN --bit_width {0} >> res_QGTC_batched_GIN_{0}bit.log".format(bitwidth))
print()
os.system("python main_qgtc.py --gpu 0 --dataset ogbn-arxiv --use_QGTC --run_GIN --bit_width {0} >> res_QGTC_batched_GIN_{0}bit.log".format(bitwidth))
print()
os.system("python main_qgtc.py --gpu 0 --dataset ogbn-products --use_QGTC --run_GIN --bit_width {0} >> res_QGTC_batched_GIN_{0}bit.log".format(bitwidth))
print()
os.system("./parse_time.py res_QGTC_batched_GIN_{0}bit.log > res_QGTC_batched_GIN_{0}bit.csv".format(bitwidth))
if not os.path.exists("logs"):
	os.system("mkdir logs/")
os.system("mv *.log logs/")