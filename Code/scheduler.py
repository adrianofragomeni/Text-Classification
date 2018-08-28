import subprocess
import pprint
import keras,sys
import tensorflow as tf
import numpy as np
from time import sleep
import os

def get_gpu_usage():
	sp = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

	out_str = sp.communicate()
	out_list = out_str[0].decode("utf-8").split('\n')

	out_dict = {}

	for item in out_list:
		try:
			key, val = item.split(':')
			key, val = key.strip(), val.strip()
			if key in out_dict:
				out_dict[key+'_1'] = val
			else: out_dict[key] = val
		except:
			pass
	return out_dict

	
def set_gpu():		
	iteration = 1
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	# The GPU id to use, usually either "0" or "1"
	
	while 1:
		out_dict = get_gpu_usage()
		if float(out_dict['Gpu'][:-1]) <5:
			os.environ["CUDA_VISIBLE_DEVICES"]="0"

			print('\nUsing Gpu:0')
			break
		elif float(out_dict['Gpu_1'][:-1]) <5:
			os.environ["CUDA_VISIBLE_DEVICES"]="1"

			print('\nUsing Gpu:1')
			break
		else:
			print('All gpu are used... Let\'s try again in %i minutes' %(180/iteration/60.0))
			sys.stdout.flush()
			sleep(180/iteration)
		iteration +=1
		
	#sess = tf.Session(config=config) 
	#keras.backend.set_session(sess)
	print('\n')
	print('-'*32)
	print('Running the main')
	print('-'*32)
	sys.stdout.flush()
	sleep(2)
	os.system('timeout 900 python main.py')
	print('\n Time\'s Over!\n')
	
set_gpu()
