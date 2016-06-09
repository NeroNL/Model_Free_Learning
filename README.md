For this final project, we work on the reinforcement learning with the unknow possibility model, and we compare the result from between the model-base algorithm and model-free algorithm. 

Running this model:
	1. one terminal runs "roscore"
	2. the other terminal get into the directory root of catkin, and run "roslaunch cse_190_assi_3 solution_python.launch"

Changing variable:
	1. Most variables come from the json files, especially the learning rate. 
	2. If you want to test how train time affect the result, you need to go into mfmdp.py to change the iteration_size and max_iteration_size, which handles the iteration of single grid and whole map respectively.