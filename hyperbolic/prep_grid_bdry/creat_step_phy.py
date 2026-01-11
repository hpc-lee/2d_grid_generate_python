import numpy as np

# due to z axis up, step file start from bottom
# k=1 => bottom, k=nz => top
nz = 601
num_of_step = nz - 1
flag_gradient = 0
file_name = './step_file_2d.txt'

dh = -50 # A minus sign indicates the negative direction of the axis.
step = np.zeros(num_of_step)
if flag_gradient:
    # Gradient Grid
    incre_layer = 8
    max_ratio = 2
    incre_ratio = np.exp((np.log(max_ratio) / incre_layer))

    step[:10] = dh/2

    start_value = step[9]
    geom_seq = start_value * (incre_ratio ** np.arange(1, incre_layer + 1))
    step[10:10 + incre_layer] = geom_seq

    last_value = step[10 + incre_layer - 1]
    step[10 + incre_layer:] = last_value
else:
    step[:] = dh




with open(file_name, 'w') as fid:
    fid.write(f"# number of step, num_of_step is {num_of_step}\n")
    for i in range(num_of_step):
        fid.write(f'{step[i]:.9e} \n')

print(f"nz is {nz}")
print(f"num_of_step is {num_of_step}")
print("Script execution completed.")
