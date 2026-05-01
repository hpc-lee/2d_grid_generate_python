import numpy as np

# due to z axis up, step file start from bottom
# k=1 => bottom, k=nz => top
nz = 601
num_of_step = nz - 1
flag_gradient = 0
# if dense layer from top, need flip, because k=nz => top
flag_flip = 1  
file_name = './arc_len_file1.txt'

step = np.zeros(num_of_step)
arc_len = np.zeros(nz)
if flag_gradient:
    # Gradient Grid
    incre_layer = 12
    max_ratio = 3
    incre_ratio = np.exp((np.log(max_ratio) / incre_layer))

    step[:10] = 1

    start_value = step[9]
    geom_seq = start_value * (incre_ratio ** np.arange(1, incre_layer + 1))
    step[10:10 + incre_layer] = geom_seq

    last_value = step[10 + incre_layer - 1]
    step[10 + incre_layer:] = last_value

    if flag_flip:
        step = np.flip(step)
else:
    step[:] = 1

sum_step = np.sum(step)

step_nor = step / sum_step

for i in range(1,nz):
    arc_len[i] = arc_len[i-1] + step_nor[i-1]

if (arc_len[nz-1] - 1) > 1e-8:
    raise ValueError("step set is error, please check and reset")

with open(file_name, 'w') as fid:
    fid.write(f"# number of nz is {nz}\n")
    for i in range(nz):
        fid.write(f'{arc_len[i]:.9e} \n')

print(f"nz is {nz}")
print("Script execution completed.")
