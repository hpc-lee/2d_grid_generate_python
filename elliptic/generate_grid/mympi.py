from mpi4py import MPI
import numpy as np
from grid_data import GridData


class MyMPI:
    def __init__(self):
        self.nprocx = 0
        self.nprocz = 0
        self.myid = 0
        self.comm = None          # communicator
        self.topocomm = None      # Cartesian topology communicator
        self.topoid = np.array([0, 0], dtype=np.int32)      # 2D coordinates: [x, z]
        self.neighid = np.array([0, 0, 0, 0], dtype=np.int32)  # [left, right, down, up]


def mympi_set(cfgs: dict, myid: int, comm: MPI.Comm) -> MyMPI:
    """
    Initialize MPI 2D Cartesian topology and neighbor info.
    """
    mympi = MyMPI()
    mympi.nprocx = cfgs['number_of_mpiprocs_x']
    mympi.nprocz = cfgs['number_of_mpiprocs_z']
    mympi.myid = myid
    mympi.comm = comm

    # Create 2D Cartesian topology
    pdims = [mympi.nprocx, mympi.nprocz]
    periods = [False, False]  # non-periodic in both directions

    mympi.topocomm = comm.Create_cart(dims=pdims, periods=periods, reorder=False)

    # Get local 2D coordinates
    mympi.topoid = list(mympi.topocomm.Get_coords(myid))

    # Get neighbors: [left, right, down, up]
    # MPI_Cart_shift(topo, direction, disp, &src, &dest)
    # direction=0 → x-direction; direction=1 → z-direction
    left, right = mympi.topocomm.Shift(0, 1)
    down, up   = mympi.topocomm.Shift(1, 1)

    mympi.neighid[0] = left   # west / left
    mympi.neighid[1] = right  # east / right
    mympi.neighid[2] = down   # south / down (z-direction)
    mympi.neighid[3] = up     # north / up

    return mympi


def grid_coord_exchange(gdcurv: GridData, mpi: MyMPI) -> None:
    """
    Exchange boundary data of 2D grid (nz, nx) 
    between MPI ranks for X/Z directions
    Core logic: Swap boundary columns/rows of X (left/right) 
    and Z (down/up) directions to ghost points
    """
    nx = gdcurv.nx
    nz = gdcurv.nz
    ni1 = gdcurv.ni1    # Column index for sending in X direction
    ni2 = gdcurv.ni2    # Associated column index for receiving in X direction
    nk1 = gdcurv.nk1    # Row index for sending in Z direction
    nk2 = gdcurv.nk2    # Associated row index for receiving in Z direction
    x2d = gdcurv.x2d
    z2d = gdcurv.z2d
    
    topocomm = mpi.topocomm
    neighid = mpi.neighid   # Neighbor order: [left, right, down, up]
    x2_neigh = neighid[1]   # X2 direction (right neighbor)
    x1_neigh = neighid[0]   # X1 direction (left neighbor)
    z2_neigh = neighid[3]   # Z2 direction (up neighbor)
    z1_neigh = neighid[2]   # Z1 direction (down neighbor)
    
    # 1. Send to left neighbor, receive from right neighbor
    send_x = x2d[:, ni1].copy()
    send_z = z2d[:, ni1].copy()
    recv_x = np.empty(nz, dtype=np.float32)
    recv_z = np.empty(nz, dtype=np.float32)
    topocomm.Sendrecv(send_x, dest=x1_neigh, sendtag=110,
                      recvbuf=recv_x, source=x2_neigh, recvtag=110)
    topocomm.Sendrecv(send_z, dest=x1_neigh, sendtag=110,
                      recvbuf=recv_z, source=x2_neigh, recvtag=110)
    
    if x2_neigh != MPI.PROC_NULL:
        x2d[:, ni2+1] = recv_x
        z2d[:, ni2+1] = recv_z
    
    # 2. Send to right neighbor, receive from left neighbor
    send_x = x2d[:, ni2].copy()
    send_z = z2d[:, ni2].copy()
    recv_x = np.empty(nz, dtype=np.float32)
    recv_z = np.empty(nz, dtype=np.float32)
    
    topocomm.Sendrecv(send_x, dest=x2_neigh, sendtag=120,
                      recvbuf=recv_x, source=x1_neigh, recvtag=120)
    topocomm.Sendrecv(send_z, dest=x2_neigh, sendtag=120,
                      recvbuf=recv_z, source=x1_neigh, recvtag=120)
    
    if x1_neigh != MPI.PROC_NULL:
        x2d[:, ni1-1] = recv_x
        z2d[:, ni1-1] = recv_z
    
    # 3. Send to down neighbor, receive from up neighbor
    send_x = x2d[nk1, :].copy()
    send_z = z2d[nk1, :].copy()
    recv_x = np.empty(nx, dtype=np.float32)
    recv_z = np.empty(nx, dtype=np.float32)
    
    topocomm.Sendrecv(send_x, dest=z1_neigh, sendtag=210,
                      recvbuf=recv_x, source=z2_neigh, recvtag=210)
    topocomm.Sendrecv(send_z, dest=z1_neigh, sendtag=210,
                      recvbuf=recv_z, source=z2_neigh, recvtag=210)
    
    if z2_neigh != MPI.PROC_NULL:
        x2d[nk2+1, :] = recv_x
        z2d[nk2+1, :] = recv_z
    
    # 4. Send to up neighbor, receive from down neighbor
    send_x = x2d[nk2, :].copy()
    send_z = z2d[nk2, :].copy()
    recv_x = np.empty(nx, dtype=np.float32)
    recv_z = np.empty(nx, dtype=np.float32)
    
    topocomm.Sendrecv(send_x, dest=z2_neigh, sendtag=220,
                      recvbuf=recv_x, source=z1_neigh, recvtag=220)
    topocomm.Sendrecv(send_z, dest=z2_neigh, sendtag=220,
                      recvbuf=recv_z, source=z1_neigh, recvtag=220)
    
    if z1_neigh != MPI.PROC_NULL:
        x2d[nk1-1, :] = recv_x
        z2d[nk1-1, :] = recv_z


def grid_comm_neighbor_adapted(mympi: MyMPI, x2d: np.ndarray, 
                               z2d: np.ndarray, nx: int, nz: int) -> None:
    """
    Cross-node boundary exchange using Neighbor_alltoallw
    """
    neighbors = mympi.neighid  
    
    topocomm = mympi.topocomm 
    myid = mympi.myid

    # Step 1: Prepare SEND data
    float_size = np.dtype(np.float32).itemsize
    send_counts = [0, 0, 0, 0]  # [left, right, down, up]
    send_displs = [0, 0, 0, 0]  # Byte offsets
    send_types = [MPI.FLOAT] * 4
    
    # Calculate send counts per neighbor
    for i, neighbor in enumerate(neighbors):
        if neighbor == MPI.PROC_NULL:
            continue
        if i < 2:  # Left/right: send column data (2*nz floats)
            send_counts[i] = 2 * nz
        else:  # Down/up: send row data (2*nx floats)
            send_counts[i] = 2 * nx
    
    # Create contiguous send buffer
    total_send_elems = sum(send_counts)
    send_buffer = np.empty(total_send_elems, dtype=np.float32)
    
    # Fill send buffer and set displacements
    current_offset = 0
    for i, neighbor in enumerate(neighbors):
        if neighbor == MPI.PROC_NULL:
            continue
        
        # Get data to send
        if i == 0:  # Left: send column index=1
            start = current_offset
            send_buffer[start:start+nz] = x2d[:, 1]
            send_buffer[start+nz:start+2*nz] = z2d[:, 1]
        elif i == 1:  # Right: send column index=-2
            start = current_offset
            send_buffer[start:start+nz] = x2d[:, -2]
            send_buffer[start+nz:start+2*nz] = z2d[:, -2]
        elif i == 2:  # Down: send row index=1
            start = current_offset
            send_buffer[start:start+nx] = x2d[1, :]
            send_buffer[start+nx:start+2*nx] = z2d[1, :]
        elif i == 3:  # Up: send row index=-2
            start = current_offset
            send_buffer[start:start+nx] = x2d[-2, :]
            send_buffer[start+nx:start+2*nx] = z2d[-2, :]
        
        send_displs[i] = current_offset * float_size
        current_offset += send_counts[i]

    # Step 2: Prepare RECV data
    recv_counts = [0, 0, 0, 0]
    recv_displs = [0, 0, 0, 0]
    recv_types = [MPI.FLOAT] * 4
    
    # Calculate receive counts (symmetric to send)
    for i, neighbor in enumerate(neighbors):
        if neighbor == MPI.PROC_NULL:
            continue
        if i < 2:  # Left/right neighbors send columns to this rank
            recv_counts[i] = 2 * nz
        else:  # Down/up neighbors send rows to this rank
            recv_counts[i] = 2 * nx
    
    # Create contiguous receive buffer
    total_recv_elems = sum(recv_counts)
    recv_buffer = np.empty(total_recv_elems, dtype=np.float32)
    
    # Set receive displacements
    current_offset = 0
    for i in range(4):
        if recv_counts[i] > 0:
            recv_displs[i] = current_offset * float_size
            current_offset += recv_counts[i]

    # Step 3: Perform communication
    try:
        topocomm.Neighbor_alltoallw(
            (send_buffer, send_counts, send_displs, send_types),
            (recv_buffer, recv_counts, recv_displs, recv_types)
        )
    except MPI.Exception as e:
        print(f"Rank {myid}: MPI communication failed - {e}")
        raise  

    # Step 4: Unpack received data to ghost points
    current_offset = 0
    for i, neighbor in enumerate(neighbors):
        if neighbor == MPI.PROC_NULL:
            continue
            
        count = recv_counts[i]
        data = recv_buffer[current_offset:current_offset+count]
        current_offset += count  # 总是递增, 与recv_displs顺序一致
        
        if i == 0:  # Left neighbor -> fill left ghost column (index 0)
            x2d[:, 0] = data[:nz]
            z2d[:, 0] = data[nz:]
        elif i == 1:  # Right neighbor -> fill right ghost column (index -1)
            x2d[:, -1] = data[:nz]
            z2d[:, -1] = data[nz:]
        elif i == 2:  # Down neighbor -> fill bottom ghost row (index 0)
            x2d[0, :] = data[:nx]
            z2d[0, :] = data[nx:]
        elif i == 3:  # Up neighbor -> fill top ghost row (index -1)
            x2d[-1, :] = data[:nx]
            z2d[-1, :] = data[nx:]

    