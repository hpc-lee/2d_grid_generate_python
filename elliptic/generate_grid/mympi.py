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

        self.send_counts = None
        self.recv_counts = None
        self.send_displs = None
        self.recv_displs = None
        self.send_buffer = None
        self.recv_buffer = None
        self.send_types = [MPI.FLOAT] * 4
        self.recv_types = [MPI.FLOAT] * 4

    def init_buffers(self, nx: int, nz: int) -> None:
        float_size = np.dtype(np.float32).itemsize
        
        self.send_counts = [0, 0, 0, 0]
        self.recv_counts = [0, 0, 0, 0]
        for i, neighbor in enumerate(self.neighid):
            if neighbor != MPI.PROC_NULL:
                self.send_counts[i] = self.recv_counts[i] = 2 * (nz if i < 2 else nx)
        
        self.send_displs = [0, 0, 0, 0]
        offset = 0
        for i in range(4):
            if self.send_counts[i] > 0:
                self.send_displs[i] = offset * float_size
                offset += self.send_counts[i]
        
        self.recv_displs = [0, 0, 0, 0]
        offset = 0
        for i in range(4):
            if self.recv_counts[i] > 0:
                self.recv_displs[i] = offset * float_size
                offset += self.recv_counts[i]
        
        total_send = sum(self.send_counts)
        total_recv = sum(self.recv_counts)
        self.send_buffer = np.empty(total_send, dtype=np.float32) if total_send > 0 else None
        self.recv_buffer = np.empty(total_recv, dtype=np.float32) if total_recv > 0 else None


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
    mympi.neighid[2] = down   # south / down
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


def grid_comm_optimized(mympi: MyMPI, x2d: np.ndarray, z2d: np.ndarray) -> None:
    if mympi.send_buffer is not None:
        offset = 0
        for i, neighbor in enumerate(mympi.neighid):
            if neighbor == MPI.PROC_NULL:
                continue
            
            count = mympi.send_counts[i]
            if count == 0:
                continue
            
            if i < 2:  # Left/right
                nz = x2d.shape[0]
                col_idx = 1 if i == 0 else -2
                mympi.send_buffer[offset:offset+nz] = x2d[:, col_idx]
                mympi.send_buffer[offset+nz:offset+2*nz] = z2d[:, col_idx]
                offset += 2 * nz
            else:  # Down/up
                nx = x2d.shape[1]
                row_idx = 1 if i == 2 else -2
                mympi.send_buffer[offset:offset+nx] = x2d[row_idx, :]
                mympi.send_buffer[offset+nx:offset+2*nx] = z2d[row_idx, :]
                offset += 2 * nx
    
    mympi.topocomm.Neighbor_alltoallw(
        (mympi.send_buffer, mympi.send_counts, mympi.send_displs, mympi.send_types),
        (mympi.recv_buffer, mympi.recv_counts, mympi.recv_displs, mympi.recv_types)
    )
    
    if mympi.recv_buffer is not None:
        offset = 0
        for i, neighbor in enumerate(mympi.neighid):
            if neighbor == MPI.PROC_NULL:
                continue
            
            count = mympi.recv_counts[i]
            if count == 0:
                continue
            
            if i < 2:  # Left/right ghost columns
                nz = x2d.shape[0]
                data = mympi.recv_buffer[offset:offset+count]
                if i == 0:  # Left ghost
                    x2d[:, 0] = data[:nz]
                    z2d[:, 0] = data[nz:]
                else:  # Right ghost
                    x2d[:, -1] = data[:nz]
                    z2d[:, -1] = data[nz:]
                offset += 2 * nz
            else:  # Down/up ghost rows
                nx = x2d.shape[1]
                data = mympi.recv_buffer[offset:offset+count]
                if i == 2:  # Down ghost
                    x2d[0, :] = data[:nx]
                    z2d[0, :] = data[nx:]
                else:  # Up ghost
                    x2d[-1, :] = data[:nx]
                    z2d[-1, :] = data[nx:]
                offset += 2 * nx

    