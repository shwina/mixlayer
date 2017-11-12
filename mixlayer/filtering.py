import numpy as np

def filter5(f, alpha=1./(2**10)):
    """
    Apply 5th order filter to given 2-d array f,
    
        f = f + alpha*D
        
    where alpha is the filter amplitude and D
    is the filter.
    
    Parameters
    ----------

    f       : 2d array to filter (with size at least 10x10)
    alpha   : filtering amplitude
    

    Notes
    -----
    
    A default amplitude (alpha) of 1/2**10 is used
    for a 5th order filter (1-)^(n+1).(2**-2n) is the recommended
    amplitude (Kennedy & Carpenter 1997).
    """

    ny, nx = f.shape
    inner_filter = np.empty_like(f, dtype=np.float64)
    
    inner_filter[...] = (252*f[...] -210*(np.roll(f,-1,1)+np.roll(f,+1,1)) +
            120*(np.roll(f,-2,1)+np.roll(f,+2,1)) +
            -45*(np.roll(f,-3,1)+np.roll(f,+3,1)) +
            10*(np.roll(f,-4,1)+np.roll(f,+4,1)) +
            -1*(np.roll(f,-5,1)+np.roll(f,+5,1)))

    inner_filter[5:-5, :] += (252*f[...]
            -210*(np.roll(f,-1,0)+np.roll(f,+1,0)) + 
            120*(np.roll(f,-2,0)+np.roll(f,+2,0)) +(
            -45*(np.roll(f,-3,0)+np.roll(f,+3,0)) +
            10*(np.roll(f,-4,0)+np.roll(f,+4,0)) +
            -1*(np.roll(f,-5,0)+np.roll(f,+5,0)))[5:-5, :]

    inner_filter[0, :] += (f[0,:]-5*f[1,:]+10*f[2,:]-10*f[3,:]+5*f[4,:]-1*f[5,:])
    
    inner_filter[1, :] += (-5*f[0,:]+26*f[1,:]-55*f[2,:]+60*f[3,:]-35*f[4,:]+10*f[5,:]
        -1*f[6,:])

    inner_filter[2, :] += (10*f[0,:]-55*f[1,:]+126*f[2,:]-155*f[3,:]+110*f[4,:]-45*f[5,:]
         +10*f[6,:]-1*f[7,:])

    inner_filter[3, :] += (-10*f[0,:]+60*f[1,:]-155*f[2,:]+226*f[3,:]-205*f[4,:]+120*f[5,:]
            -45*f[6,:]+10*f[7,:]-1*f[8,:])

    inner_filter[4, :] += (5*f[0,:]-35*f[1,:]+110*f[2,:]-205*f[3,:]+251*f[4,:]-210*f[5,:]
            +120*f[6,:]-45*f[7,:]+10*f[8,:]-1*f[9,:])

    inner_filter[-1, :] += (f[-1,:]-5*f[-2,:]+10*f[-3,:]-10*f[-4,:]+5*f[-5,:]-1*f[-6,:])
    
    inner_filter[-2, :] += (-5*f[-1,:]+26*f[-2,:]-55*f[-3,:]+60*f[-4,:]-35*f[-5,:]+10*f[-6,:]
        -1*f[-7,:])

    inner_filter[-3, :] += (10*f[-1,:]-55*f[-2,:]+126*f[-3,:]-155*f[-4,:]+110*f[-5,:]-45*f[-6,:]
         +10*f[-7,:]-1*f[-8,:])

    inner_filter[-4, :] += (-10*f[-1,:]+60*f[-2,:]-155*f[-3,:]+226*f[-4,:]-205*f[-5,:]+120*f[-6,:]
            -45*f[-7,:]+10*f[-8,:]-1*f[-9,:])

    inner_filter[-5, :] += (5*f[-1,:]-35*f[-2,:]+110*f[-3,:]-205*f[-4,:]+251*f[-5,:]-210*f[-6,:]
            +120*f[-7,:]-45*f[-8,:]+10*f[-9,:]-1*f[-10,:])
    
    f[...] = f[...] - alpha*inner_filter

