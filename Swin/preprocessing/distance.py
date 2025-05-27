
import numpy as np

def get_distance(bb1, bb2):
    """
    Calculate the distance between a point and list of points

    Parameters
    ----------
    bb1 : list
        List: [r_0, r_1]
    bb2 : list
        List: [[c_0, c_1], [c_00, c_11], ... ]
    Returns
    -------
    list
        List: [d_0, d_1, ...]

    """
    
    d= []
    for cc in bb2:
        dis= np.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(bb1, cc)))
        d.append(dis)

    return d    