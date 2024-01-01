import numpy as np


# find the selection of groupings such that the rows median
# row value for each group is at least min_row_spacing apart
# while maximizing the number of groups


def rowify(positions, min_row_spacing):
    positions = np.asarray(positions, dtype=np.float64)
    groups = []
    
    for i, pos in enumerate(positions):
        ind, = np.where(np.abs(pos - positions) < min_row_spacing)
        
        groups.append(set(ind) - set([i]))
        
    print(positions)
    print(groups)
                

positions = np.random.uniform(size=(4, ))
positions.sort()

rowify(positions, 0.1)



    



        
    



