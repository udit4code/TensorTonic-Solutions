import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    v_arr = np.asarray(v, dtype=np.float64)
    original_shape = v_arr.shape
    
    # Step 1 : Treat 1D input (3,) as a 2D row vector (1, 3) for uniform processing
    if v_arr.ndim == 1:
        v_arr = v_arr.reshape(1, -1) 
        
    # Step 2 : Compute L2 norm across axis 1
    norms = np.linalg.norm(v_arr, axis=1, keepdims=True)
    
    # Step 3 : Prevent division by zero (keeps 0.0 / 1.0 = 0.0 for zero vectors)
    # We use the ternary operator : np.where(condition, x, y) , which mean 
    # result = x if condition else y. So, if condition is satisfied, then replace that cell with x, else with y.
    safe_norms = np.where(norms == 0, 1.0, norms)
    
    # 5. Broadcast division and restore original shape
    return (v_arr / safe_norms).reshape(original_shape)
    