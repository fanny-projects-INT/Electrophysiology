from pathlib import Path
import numpy as np
from functions.qc_labeler import run_qc_labeler, QCParams

#DATA_ROOT = Path(r"D:\Data_Mice")
DATA_ROOT = Path(r"E:\Aurelien\Data_Mice")
SESSION_ID = "VF069_2025_12_06"

mc_thresh = None
nc_thresh = None
amp_thresh_uv = 20  

if __name__ == "__main__":
    params = QCParams(
        mc_thresh=mc_thresh,
        nc_thresh=nc_thresh,
        amp_thresh_uv=amp_thresh_uv,  
        page_size=12,
        dtype=np.int16
    )
    run_qc_labeler(SESSION_ID, DATA_ROOT, params=params)
