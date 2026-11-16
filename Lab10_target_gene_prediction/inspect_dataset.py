
import h5py

file_path = "GSE92743_Broad_Affymetrix_training_Level3_Q2NORM_n100000x12320.gctx"

with h5py.File(file_path, "r") as f:

    print("ROW metadata:")
    row = f["0"]["META"]["ROW"]

    for key in row.keys():
        print(key, row[key].shape, row[key].dtype)
        print("First 5:", row[key][:5])

    print("\nCOL metadata:")
    col = f["0"]["META"]["COL"]

    for key in col.keys():
        print(key, col[key].shape, col[key].dtype)
        print("First 5:", col[key][:5])
