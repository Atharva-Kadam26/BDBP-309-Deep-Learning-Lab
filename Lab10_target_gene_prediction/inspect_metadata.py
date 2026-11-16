import h5py

file_path = "GSE92743_Broad_Affymetrix_training_Level3_Q2NORM_n100000x12320.gctx"

with h5py.File(file_path, "r") as f:

    print("ROW metadata:")
    for key in f["0"]["META"]["ROW"].keys():
        print(key)

    print("\nCOL metadata:")
    for key in f["0"]["META"]["COL"].keys():
        print(key)

    print("\nAll metadata groups:")
    print(list(f["0"]["META"].keys()))
