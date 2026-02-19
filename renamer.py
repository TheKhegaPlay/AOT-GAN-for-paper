import os
path = r"..\data\images\val_orig_only"
for f in os.listdir(path):
    if "_orig" in f:
        os.rename(
            os.path.join(path, f),
            os.path.join(path, f.replace("_orig", ""))
        )
print("Done")

