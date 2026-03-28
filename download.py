import urllib.request
import gzip
import shutil
import os

base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
files = {
    "train-images-idx3-ubyte.gz": "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte.gz": "train-labels-idx1-ubyte"
}

for gz_file, out_file in files.items():
    if not os.path.exists(out_file):
        print(f"Downloading {gz_file}...")
        urllib.request.urlretrieve(base_url + gz_file, gz_file)
        print(f"Extracting {gz_file}...")
        with gzip.open(gz_file, 'rb') as f_in:
            with open(out_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(gz_file)
        print(f"Saved {out_file}")
    else:
        print(f"{out_file} already exists.")
