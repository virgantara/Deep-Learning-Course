import zipfile
import gdown

file_id = "1Dyrmn3m1R7EclrzQkaU3KnCxXK5VjAJ3"
url = f"https://drive.google.com/uc?id={file_id}"

output = "animals.zip"  # or your file name
gdown.download(url, output, quiet=False)


with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall('data/animals')  # Extract into 'dataset' folder


