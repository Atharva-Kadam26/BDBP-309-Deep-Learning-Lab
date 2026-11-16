
import urllib.request

url = "https://rest.uniprot.org/uniprotkb/search?query=organism_id%3A83333%20AND%20reviewed%3Atrue&format=fasta&size=100"

output_file = "proteins.fasta"

urllib.request.urlretrieve(url, output_file)

print("FASTA file downloaded successfully!")
print("File saved as:", output_file)

