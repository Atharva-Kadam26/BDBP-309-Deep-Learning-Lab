
import pandas as pd

captions = []

with open("Flickr8k.token.txt", "r") as file:
    for line in file:
        image_name, caption = line.strip().split("\t")
        image_name = image_name.split("#")[0]

        captions.append({
            "image": image_name,
            "caption": caption
        })

df = pd.DataFrame(captions)
df.to_csv("captions.csv", index=False)

print("Captions CSV created successfully!")
print("Total image-caption pairs:", len(df))
print(df.head())
