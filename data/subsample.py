import shutil
import json
import pandas as pd

N = 1000

with open("train.json", "rb") as f:
    annotations = json.load(f)
images_df = pd.DataFrame(annotations["images"]).set_index("id")
annotations_df = pd.DataFrame(annotations["annotations"]).set_index("id")
categories_df = pd.DataFrame(annotations["categories"]).set_index("id")

subset_images_df = images_df.sample(N)
subset_ids = subset_images_df.index
subset_annotations_df = annotations_df[annotations_df.image_id.isin(subset_ids)]

subset_annotations = {"images": subset_images_df.reset_index().to_dict(orient="records"),
                      "annotations": subset_annotations_df.reset_index().to_dict(orient="records"),
                      "categories": categories_df.reset_index().to_dict(orient="records")}

with open("subtrain.json", "w") as f:
    json.dump(subset_annotations, f, indent=4)

shutil.rmtree("subtrain/")
os.mkdir("subtrain/")
for filename in subset_images_df.file_name:
    shutil.copyfile(f"train/{filename}", f"subtrain/{filename}")
