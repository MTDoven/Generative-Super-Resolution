import json
from pathlib import Path


PROMPT_TEMPLATE = (
    'Restore and enhance the low-quality input image to generate a high-resolution, high-quality version. '
    'The text within the image should be clear, accurate, and unchanged.'
)

METADATA_FILE = "./metadata.jsonl"
EDIT_METADATA_FILE = "./metadata_edit_example.csv"
LIMIT = 300


with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = [json.loads(line) for line in f]

new_metadata = []
for item in metadata:
    file_name = Path(item["file_name"])
    # extract text from metadata
    text = json.loads(item["ground_truth"])["gt_parse"]["text_sequence"]
    image = Path("hr_image") / file_name
    edit_image = Path("lr_image") / file_name
    # construct new metadata item for edit-based generation
    new_item = {
        "prompt": PROMPT_TEMPLATE.format(text=text),
        "image": str(image),
        "edit_image": str(edit_image),
    }
    # save new metadata item to edit metadata list
    new_metadata.append(new_item)
new_metadata = new_metadata[:LIMIT]

with open(EDIT_METADATA_FILE, "w", encoding="utf-8") as f:
    f.write("image,prompt,edit_image\n")
    for item in new_metadata:
        string = "\"" + item["prompt"].replace("\"", "\"\"") + "\""
        f.write(item["image"] + "," + string + "," + item["edit_image"] + "\n")
