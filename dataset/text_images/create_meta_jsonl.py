#!/usr/bin/env python3
"""
Generate metadata JSON matching the structure of
Dataset/example_image_dataset/metadata_qwen_imgae_edit_multi.json

Defaults read from:
 - train metadata: Dataset/text_images/train_metadata.jsonl
 - lr images: Dataset/text_images/lr_image
 - hr images: Dataset/text_images/hr_image

Output default: Dataset/text_images/generated_metadata.json

Each entry produced has this form:
{
  "image": "hr_image/train/<file>",
  "prompt": "... contains the text \"<extracted>\" ...",
  "edit_image": ["lr_image/train/<file>", "hr_image/train/<file>"]
}

"""
import argparse
import json
import os
import sys


PROMPT_TEMPLATE = (
	'Restore and enhance the low-quality input image to generate a high-resolution, high-quality version. '
	'The text “{text}” within the image should remain clear, accurate, and unchanged. Preserve the overall composition, '
	'color palette, and subject matter of the original image while improving its clarity, sharpness, and detail.'
)


def extract_text_from_ground_truth(gt_str):
	try:
		# ground_truth is a JSON-encoded string in the file
		parsed = json.loads(gt_str)
		# expected structure: {"gt_parse": {"text_sequence": "..."}}
		if isinstance(parsed, dict):
			gt_parse = parsed.get("gt_parse") or parsed.get("gt_parse_v2") or parsed
			if isinstance(gt_parse, dict):
				text = gt_parse.get("text_sequence") or gt_parse.get("text")
				if isinstance(text, str):
					return text
		return None
	except Exception:
		return None


def main():
	p = argparse.ArgumentParser(description="Generate metadata JSON for restoration/edit tasks")
	p.add_argument("--train_metadata", default="train_metadata.jsonl",
				   help="path to train_metadata.jsonl (relative to working dir or absolute)")
	p.add_argument("--lr_dir", default="lr_image",
				   help="low-res images directory (relative to train_metadata parent)")
	p.add_argument("--hr_dir", default="hr_image",
				   help="high-res images directory (relative to train_metadata parent)")
	p.add_argument("--out", default="generated_metadata.json",
				   help="output json file (array) to write")
	p.add_argument("--relative", action="store_true",
				   help="write image paths relative to train_metadata parent (default: absolute if input is absolute)")
	args = p.parse_args()

	train_meta_path = args.train_metadata
	if not os.path.isabs(train_meta_path):
		# assume file is in the same folder as this script if relative
		script_dir = os.path.dirname(os.path.abspath(__file__))
		train_meta_path = os.path.join(script_dir, train_meta_path)

	if not os.path.exists(train_meta_path):
		print(f"train metadata not found: {train_meta_path}", file=sys.stderr)
		sys.exit(2)

	base_dir = os.path.dirname(train_meta_path)

	out_path = args.out
	if not os.path.isabs(out_path):
		out_path = os.path.join(base_dir, out_path)

	entries = []
	with open(train_meta_path, "r", encoding="utf-8") as fh:
		for line in fh:
			line = line.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
			except Exception:
				continue

			file_name = obj.get("file_name") or obj.get("image") or obj.get("img")
			if not file_name:
				continue

			gt = obj.get("ground_truth") or obj.get("text") or obj.get("caption")
			if not gt:
				continue

			text = extract_text_from_ground_truth(gt)
			if not text:
				# if extraction failed, try if ground_truth itself is plain text
				if isinstance(gt, str) and len(gt) < 1000:
					text = gt
				else:
					continue

			# build paths
			lr_path = os.path.join(base_dir, args.lr_dir, "train", file_name)
			hr_path = os.path.join(base_dir, args.hr_dir, "train", file_name)

			if args.relative:
				# make paths relative to base_dir
				lr_path = os.path.relpath(lr_path, base_dir)
				hr_path = os.path.relpath(hr_path, base_dir)

			prompt = PROMPT_TEMPLATE.format(text=text)

			entry = {
				"image": hr_path,
				"prompt": prompt,
				"edit_image": [lr_path, hr_path]
			}
			entries.append(entry)

	# write JSON array
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	with open(out_path, "w", encoding="utf-8") as of:
		json.dump(entries, of, ensure_ascii=False, indent=4)

	print(f"Wrote {len(entries)} entries to {out_path}")


if __name__ == "__main__":
	main()

