synthtiger -o ../hr_image -c 10000 -w 24 -v template.py SynthDoG ./config/config_en.yaml

python degradate.py --src ../hr_image --dst ../lr_image --cores 24 --short_min 1024 --short_max 1024
