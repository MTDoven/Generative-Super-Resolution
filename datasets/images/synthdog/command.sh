synthtiger -o ../hr_image -c 10000 -w 24 -v template.py SynthDoG ./config/config_en.yaml

python degradate.py --src ../hr_image/train --dst ../lr_image/train --cores 24 --short_min 256 --short_max 256
python degradate.py --src ../hr_image/test --dst ../lr_image/test --cores 24 --short_min 256 --short_max 256
