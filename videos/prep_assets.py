"""Crops used by mkintro.py. Usage: python3 prep_assets.py <karios-3.0.0 issues dir>/images"""
import os, sys
from PIL import Image
I = sys.argv[1].rstrip("/") + "/"
W = os.environ.get("VIDEO_WORK", "work") + "/"
os.makedirs(W, exist_ok=True)
for n in ("default", "coarse_to_fine"):  # "Key point density" panel of the --debug coverage plot
    Image.open(I + f"coverage_{n}.png").crop((0, 180, 1340, 1380)).save(W + f"dens_{n}.png")
r = Image.open(I + "report_report.png")
r.crop((90, 0, 1310, 915)).save(W + "rep_top.png")      # report header + input images
r.crop((90, 920, 1310, 1540)).save(W + "rep_acc.png")   # "Accuracy Results" section
