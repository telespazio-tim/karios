import os
WORK = os.environ.get("VIDEO_WORK", "work") + "/"  # scratch dir: run outputs, crops, intermediates
import subprocess, sys
from PIL import Image, ImageDraw, ImageFont
W, H, FPS = 1920, 1080, 15
BG, BAR, FG, GREEN, DIM = (23,24,28), (42,45,51), (232,232,232), (110,220,130), (150,155,165)
OUT = "/home/gautier/projects/isl/data/karios-out/lacraux-overlay/LC81960302017318MTI00_B4_REFRAMED_GDAL_S2A_OPER_MSI_L1C_TL_MPS__20161018T120000_A000008_T31TFJ_B04_30m/"
mono = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 30)
sans = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
CMD = ["$ karios process LC81960302017318MTI00_B4_REFRAMED_GDAL.TIF \\",
       "    S2A_OPER_MSI_L1C_TL_MPS__20161018T120000_A000008_T31TFJ_B04_30m.tif --out results"]
LOG = ["[INFO   ] karios.cli.commands - Start KARIOS 3.0.0",
 "[INFO   ] karios.api.core - Process LC81960302017318MTI00_B4_REFRAMED_GDAL.TIF",
 "[INFO   ] karios.api.core - No mask provided",
 "[INFO   ] karios.matcher.klt - Nb valid pixels: 9804715/13395600",
 "[INFO   ] karios.matcher.klt - Start KLT tracking, 1 levels, (25, 25) window size",
 "[INFO   ] karios.matcher.klt - NbPoints(init/final): 20000 / 13743",
 "[INFO   ] karios.matcher.klt - DX/DY(KLT) MEAN: -0.42142904 / 0.5444398",
 "[INFO   ] karios.api.core - Compute patch scores for 10084 points",
 "[INFO   ] karios.accuracy_analysis.accuracy_statistics - Keep  73.38% of total match",
 "[INFO   ] karios.api.core - Matching confidence: reliable (100 %)"]
IMGS = [("01_overview.png","01_overview.png - overview of the shifts"),
        ("02_dx.png","02_dx.png - shift along X (dx)"),
        ("03_dy.png","03_dy.png - shift along Y (dy)"),
        ("04_ce.png","04_ce.png - circular error: CE90 / CE95")]

ff = subprocess.Popen(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}","-r",str(FPS),"-i","-",
    "-c:v","libx264","-pix_fmt","yuv420p","-crf","20","-movflags","+faststart",sys.argv[1]], stdin=subprocess.PIPE)
def emit(img, n=1):
    b = img.tobytes()
    for _ in range(n): ff.stdin.write(b)
def term(cmd_chars, nlog, cursor):
    im = Image.new("RGB",(W,H),BG); d = ImageDraw.Draw(im)
    d.rectangle([0,0,W,52],fill=BAR); d.text((24,10),"karios - terminal",font=small,fill=FG)
    y, left = 90, cmd_chars
    for line in CMD:
        t = line[:max(left,0)]; left -= len(line)
        d.text((24,y),t,font=mono,fill=GREEN); y += 42
    if cursor and left <= 0 and nlog == 0:
        d.rectangle([24+18*len(t),y-42,24+18*len(t)+16,y-8],fill=FG) if False else None
    y += 10
    for l in LOG[:nlog]:
        d.text((24,y),l,font=mono,fill=FG); y += 42
    return im
total = sum(map(len, CMD))
emit(term(0,0,True), FPS)
for c in range(0,total+1,3): emit(term(c,0,True))
emit(term(total,0,False), FPS//2)
for i in range(1,len(LOG)+1): emit(term(total,i,False), 5)
emit(term(total,len(LOG),False), FPS*2)
def slide(path, cap):
    im = Image.new("RGB",(W,H),BG); d = ImageDraw.Draw(im)
    pic = Image.open(OUT+path).convert("RGB"); s = min(W/pic.width,(H-130)/pic.height)
    pic = pic.resize((int(pic.width*s),int(pic.height*s)),Image.LANCZOS)
    im.paste(pic,((W-pic.width)//2,20))
    tw = d.textlength(cap,font=sans); d.text(((W-tw)/2,H-80),cap,font=sans,fill=FG)
    return im
prev = term(total,len(LOG),False)
for p,c in IMGS:
    cur = slide(p,c)
    for k in range(1,8): emit(Image.blend(prev,cur,k/8))
    emit(cur, FPS*4); prev = cur
ff.stdin.close(); ff.wait()
