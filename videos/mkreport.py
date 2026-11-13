import os
WORK = os.environ.get("VIDEO_WORK", "work") + "/"  # scratch dir: run outputs, crops, intermediates
import subprocess, sys
from PIL import Image, ImageDraw, ImageFont
W, H, FPS = 1920, 1080, 15
BG, BAR, FG, LINE = (23,24,28), (42,45,51), (232,232,232), (70,74,82)
S = sys.argv[2]
sans = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 34)
ff = subprocess.Popen(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}","-r",str(FPS),"-i","-",
    "-c:v","libx264","-pix_fmt","yuv420p","-crf","18","-movflags","+faststart",sys.argv[1]], stdin=subprocess.PIPE)
def emit(img, n=1):
    b = img.tobytes()
    for _ in range(n): ff.stdin.write(b)
def page(name): return Image.open(f"{S}/shots/{name}_c.png").convert("RGB")
SPLIT = 5520   # 2x px: boundary between 01 overview and 02/03 shifts
rep = page("report").crop((0,0,2560,8660))   # footer cropped
panels = {"sum_top": rep.crop((0,0,rep.width,SPLIT)), "sum_bot": rep.crop((0,SPLIT,rep.width,rep.height)),
          "mosaic": page("mosaic").crop((0,0,2560,3800)), "overlay": page("overlay").crop((0,0,2560,3800)),
          "chips": page("chips").crop((0,0,2560,3800)), "chips2": page("chips").crop((0,3800,2560,7600))}
def scene(a, b, cap):
    im = Image.new("RGB",(W,H),BG); d = ImageDraw.Draw(im)
    avail_h, gap = H-110, 40
    ims = [panels[a], panels[b]]
    s = min(avail_h/max(p.height for p in ims), (W-3*gap)/2/ims[0].width)
    ims = [p.resize((round(p.width*s),round(p.height*s)),Image.LANCZOS) for p in ims]
    x = (W - sum(p.width for p in ims) - gap)//2
    for p in ims:
        y = 20
        d.rectangle([x-1,y-1,x+p.width,y+p.height],outline=LINE); im.paste(p,(x,y)); x += p.width+gap
    d.rectangle([0,H-80,W,H],fill=BAR)
    d.text(((W-d.textlength(cap,font=sans))/2,H-64),cap,font=sans,fill=FG)
    return im
scenes = [scene("sum_top","sum_bot","Summary - inputs, accuracy statistics, matching confidence, plots"),
          scene("mosaic","overlay","Mosaic and Overlay - visual check of the registration"),
          scene("chips","chips2","Chips - reference / monitored key point chips with their Laplacian")]
NAT = Image.open(f"{S}/fr/mosaic_native.png").convert("RGB")
def zframe(t, cap):
    # t in 0..1: from whole checkerboard to a detail crop (900 px square, centre .30/.42)
    e = ease(t); full = NAT.width; cs = 900
    cx0, cy0 = full/2, full/2; cx1, cy1 = .30*full, .42*full
    sz = full*(cs/full)**e; cx = cx0+(cx1-cx0)*e; cy = cy0+(cy1-cy0)*e
    box = (cx-sz/2, cy-sz/2, cx+sz/2, cy+sz/2); side = H-110
    pic = NAT.resize((side,side),Image.BILINEAR, box=box) if False else NAT.crop(tuple(round(v) for v in box)).resize((side,side),Image.LANCZOS)
    im = Image.new("RGB",(W,H),BG); d = ImageDraw.Draw(im); x = (W-side)//2
    d.rectangle([x-1,19,x+side,20+side],outline=LINE); im.paste(pic,(x,20))
    d.rectangle([0,H-80,W,H],fill=BAR); d.text(((W-d.textlength(cap,font=sans))/2,H-64),cap,font=sans,fill=FG)
    return im
ZCAP = "Zoom on the checkerboard - rivers and fields stay continuous across tile edges"
REP = rep; CHP = page("chips")
ASP = W/(H-110)
def ease(t): return t*t*t*(t*(6*t-15)+10)   # smootherstep: slow start and end
def lerp_box(a,b,e):
    ca=((a[0]+a[2])/2,(a[1]+a[3])/2); cb=((b[0]+b[2])/2,(b[1]+b[3])/2)
    wa,wb=a[2]-a[0],b[2]-b[0]; ha,hb=a[3]-a[1],b[3]-b[1]
    w=wa*(wb/wa)**e; h=ha*(hb/ha)**e   # geometric: constant perceived zoom rate
    cx=ca[0]+(cb[0]-ca[0])*e; cy=ca[1]+(cb[1]-ca[1])*e
    return (cx-w/2,cy-h/2,cx+w/2,cy+h/2)
def fit(box):
    x0,y0,x1,y1 = box; cx,cy = (x0+x1)/2,(y0+y1)/2; w,h = x1-x0,y1-y0
    if w/h < ASP: w = h*ASP
    else: h = w/ASP
    return (cx-w/2,cy-h/2,cx+w/2,cy+h/2)
def vframe(panel, box, cap):
    box = tuple(round(v) for v in box)
    pic = panel.crop(box).resize((W,H-110),Image.LANCZOS)
    im = Image.new("RGB",(W,H),BG); d = ImageDraw.Draw(im); im.paste(pic,(0,20))
    d.rectangle([0,H-80,W,H],fill=BAR); d.text(((W-d.textlength(cap,font=sans))/2,H-64),cap,font=sans,fill=FG)
    return im
def zoom_to(panel, target, cap, move=5.0, hold=2.5):
    global prev
    end = fit(target); cy = (end[1]+end[3])/2; pw = panel.width
    hh = pw/ASP; y0 = min(max(cy-hh/2,0),panel.height-hh); start = (0,y0,pw,y0+hh)
    f0 = vframe(panel,start,cap); blend_in(f0); emit(f0, FPS//2)
    n = round(FPS*move)
    for k in range(1,n+1):
        emit(vframe(panel, lerp_box(start,end,ease(k/n)), cap))
    f1 = vframe(panel,end,cap); emit(f1, round(FPS*hold)); prev = f1
NAT = Image.open(f"{S}/fr/mosaic_native.png").convert("RGB")
def zframe(t, cap):
    e = ease(t); full = NAT.width; cs = 900
    cx0, cy0 = full/2, full/2; cx1, cy1 = .30*full, .42*full
    sz = full*(cs/full)**e; cx = cx0+(cx1-cx0)*e; cy = cy0+(cy1-cy0)*e
    box = (cx-sz/2, cy-sz/2, cx+sz/2, cy+sz/2); side = H-110
    pic = NAT.crop(tuple(round(v) for v in box)).resize((side,side),Image.LANCZOS)
    im = Image.new("RGB",(W,H),BG); d = ImageDraw.Draw(im); x = (W-side)//2
    d.rectangle([x-1,19,x+side,20+side],outline=LINE); im.paste(pic,(x,20))
    d.rectangle([0,H-80,W,H],fill=BAR); d.text(((W-d.textlength(cap,font=sans))/2,H-64),cap,font=sans,fill=FG)
    return im
ZCAP = "Zoom on the checkerboard - rivers and fields stay continuous across tile edges"
prev = Image.new("RGB",(W,H),BG)
def blend_in(cur):
    for k in range(1,8): emit(Image.blend(prev,cur,k/8))
for idx, cur in enumerate(scenes):
    blend_in(cur); emit(cur, FPS*8); prev = cur
    if idx == 0:
        zoom_to(REP,(80,1880,2480,3092),"Accuracy results - statistics and matching confidence (reliable, 100 %)")
        zoom_to(REP,(80,5640,2480,6852),"DX and DY shifts - mean and std by row / column")
        zoom_to(REP,(100,7080,2460,8560),"Circular error distribution - CE90 / CE95")
    if idx == 1:
        z0 = zframe(0, ZCAP); blend_in(z0); emit(z0, FPS*1)
        for k in range(1, FPS*6+1): emit(zframe(k/(FPS*6), ZCAP))
        z1 = zframe(1, ZCAP); emit(z1, FPS*3); prev = z1
    if idx == 2:
        zoom_to(CHP,(120,1500,2240,2570),"Key point chips - reference / monitored, with Laplacian and scores")
ff.stdin.close(); ff.wait()
