import os
WORK = os.environ.get("VIDEO_WORK", "work") + "/"  # scratch dir: run outputs, crops, intermediates
import subprocess, sys
from PIL import Image, ImageDraw, ImageFont
W,H,FPS=1920,1080,15
BG,FG,ACC,DIM=(23,24,28),(235,235,235),(110,220,130),(150,155,165)
I="/home/gautier/projects/isl/issues/karios-3.0.0/images/"
F="/usr/share/fonts/truetype/dejavu/DejaVuSans"
big=ImageFont.truetype(F+"-Bold.ttf",96); h1=ImageFont.truetype(F+"-Bold.ttf",56)
txt=ImageFont.truetype(F+".ttf",36); cap=ImageFont.truetype(F+".ttf",26)
tag=ImageFont.truetype(F+"-Bold.ttf",28)
ff=subprocess.Popen(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}","-r",str(FPS),"-i","-",
 "-c:v","libx264","-pix_fmt","yuv420p","-crf","18","-movflags","+faststart",sys.argv[1]],stdin=subprocess.PIPE)
def emit(im,n=1):
    b=im.tobytes()
    for _ in range(n): ff.stdin.write(b)
def fit(p,w,h):
    s=min(w/p.width,h/p.height); return p.resize((int(p.width*s),int(p.height*s)),Image.LANCZOS)
def slide(title,bullets,nb,pics,badge=None):
    im=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(im)
    d.text((80,70),title,font=h1,fill=FG)
    if badge: d.text((80,150),badge,font=tag,fill=ACC)
    y=250
    for b in bullets[:nb]:
        d.text((80,y),"•",font=txt,fill=ACC)
        for ln in b.split("\n"):
            d.text((120,y),ln,font=txt,fill=FG); y+=50
        y+=26
    x0,aw=1000,860; n=len(pics); gap=20; w=(aw-gap*(n-1))//n
    x=x0
    for p,c in pics:
        q=fit(Image.open(p if p.startswith("/") else I+p).convert("RGB"),w,720); im.paste(q,(x,200))
        tw=d.textlength(c,font=cap); d.text((x+(q.width-tw)/2,200+q.height+14),c,font=cap,fill=DIM)
        x+=w+gap
    return im
last=None
def show(title,bullets,pics,badge=None,hold=24):
    global last
    for k in range(len(bullets)+1):
        cur=slide(title,bullets,k,pics,badge)
        if k==0:
            for j in range(1,6): emit(Image.blend(last,cur,j/5))
        emit(cur, 45 if k==len(bullets) else 30 if k else 15)
    last=cur
def sec(title,sub):
    global last
    s=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(s)
    d.text((W/2,480),title,font=big,fill=ACC,anchor="mm"); d.text((W/2,590),sub,font=txt,fill=DIM,anchor="mm")
    for j in range(1,6): emit(Image.blend(last,s,j/5))
    emit(s,FPS*3); last=s
t=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(t)
d.text((W/2,420),"KARIOS 3.0",font=big,fill=FG,anchor="mm")
d.text((W/2,540),"Measure the geometric accuracy of Earth observation images",font=txt,fill=DIM,anchor="mm")
d.text((W/2,610),"KLT-based Algorithm for Registration of Images from Observing Systems",font=cap,fill=DIM,anchor="mm")
emit(t,FPS*5); last=t
sec("What KARIOS does","in 30 seconds")
show("Compare two images",
 ["Reference vs monitored image,\noptical or SAR, any resolution","KLT matching of corner key points,\nwith image filtering (Laplacian)","Outlier filtering keeps\nonly reliable matches"],
 [("overview.png","Shift of every key point (dx, dy)")])
L="/home/gautier/projects/isl/data/karios-out/lacraux-overlay/LC81960302017318MTI00_B4_REFRAMED_GDAL_S2A_OPER_MSI_L1C_TL_MPS__20161018T120000_A000008_T31TFJ_B04_30m/"
show("Accuracy metrics and plots",
 ["Mean shift, RMSE, CE90 / CE95","Overview, dx / dy and CE plots","CSV of key points,\nGeoTIFF shift maps","Mask, DEM and no-data aware"],
 [(L+"02_dx.png","Shift along X"),(L+"04_ce.png","Circular error")])
sec("What's new in 3.0","major additions")
def scroll_slide(title,bullets,path,capt):
    global last
    full=Image.open(path).convert("RGB"); full=full.resize((860,int(full.height*860/full.width)),Image.LANCZOS)
    VH=720; mx=full.height-VH
    def frame(nb,off):
        im=slide(title,bullets,nb,[("report_report.png","")])
        d=ImageDraw.Draw(im); d.rectangle([1000,190,1870,1000],fill=BG)
        im.paste(full.crop((0,off,860,off+VH)),(1000,200))
        tw=d.textlength(capt,font=cap); d.text((1000+(860-tw)/2,200+VH+14),capt,font=cap,fill=DIM)
        return im
    nb=len(bullets); N=15+30*nb+45
    for f in range(N):
        k=0 if f<15 else min(nb,(f-15)//30+1)
        if f==0:
            for j in range(1,6): emit(Image.blend(last,frame(0,0),j/5))
        t=f/(N-1); e=t*t*t*(t*(6*t-15)+10)
        emit(frame(k,int(mx*e)))
    last=frame(len(bullets),mx)
scroll_slide("New HTML report",
 ["NEW in 3.0: a complete HTML report","Summary, Products, Mosaic,\nOverlay and Chips tabs","Full processing configuration\nincluded"],
 I+"report_report.png","Summary tab")
show("Mosaic and overlay",
 ["Checkerboard mosaic\n(--mosaic-tile-size)","Colour overlay\n(--generate-overlay)"],
 [("mosaic_checkerboard_crop_native.png","Checkerboard mosaic"),("overlay_color_crop_native.png","Colour overlay")])
show("Coarse-to-fine matching",
 ["--enable-coarse-to-fine","Many more key points,\nbetter coverage","Key point density map\n(--debug, 05_coverage.png)","Configurable pyramid depth\n(klt_matching.maxLevel)"],
 [(WORK+"dens_default.png","Key point density: default"),(WORK+"dens_coarse_to_fine.png","Key point density: coarse-to-fine")])
show("karios align",
 ["Global alignment: SIFT + RANSAC + ECC","Any CRS, rotated grids,\ndifferent resolutions","Then run karios process on the result"],
 [("align_before_overlay_crop_native.png","Before"),("align_after_overlay_crop_native.png","After")],badge="NEW COMMAND")
show("Matching confidence",
 ["Verdict: reliable / doubtful /\nunreliable","In the console, the HTML report\nand summary.json","Know at once whether\nto trust the numbers"],
 [(WORK+"rep_acc.png","Accuracy results")])
show("Key point chips",
 ["Raw and Laplacian image chips","ZNCC and mutual information (NMI)\nper key point","Sort, filter and XOR overlay"],
 [("report_chips.png","Chips tab")])
show("And more",
 ["DEM elevation in the CSV output","Coverage and quality maps\n(--debug)","Vector masks, --no-value filtering,\nautomatic no-data","Hardened inputs and HTML escaping"],
 [("report_products.png","Products tab")])
end=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(end)
d.text((W/2,470),"KARIOS 3.0",font=big,fill=FG,anchor="mm")
d.text((W/2,590),"github.com/telespazio-tim/karios",font=txt,fill=ACC,anchor="mm")
for j in range(1,6): emit(Image.blend(last,end,j/5))
emit(end,FPS*3)
ff.stdin.close(); ff.wait()
