import os
WORK = os.environ.get("VIDEO_WORK", "work") + "/"  # scratch dir: run outputs, crops, intermediates
S = WORK+''
import datetime, subprocess, sys, re, json, glob, os
from PIL import Image, ImageDraw, ImageFont
W, H, FPS = 1920, 1080, 15
BG, BAR, FG, GREEN, DIM, ORANGE, RED = (23,24,28), (42,45,51), (232,232,232), (110,220,130), (150,155,165), (240,170,70), (240,100,100)
D = WORK+"demo/"
F = "/usr/share/fonts/truetype/dejavu/"
mono = ImageFont.truetype(F+"DejaVuSansMono.ttf", 27)
sans = ImageFont.truetype(F+"DejaVuSans.ttf", 36)
small = ImageFont.truetype(F+"DejaVuSans.ttf", 26)
big = ImageFont.truetype(F+"DejaVuSans-Bold.ttf", 56)
mid = ImageFont.truetype(F+"DejaVuSans.ttf", 34)
KEEP = [r"Start KARIOS", r"Process ", r"Large", r"Will apply", r"Nb valid pixels", r"NbPoints", r"DX/DY\(KLT\) MEAN",
        r"Compute patch scores", r"Keep ", r"Matching confidence"]
def loglines(run, keep=None):
    out, seen = [], set()
    for l in open(D+run+".log", errors="replace"):
        m = re.match(r"\[(INFO|WARNING)\s*\] \[[^\]]+\] (\S+) +- (.*)", l)
        if not m: continue
        for k in (keep or KEEP):
            if re.search(k, l) and k not in seen:
                seen.add(k); lvl, mod, msg = m.groups()
                msg = re.sub(r"/tmp/\S+/", "", msg)
                out.append(f"[{lvl:<7}] {mod} - {msg}"[:112]); break
    return out
def summary(run):
    return json.load(open(glob.glob(D+run+"/*/summary.json")[0]))
def img(run, name):
    if name.startswith("/"): return name
    return glob.glob(D+run+f"/*/{name}")[0]

class Video:
    def __init__(s, path):
        s.ff = subprocess.Popen(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}","-r",str(FPS),"-i","-",
            "-c:v","libx264","-pix_fmt","yuv420p","-crf","20","-movflags","+faststart",path], stdin=subprocess.PIPE)
        s.prev = Image.new("RGB",(W,H),BG)
    def emit(s, im, n=1):
        b = im.tobytes()
        for _ in range(n): s.ff.stdin.write(b)
        s.prev = im
    def fade(s, cur, hold):
        for k in range(1,8): s.emit(Image.blend(s.prev,cur,k/8))
        s.emit(cur, int(hold*FPS))
    def close(s): s.ff.stdin.close(); s.ff.wait()

def title(num, name, lines, opts):
    im = Image.new("RGB",(W,H),BG); d = ImageDraw.Draw(im)
    d.text((140,260),f"Scenario {num}",font=mid,fill=ORANGE)
    d.text((140,320),name,font=big,fill=FG)
    y = 450
    for l in lines: d.text((140,y),l,font=mid,fill=FG); y += 56
    if opts:
        y += 30; d.text((140,y),"KARIOS options: "+opts,font=mid,fill=GREEN)
    return im
def term(cmd, ncmd, nlog, log, extra=None):
    im = Image.new("RGB",(W,H),BG); d = ImageDraw.Draw(im)
    d.rectangle([0,0,W,52],fill=BAR); d.text((24,10),"karios - terminal",font=small,fill=FG)
    y, left = 90, ncmd
    for line in cmd:
        d.text((24,y),line[:max(left,0)],font=mono,fill=GREEN); left -= len(line); y += 40
    y += 14
    for l in log[:nlog]:
        col = ORANGE if "WARNING" in l else FG
        d.text((24,y),l,font=mono,fill=col); y += 40
    if extra and nlog >= len(log):
        y += 14
        for t, col in (extra if isinstance(extra, list) else [extra]):
            d.text((24,y),t,font=mono,fill=col); y += 40
    return im
def slide(path, cap):
    im = Image.new("RGB",(W,H),BG); d = ImageDraw.Draw(im)
    pic = Image.open(path).convert("RGB"); s = min(W/pic.width,(H-130)/pic.height)
    pic = pic.resize((int(pic.width*s),int(pic.height*s)),Image.LANCZOS); im.paste(pic,((W-pic.width)//2,20))
    d.text(((W-d.textlength(cap,font=sans))/2,H-80),cap,font=sans,fill=FG); return im

OLD5 = "/home/gautier/projects/isl/issues/karios-3.0.0/2025-KARIOS_DEMO/5/out_clip_work_masked_S2C_MSIL1C_20250304T112141_N0511_R037_T28PDU_B08/"
def text_slide(v, head, lines, secs=5):
    im = Image.new("RGB",(W,H),BG); d = ImageDraw.Draw(im); d.text((140,260),head,font=big,fill=ORANGE); y = 400
    for l in lines: d.text((140,y),l,font=mid,fill=FG); y += 60
    v.fade(im, secs)
ALIGN_KEEP = [r"Start align", r"SIFT feature", r"Matches:", r"RANSAC initial", r"Translation search", r"Gradient correlation", r"Selected alignment", r"Write|Wrote|saved"]
def align_block(v, cmd, run, extra):
    log = loglines(run, ALIGN_KEEP); total = sum(map(len,cmd))
    v.emit(term(cmd,0,0,log), FPS)
    for c in range(0,total+1,5): v.emit(term(cmd,c,0,log))
    v.emit(term(cmd,total,0,log), FPS//2)
    for i in range(1,len(log)+1): v.emit(term(cmd,total,i,log), 6)
    v.emit(term(cmd,total,len(log),log,extra), int(FPS*3.5))
def run_block(v, cmd, run, slides):
    if run == "5old":
        h, r = [l.split() for l in open(OLD5+"correl_res.txt").read().strip().splitlines()[:2]]
        c = dict(zip(h, r)); log = []
        verdict = (f"=> {int(c['sample_pixel'])} key points kept (confidence >= {c['confidence_th']})  |  mean shift x/y: {float(c['mean_x']):+.2f} / {float(c['mean_y']):+.2f} px  |  std {float(c['std_x']):.2f} / {float(c['std_y']):.2f} px", GREEN)
        total = sum(map(len,cmd))
        v.emit(term(cmd,0,0,log), FPS)
        for k in range(0,total+1,5): v.emit(term(cmd,k,0,log))
        v.emit(term(cmd,total,0,log,verdict), int(FPS*3.5))
        for name, cap in slides: v.fade(slide(OLD5+name, cap), 3.5)
        return
    log = loglines(run); sm = summary(run); q = sm["quality"]; st = sm["statistics"]
    conf = round(q["confidence"]*100)
    verdict = (f"=> matching confidence: {q['verdict']} ({conf} %)  |  mean shift x/y: {st['mean_x']:+.2f} / {st['mean_y']:+.2f} px  |  CE90 {st['ce90']:.1f} m",
               GREEN if q["verdict"]=="reliable" else (ORANGE if q["verdict"]=="doubtful" else RED))
    ts = [datetime.datetime.strptime(m.group(1),"%Y-%m-%d %H:%M:%S") for m in (re.search(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]", l) for l in open(D+run+".log", errors="replace")) if m]
    el = int((ts[-1]-ts[0]).total_seconds()); verdict = [verdict, (f"=> processing time: {el//60} min {el%60:02d} s  |  {sm['matched_points']} key points", FG)]
    total = sum(map(len,cmd))
    v.emit(term(cmd,0,0,log), FPS)
    for c in range(0,total+1,5): v.emit(term(cmd,c,0,log))
    v.emit(term(cmd,total,0,log), FPS//2)
    for i in range(1,len(log)+1): v.emit(term(cmd,total,i,log), 4)
    v.emit(term(cmd,total,len(log),log,verdict), int(FPS*2.5))
    for name, cap in slides:
        cap = cap.format(ce90=st["ce90"], ce95=st["ce95"])
        v.fade(slide(img(run,name),cap), 3.5)

STD = lambda run: [("01_overview.png","01_overview.png - overview of the shifts"),("02_dx.png","02_dx.png - shift along X (dx)"),
                   ("03_dy.png","03_dy.png - shift along Y (dy)"),("04_ce.png","04_ce.png - circular error: CE90 {ce90:.1f} m / CE95 {ce95:.1f} m")]
C = "--conf processing_configuration_sans_outliers.json"
SC = {
 "1": dict(slug="ls8-sen2like", name="Landsat 8 / Sen2Like",
   lines=["Monitored: L2H_T34RGT_20171020T085527_LS8_R181_B04_30m (Sen2Like L2H)","Reference: LC81810402017293MTI00_B5 (Landsat 8)","Compare LS8 data with / without surface reflectance"],
   opts="-gip (intermediate product)  |  minDistance 5",
   info=["Both images: 7532 x 7682 px, 1 band UInt16, 30 m pixel (226 x 230 km)","Grid: WGS 84 / UTM zone 34N (EPSG:32634), identical origin (662370, 3311430)","Monitored: Sen2Like L2H B04 (red), 20171020T085527","Reference: Landsat 8 L1 B5 (NIR), orbit 181 / row 040, 20171020","Valid pixels: 41.5 M of 57.9 M (72 %)"],
   blocks=[(["$ karios process --conf processing_configuration_md5.json -gip \\","    WD/L2H_T34RGT_20171020T085527_LS8_R181_B04_30m_clip.tif \\","    WD/LC81810402017293MTI00_B5_work.tif"],"1",None)]),
 "2": dict(slug="s2a-s2c-commissioning", name="Sentinel-2A / Sentinel-2C - commissioning",
   lines=["Monitored: S2A_T30SYJ_20241105T105231_B04","Reference: S2C_T30SYJ_20241105T105251_B04","Investigate the key points mask output"],
   opts="--generate-key-points-mask",
   blocks=[(["$ karios process "+C+" \\","    --generate-key-points-mask \\","    WD/S2A_T30SYJ_20241105T105231_B04_clip.tif \\","    WD/S2C_T30SYJ_20241105T105251_B04_work.tif"],"2",STD("2")+[(S+"qgis/q1.png","QGIS: kp_delta key points over the monitored image"),(S+"qgis/q3c.png","QGIS: metadata (dx, dy, score, ZNCC...) of one key point")])]),
 "3": dict(slug="mss-gri-dem", name="Landsat MSS / Sentinel-2 GRI with a DEM",
   lines=["Monitored: Landsat 5 MSS (B04)","Reference: Sentinel-2 GRI (B8A)","Old data: relief dependency analysed with COP-DEM GLO-90"],
   opts="DEM file argument, --dem-description",
   blocks=[(["$ karios process "+C+" \\","    --dem-description \"cop DEM\" \\","    WD/clip_ref_clip.tif WD/clip_work_work.tif - DEM/dem_win.tif"],"3",
            STD("3")+[("dem_dx.png","dem_dx.png - dx versus altitude"),("dem_dy.png","dem_dy.png - dy versus altitude"),("dem_radial_error.png","dem_radial_error.png - radial error versus altitude")])]),
 "4": dict(slug="spot-bsg-large-offset-align", name="SPOT / BSG - large offset: karios align, then process",
   lines=["Monitored: BSG-103-20191005-105550-464915 ortho ms","Reference: SPOT5_MS","Detect a large offset, correct it with karios align, process again"],
   opts="karios align, then karios process",
   blocks=[(["$ karios process "+C+" \\","    --generate-key-points-mask \\","    WD/BSG-103-20191005-105550-464915_ortho-ms_8b_rpc_clip_work_masked_work.tif \\","    WD/SPOT5_MS_clip.tif"],"4a",
            [("01_overview.png","Before alignment - 01_overview.png")]),
           ("TEXT","Conclusion",["The mean shift is far larger than a pixel: about -8.6 px in x and +10.5 px in y (CE90 above 10 px)","The monitored image has a LARGE OFFSET: KLT matching is unreliable on it","=> correct the offset first with karios align (SIFT + RANSAC + ECC)"]),
           ("ALIGN",["$ karios align --out aligned \\","    WD/BSG-103-20191005-105550-464915_ortho-ms_8b_rpc_clip_work_masked_work.tif \\","    WD/SPOT5_MS_clip.tif","$ gdalwarp -te 646587.5 4820077.5 653782.5 4827862.5 -tr 2.5 2.5 -r bilinear \\","    aligned/..._global_aligned.tif aligned/..._global_aligned_refgrid.tif"],"4al"),
           (["$ karios process "+C+" \\","    --generate-key-points-mask \\","    aligned/BSG-103-..._work_global_aligned_refgrid.tif \\","    WD/SPOT5_MS_clip.tif"],"4c",
            [("01_overview.png","After alignment - 01_overview.png")]+STD("4c")[1:])]),
 "5": dict(slug="sar-sentinel2", name="SAR / Sentinel-2",
   lines=["Monitored: clip_work_masked (ASAR ASA_IMM_1PNDSI2004030, SAR)","Reference: S2C_MSIL1C_20250304T112141_N0511_R037_T28PDU, B08","Multi-sensor matching, SAR against optical"],
   opts="",
   blocks=[(["$ karios process --conf processing_configuration_sans_outliers_1C.json \\","    --out out_clip_work_masked_S2C_MSIL1C_20250304T112141_N0511_R037_T28PDU_B08 \\","    clip_work_masked.TIF S2C_MSIL1C_20250304T112141_N0511_R037_T28PDU_B08.TIFF"],"5old",
            [("01_overview.png","01_overview.png - overview of the shifts"),("02_dx.png","02_dx.png - shift along X (dx)"),("03_dy.png","03_dy.png - shift along Y (dy)"),("04_ce.png","04_ce.png - circular error")])]),
 "6": dict(slug="interband", name="Interband - with key point chips",
   lines=["Monitored: 3122863_3423606_2020-02-10_2257_BGREN_Analytic_band2","Reference: 3122863_3423606_2020-02-10_2257_BGREN_Analytic_band5","Registration between two bands of the same product"],
   opts="-gkc (generate key point chips)",
   blocks=[(["$ karios process "+C+" -gkc \\","    WD/3122863_3423606_2020-02-10_2257_BGREN_Analytic_band2_clip.tif \\","    WD/3122863_3423606_2020-02-10_2257_BGREN_Analytic_band5_work.tif"],"6",STD("6")+[(S+"chips6/c1.png","chips.html - key point chips (-gkc): reference | monitored"),(S+"chips6/c2.png","Laplacian chips, with dx, dy, ZNCC and NMI of each key point")])]),
}
n, outdir = sys.argv[1], sys.argv[2]
sc = SC[n]; os.makedirs(outdir, exist_ok=True)
v = Video(f"{outdir}/scenario-{n}-{sc['slug']}.mp4")
v.fade(title(n, sc["name"], sc["lines"], sc["opts"]), 4)
if sc.get("info"):
    im = Image.new("RGB",(W,H),BG); d = ImageDraw.Draw(im); d.text((140,160),"Input images",font=big,fill=FG); y = 300
    for l in sc["info"]: d.text((140,y),l,font=mid,fill=FG); y += 56
    v.fade(im, 6)
for b in sc["blocks"]:
    if b[0] == "TEXT": text_slide(v, b[1], b[2], 8)
    elif b[0] == "ALIGN":
        al = open(D+b[2]+".log", errors="replace").read(); mm = re.search(r"real\s+(\d+)m([\d.]+)s", al)
        align_block(v, b[1], b[2], (f"=> aligned image + alignment.json written  |  elapsed {mm.group(1)} min {float(mm.group(2)):.0f} s" if mm else "=> aligned image written", GREEN))
    else:
        cmd, run, slides = b; run_block(v, cmd, run, slides or STD(run))
v.close()
