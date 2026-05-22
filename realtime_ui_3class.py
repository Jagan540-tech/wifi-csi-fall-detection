"""
=============================================================================
  WiFi CSI — Advanced 3-Class Realtime Dashboard
  Fall | Walking | No Movement  +  Telegram Alerts
  Run: python realtime_ui.py
       python realtime_ui.py --port COM3
  Open: http://localhost:8765  (auto-opens)
=============================================================================
"""
import threading, json, time, os, sys, argparse
import urllib.request, urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from collections import deque, Counter
from scipy.ndimage import median_filter
import numpy as np, joblib

# ── CONFIG ────────────────────────────────────────────────────────
BAUD_RATE        = 115200
N_SC             = 52
ACTIVITIES       = ["fall","walking","no_movement"]
WINDOW_SIZE      = 50
PREDICT_EVERY    = 20
VOTE_WINDOW      = 3
IMPACT_THRESHOLD = 4.0
UI_PORT          = 8765

MODEL_DIR   = "models_3class"
MODEL_PATH  = f"{MODEL_DIR}/voting_clf.pkl"
SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"
TOP_SC_PATH = f"{MODEL_DIR}/top_sc.npy"

TELEGRAM_TOKEN   = "7663744838:AAEwnym6zaqgYeFqtJ_eRy1GY8zdm85MM9E"
TELEGRAM_CHAT_ID = "1793002051"
ALERT_COOLDOWN   = 30

# ── SHARED STATE ──────────────────────────────────────────────────
lock   = threading.Lock()
state  = {
    "prediction":"no_movement","confidence":0.0,
    "proba":{"fall":0.0,"walking":0.0,"no_movement":0.0},
    "frame_count":0,"pred_count":0,"fps":0.0,
    "source":"idle","voted":False,
    "history":[],"signal":[],"connected":False,
    "running":False,"session_sec":0,
    "durations":{"fall":0,"walking":0,"no_movement":0},
    "conf_history":[],
    "telegram_status":"idle",
}
_esp_thread    = None
_last_alert    = 0
_session_start = 0

# ── TELEGRAM ──────────────────────────────────────────────────────
def send_telegram(source, fc):
    global _last_alert
    now = time.time()
    if now - _last_alert < ALERT_COOLDOWN: return
    _last_alert = now
    ts  = time.strftime("%Y-%m-%d %H:%M:%S")
    src = "⚡ Impact Detector" if source=="impact" else "🤖 ML Model"
    msg = (f"🚨 *FALL DETECTED!*\n\n"
           f"🕐 Time   : `{ts}`\n"
           f"📡 Source : {src}\n"
           f"🎞 Frame  : `{fc}`\n"
           f"📍 System : Home CSI Monitor\n\n"
           f"⚠️ *Please check immediately!*")
    try:
        url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = urllib.parse.urlencode(
            {"chat_id":TELEGRAM_CHAT_ID,"text":msg,"parse_mode":"Markdown"}
        ).encode()
        urllib.request.urlopen(urllib.request.Request(url,data=data), timeout=5)
        with lock: state["telegram_status"] = "sent"
        print("  ✅ Telegram alert sent!")
    except Exception as e:
        with lock: state["telegram_status"] = "failed"
        print(f"  ⚠ Telegram failed: {e}")

# ── PREPROCESSING + FEATURES ──────────────────────────────────────
def preprocess(data):
    c=data.copy(); dead=np.std(c,axis=0)<1e-6; c[:,dead]=0.0
    for i in range(c.shape[1]):
        col=c[:,i]
        if np.std(col)<1e-6: continue
        mu,sig=np.mean(col),np.std(col)
        c[:,i]=np.clip(col,mu-3*sig,mu+3*sig)
        c[:,i]=median_filter(c[:,i],size=5)
    mn,mx=c.min(0),c.max(0)
    return (c-mn)/np.where((mx-mn)<1e-6,1.0,mx-mn)

def features(window):
    T,C=window.shape; feats=[]; diff=np.diff(window,axis=0)
    h1,h2=T//3,T-T//3
    for ci in range(C):
        col=window[:,ci]; dcol=diff[:,ci]; mu=np.mean(col); sig=np.std(col)+1e-9
        feats+=[mu,sig,np.var(col),np.sum(col**2),np.max(col)-np.min(col),
                np.mean(np.abs(dcol)),np.var(dcol),np.max(np.abs(dcol)),
                np.mean(((col-mu)/sig)**3),np.mean(((col-mu)/sig)**4)-3,
                np.sum(np.diff(np.sign(col-mu))!=0)/(T-1)]
        fm=np.abs(np.fft.rfft(col))[1:]; ff=np.fft.rfftfreq(T)[1:]
        if len(fm)>3:
            nf=len(fm); tot=np.sum(fm**2)+1e-9; pk=np.argmax(fm); p=fm**2/tot
            feats+=[fm[pk],ff[pk],-np.sum(p*np.log(p+1e-12)),fm[pk]**2/tot,
                    np.sum(fm[:nf//3]**2)/tot,np.sum(fm[nf//3:2*nf//3]**2)/tot]
        else: feats+=[0.0]*6
        if sig>1e-6:
            cn=(col-mu)/sig; ac=np.correlate(cn,cn,mode="full")[len(cn)-1:]
            ac/=(ac[0]+1e-9); feats.append(ac[min(5,len(ac)-1)])
        else: feats.append(0.0)
        vf=np.var(col[:h1])+1e-9; vl=np.var(col[h2:])+1e-9; vm=np.var(col[h1:h2])+1e-9
        feats+=[vf,vl,vm/(vf+vl),abs(vf-vl)/(vf+vl),
                np.max(np.abs(dcol))/(np.std(dcol)+1e-9)]
        ce=np.cumsum((col-mu)**2); feats.append(np.searchsorted(ce/(ce[-1]+1e-9),0.5)/(T-1))
    feats+=[np.sum(window**2),np.mean(np.abs(diff)),np.max(np.abs(diff))]
    psc=np.std(window,axis=0); feats+=[np.mean(psc),float(np.sum(psc>0.05))]
    active=np.where(psc>1e-3)[0]
    if len(active)>2:
        rs=[np.corrcoef(window[:,active[i]],window[:,active[i+1]])[0,1] for i in range(len(active)-1)]
        feats.append(np.mean([r for r in rs if not np.isnan(r)]) if rs else 0.0)
    else: feats.append(0.0)
    return np.array(feats)

# ── IMPACT DETECTOR ───────────────────────────────────────────────
class ImpactDetector:
    def __init__(self):
        self.prev=None; self.buf=deque(maxlen=50); self.cd=0
    def update(self,row):
        triggered=False
        if self.prev is not None:
            fd=np.linalg.norm(row-self.prev); self.buf.append(fd)
            if self.cd>0: self.cd-=1
            elif len(self.buf)>=10:
                bm=np.mean(list(self.buf)[:-1]); bs=np.std(list(self.buf)[:-1])+1e-9
                if (fd-bm)/bs>IMPACT_THRESHOLD: triggered=True; self.cd=43
        self.prev=row.copy(); return triggered

# ── PORT DETECT ───────────────────────────────────────────────────
def find_port():
    try:
        import serial.tools.list_ports
        ports=list(serial.tools.list_ports.comports())
        for p in ports:
            if any(k in p.description.lower() for k in ["cp210","ch340","ch341","uart","esp32"]):
                return p.device
        if len(ports)==1: return ports[0].device
    except: pass
    return None

# ── SESSION TIMER THREAD ──────────────────────────────────────────
def session_timer():
    global _session_start
    while True:
        time.sleep(1)
        with lock:
            if state["running"] and _session_start>0:
                state["session_sec"]=int(time.time()-_session_start)
                pred=state["prediction"]
                state["durations"][pred]=state["durations"].get(pred,0)+1

# ── ESP32 THREAD ──────────────────────────────────────────────────
def esp32_thread(port, clf, scaler, top_sc):
    global _session_start
    try: import serial
    except: print("pip install pyserial"); sys.exit(1)

    print(f"  Connecting to {port}...")
    try:
        ser=serial.Serial(port,BAUD_RATE,timeout=1); time.sleep(2)
    except Exception as e:
        print(f"  Cannot open: {e}"); sys.exit(1)
    ser.write(b"START\n"); time.sleep(0.3)
    with lock: state["connected"]=True
    print("  ESP32 connected ✓  — waiting for START button in browser")

    rolling=deque(maxlen=WINDOW_SIZE); recent=deque(maxlen=VOTE_WINDOW)
    imp=ImpactDetector(); frames_since=0
    fps_buf=deque(maxlen=43); fps_last=time.time()
    prev_running=False

    while True:
        with lock: running=state["running"]
        if not running:
            try: ser.reset_input_buffer()
            except: pass
            time.sleep(0.05); prev_running=False; continue
        # Just clicked START — flush stale buffer, reset everything
        if not prev_running:
            try: ser.reset_input_buffer()
            except: pass
            rolling.clear(); recent.clear()
            imp=ImpactDetector(); frames_since=0
            fps_buf.clear(); fps_last=time.time()
            print("  ▶ Prediction started!")
            prev_running=True

        try: raw=ser.readline().decode("utf-8",errors="ignore").strip()
        except: continue
        if not raw.startswith("CSI_DATA,"): continue
        try:
            parts=raw[len("CSI_DATA,"):].split(",")
            if len(parts)!=N_SC: continue
            row=np.array([float(v) for v in parts])
        except: continue

        rolling.append(row); frames_since+=1
        fps_buf.append(time.time())
        now=time.time()
        fps=len(fps_buf)/(now-fps_last+1e-9) if now-fps_last>0.5 else state["fps"]
        if now-fps_last>0.5: fps_last=now

        with lock:
            state["frame_count"]+=1; state["fps"]=round(fps,1)
            state["signal"]=row.tolist()

        # Impact detector
        if imp.update(row):
            recent.append("fall")
            with lock:
                fc=state["frame_count"]
                state.update({"prediction":"fall","confidence":99.0,
                    "proba":{"fall":99.0,"walking":0.5,"no_movement":0.5},
                    "source":"impact","voted":False,
                    "pred_count":state["pred_count"]+1,
                    "telegram_status":"sending"})
                h=state["history"]; h.append("fall"); state["history"]=h[-30:]
                ch=state["conf_history"]; ch.append(99.0); state["conf_history"]=ch[-30:]
            threading.Thread(target=send_telegram,args=("impact",fc),daemon=True).start()
            frames_since=0; continue

        if frames_since>=PREDICT_EVERY and len(rolling)==WINDOW_SIZE:
            frames_since=0
            win=preprocess(np.array(rolling))[:,top_sc]
            fs=scaler.transform(features(win).reshape(1,-1))
            proba=clf.predict_proba(fs)[0]
            raw_pred=clf.classes_[np.argmax(proba)]
            conf=float(np.max(proba)*100)
            prob_d={a:float(p*100) for a,p in zip(clf.classes_,proba)}
            recent.append(raw_pred)
            voted=Counter(recent).most_common(1)[0][0]
            is_voted=len(recent)==VOTE_WINDOW
            with lock:
                state.update({"prediction":voted,"confidence":conf,
                    "proba":prob_d,"source":"model","voted":is_voted,
                    "pred_count":state["pred_count"]+1})
                h=state["history"]; h.append(voted); state["history"]=h[-30:]
                ch=state["conf_history"]; ch.append(conf); state["conf_history"]=ch[-30:]
            if voted=="fall":
                with lock: fc=state["frame_count"]
                threading.Thread(target=send_telegram,args=("model",fc),daemon=True).start()

# ── HTML DASHBOARD ────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Realtime Predictions</title>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Fira+Code:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root{
  --fall:#ff1744;--walk:#00e676;--nomv:#2979ff;
  --bg:#02050a;--p1:#060d16;--p2:#0a1628;
  --cyan:#00e5ff;--gold:#ffd600;
  --bd:#0d2035;--txt:#7ab3cc;--dim:#112030;
}
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;overflow:hidden}
body{
  background:var(--bg);color:var(--txt);
  font-family:'Fira Code',monospace;font-size:13px;
}

/* ── HEX GRID BG ── */
body::before{
  content:'';position:fixed;inset:0;z-index:0;opacity:.4;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='56' height='100'%3E%3Cpath d='M28 66L0 50V18L28 2l28 16v32zm0 0l28 16v18M28 66v18M0 68l28 16' stroke='%230d2035' stroke-width='1' fill='none'/%3E%3C/svg%3E");
}

/* ── SCANLINES ── */
body::after{
  content:'';position:fixed;inset:0;z-index:1;pointer-events:none;
  background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,.06) 3px,rgba(0,0,0,.06) 4px);
}

.app{position:relative;z-index:2;height:100vh;display:flex;flex-direction:column;padding:10px;gap:8px}

/* ── HEADER ── */
header{
  display:flex;align-items:center;justify-content:space-between;
  background:var(--p1);border:1px solid var(--bd);border-radius:4px;
  padding:10px 20px;flex-shrink:0;
  box-shadow:0 0 30px rgba(0,229,255,.04);
}
.logo{
  font-family:'Rajdhani',sans-serif;font-weight:700;font-size:1.3rem;
  letter-spacing:6px;color:#fff;text-transform:uppercase;
}
.logo span{color:var(--cyan);text-shadow:0 0 20px var(--cyan)}
.header-mid{text-align:center}
.header-title{
  font-family:'Rajdhani',sans-serif;font-size:1.05rem;font-weight:600;
  letter-spacing:4px;color:var(--cyan);opacity:.7;
}
.session-time{
  font-size:.75rem;letter-spacing:2px;color:var(--gold);
  font-family:'Rajdhani',sans-serif;font-weight:600;
}
.header-right{display:flex;align-items:center;gap:12px}
.stat-pill{
  background:var(--dim);border:1px solid var(--bd);border-radius:3px;
  padding:4px 10px;font-size:.65rem;letter-spacing:2px;color:#3a6080;
}
.stat-pill span{color:var(--cyan)}

/* ── START/STOP BUTTON ── */
.btn-start{
  padding:8px 28px;border-radius:3px;border:none;cursor:pointer;
  font-family:'Rajdhani',sans-serif;font-weight:700;font-size:.95rem;
  letter-spacing:4px;text-transform:uppercase;transition:all .2s;
}
.btn-start.start{
  background:transparent;border:2px solid var(--walk);color:var(--walk);
  box-shadow:0 0 20px rgba(0,230,118,.2);
}
.btn-start.start:hover{background:rgba(0,230,118,.1);box-shadow:0 0 30px rgba(0,230,118,.4)}
.btn-start.stop{
  background:transparent;border:2px solid var(--fall);color:var(--fall);
  box-shadow:0 0 20px rgba(255,23,68,.2);
}
.btn-start.stop:hover{background:rgba(255,23,68,.1);box-shadow:0 0 30px rgba(255,23,68,.4)}

/* ── MAIN GRID ── */
.main{flex:1;display:grid;grid-template-columns:220px 1fr 220px;gap:8px;min-height:0}

/* ── PANELS ── */
.panel{
  background:var(--p1);border:1px solid var(--bd);border-radius:4px;
  padding:14px;position:relative;overflow:hidden;display:flex;flex-direction:column;gap:10px;
}
.panel::before{
  content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--cyan),transparent);opacity:.2;
}
.plabel{font-size:.55rem;letter-spacing:3px;color:#1a3a5a;text-transform:uppercase;margin-bottom:2px}

/* ── LEFT PANEL ── */
.left-panel{}

/* FPS + FRAMES ── */
.metric-box{background:var(--dim);border:1px solid var(--bd);border-radius:3px;padding:8px 10px}
.metric-val{font-family:'Rajdhani',sans-serif;font-size:1.6rem;font-weight:700;color:var(--cyan);letter-spacing:1px}
.metric-sub{font-size:.58rem;letter-spacing:2px;color:#1a3a5a;margin-top:1px}

/* Connection status */
.conn-row{display:flex;align-items:center;gap:8px}
.dot{width:8px;height:8px;border-radius:50%;background:#333;flex-shrink:0;transition:all .3s}
.dot.on{background:var(--walk);box-shadow:0 0 10px var(--walk)}
.dot.warn{background:var(--gold);box-shadow:0 0 10px var(--gold);animation:blink .8s infinite}
.dot.err{background:var(--fall);box-shadow:0 0 10px var(--fall);animation:blink .5s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
.conn-txt{font-size:.65rem;letter-spacing:1px}

/* Conf trend sparkline */
#sparkCanvas{width:100%;height:52px;display:block;border-radius:2px}

/* Activity durations */
.dur-row{display:flex;align-items:center;gap:8px;margin-bottom:4px}
.dur-icon{font-size:.9rem}
.dur-bar-bg{flex:1;height:5px;background:var(--dim);border-radius:3px;overflow:hidden}
.dur-bar{height:100%;border-radius:3px;transition:width .5s ease}
.dur-bar.fall{background:var(--fall);box-shadow:0 0 6px var(--fall)}
.dur-bar.walking{background:var(--walk);box-shadow:0 0 6px var(--walk)}
.dur-bar.no_movement{background:var(--nomv);box-shadow:0 0 6px var(--nomv)}
.dur-sec{font-size:.6rem;color:#2a4a6a;min-width:28px;text-align:right}

/* Telegram status */
.tg-row{display:flex;align-items:center;gap:6px;font-size:.62rem;letter-spacing:1px}
.tg-dot{width:6px;height:6px;border-radius:50%;background:#1a3a5a}
.tg-dot.sent{background:#00e676;box-shadow:0 0 8px #00e676}
.tg-dot.sending{background:var(--gold);animation:blink .4s infinite}
.tg-dot.failed{background:var(--fall)}

/* ── CENTER PANEL ── */
.center-panel{align-items:center;justify-content:center}

/* Activity name */
.act-name{
  font-family:'Rajdhani',sans-serif;font-weight:700;
  font-size:2.4rem;letter-spacing:8px;text-transform:uppercase;
  transition:all .3s;text-align:center;
}
.act-name.fall{color:var(--fall);text-shadow:0 0 30px var(--fall),0 0 60px rgba(255,23,68,.3)}
.act-name.walking{color:var(--walk);text-shadow:0 0 30px var(--walk),0 0 60px rgba(0,230,118,.3)}
.act-name.no_movement{color:var(--nomv);text-shadow:0 0 30px var(--nomv),0 0 60px rgba(41,121,255,.3)}

.pred-meta{font-size:.65rem;letter-spacing:2px;color:#2a4a6a;margin-top:4px;text-align:center}
.pred-meta span{color:var(--cyan)}

/* Circular confidence ring */
.ring-wrap{position:relative;width:240px;height:240px;flex-shrink:0}
.ring-svg{width:100%;height:100%;transform:rotate(-90deg)}
.ring-bg{fill:none;stroke:var(--dim);stroke-width:6}
.ring-fill{fill:none;stroke-width:6;stroke-linecap:round;
  transition:stroke-dashoffset .5s ease,stroke .4s ease}
.ring-pct{
  position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  font-family:'Rajdhani',sans-serif;font-weight:700;font-size:2rem;
  color:#fff;text-align:center;pointer-events:none;
}
.ring-pct-label{font-size:.6rem;letter-spacing:2px;color:#2a4a6a;display:block;margin-top:-4px}

/* Stickman inside ring */
.stickman-container{position:absolute;inset:30px;display:flex;align-items:center;justify-content:center}
.stick-svg{width:120px;height:150px;filter:drop-shadow(0 0 12px currentColor);transition:color .4s}
.stick-svg.fall{color:var(--fall)}
.stick-svg.walking{color:var(--walk)}
.stick-svg.no_movement{color:var(--nomv)}
.stick-svg circle,.stick-svg line{stroke:currentColor;fill:none;stroke-linecap:round;stroke-linejoin:round}
.stick-svg circle.head{fill:currentColor;opacity:.15;stroke-width:2.5}

/* Walking anim */
@keyframes wb{0%,100%{transform:translateY(0)}50%{transform:translateY(-4px)}}
@keyframes wla{0%,100%{transform:rotate(-32deg)}50%{transform:rotate(32deg)}}
@keyframes wra{0%,100%{transform:rotate(32deg)}50%{transform:rotate(-32deg)}}
@keyframes wll{0%,100%{transform:rotate(-28deg)}50%{transform:rotate(28deg)}}
@keyframes wrl{0%,100%{transform:rotate(28deg)}50%{transform:rotate(-28deg)}}
.walking .wbody{animation:wb .55s ease-in-out infinite;transform-origin:60px 78px}
.walking .wla{animation:wla .55s ease-in-out infinite;transform-origin:60px 88px}
.walking .wra{animation:wra .55s ease-in-out infinite;transform-origin:60px 88px}
.walking .wll{animation:wll .55s ease-in-out infinite;transform-origin:60px 114px}
.walking .wrl{animation:wrl .55s ease-in-out infinite;transform-origin:60px 114px}
/* Breathing */
@keyframes breathe{0%,100%{transform:scaleY(1)}50%{transform:scaleY(1.03)}}
.no_movement .wbody{animation:breathe 3.5s ease-in-out infinite;transform-origin:60px 100px}
/* Fall */
@keyframes falling{0%{transform:rotate(0) translateY(0)}70%{transform:rotate(-85deg) translateY(35px)}100%{transform:rotate(-90deg) translateY(38px)}}
@keyframes fall-flash{0%,100%{opacity:1}50%{opacity:.5}}
.fall .fall-g{animation:falling .7s cubic-bezier(.25,.46,.45,.94) forwards;transform-origin:60px 140px}

/* History row */
.hist-wrap{display:flex;gap:4px;flex-wrap:wrap;justify-content:center}
.hdot{width:14px;height:14px;border-radius:2px;background:var(--dim);transition:all .3s;cursor:default}
.hdot.fall{background:var(--fall);box-shadow:0 0 6px var(--fall)}
.hdot.walking{background:var(--walk);box-shadow:0 0 6px var(--walk)}
.hdot.no_movement{background:var(--nomv);box-shadow:0 0 6px var(--nomv)}

/* CSI Signal */
#sigCanvas{width:100%;height:58px;display:block}

/* ── RIGHT PANEL ── */
.right-panel{}

/* Prob arcs */
.prob-arc-wrap{display:flex;flex-direction:column;gap:12px}
.prob-item{}
.prob-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:5px}
.prob-name{font-size:.62rem;letter-spacing:2px}
.prob-name.fall{color:var(--fall)}
.prob-name.walking{color:var(--walk)}
.prob-name.no_movement{color:var(--nomv)}
.prob-pct{font-family:'Rajdhani',sans-serif;font-size:1rem;font-weight:700;color:#fff}
.prob-bar-bg{height:8px;background:var(--dim);border-radius:4px;overflow:hidden;position:relative}
.prob-bar-fill{height:100%;border-radius:4px;transition:width .4s ease}
.prob-bar-fill.fall{background:linear-gradient(90deg,#7b0020,var(--fall));box-shadow:0 0 10px var(--fall)}
.prob-bar-fill.walking{background:linear-gradient(90deg,#006035,var(--walk));box-shadow:0 0 10px var(--walk)}
.prob-bar-fill.no_movement{background:linear-gradient(90deg,#002a80,var(--nomv));box-shadow:0 0 10px var(--nomv)}
.prob-bar-tick{position:absolute;top:0;bottom:0;width:1px;background:rgba(255,255,255,.15)}

/* Source / voted tag */
.tag{
  display:inline-block;font-size:.55rem;letter-spacing:2px;
  padding:2px 8px;border-radius:2px;border:1px solid var(--bd);color:#1a3a5a;margin-top:4px;
}
.tag.impact{border-color:var(--fall);color:var(--fall)}
.tag.voted{border-color:var(--walk);color:var(--walk)}
.tag.model{border-color:var(--cyan);color:var(--cyan)}

/* Divider */
.divider{height:1px;background:linear-gradient(90deg,transparent,var(--bd),transparent)}

/* ── FALL OVERLAY ── */
#fallOverlay{
  display:none;position:fixed;inset:0;z-index:100;pointer-events:none;
  border:4px solid var(--fall);
  animation:fall-pulse .25s ease infinite alternate;
}
#fallOverlay.show{display:block}
@keyframes fall-pulse{
  from{box-shadow:inset 0 0 60px rgba(255,23,68,.1)}
  to{box-shadow:inset 0 0 140px rgba(255,23,68,.3)}
}
#fallOverlay .fall-banner{
  position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  font-family:'Rajdhani',sans-serif;font-size:2.5rem;font-weight:700;
  color:var(--fall);text-shadow:0 0 40px var(--fall);letter-spacing:8px;
  animation:fall-flash .25s infinite;white-space:nowrap;
}

/* ── IDLE SCREEN ── */
#idleScreen{
  position:fixed;inset:0;z-index:50;
  background:rgba(2,5,10,.92);
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:24px;
  backdrop-filter:blur(4px);
}
#idleScreen.hidden{display:none}
.idle-title{
  font-family:'Rajdhani',sans-serif;font-size:2.2rem;font-weight:700;
  letter-spacing:8px;color:#fff;text-align:center;
}
.idle-title span{color:var(--cyan);text-shadow:0 0 30px var(--cyan)}
.idle-sub{font-size:.75rem;letter-spacing:3px;color:#2a4a6a}
.idle-conn{font-size:.7rem;letter-spacing:2px;color:#1a3a5a}
.idle-conn.ok{color:var(--walk)}
.btn-big{
  padding:14px 48px;border-radius:4px;border:2px solid var(--walk);
  background:transparent;color:var(--walk);cursor:pointer;
  font-family:'Rajdhani',sans-serif;font-weight:700;font-size:1.2rem;letter-spacing:6px;
  text-transform:uppercase;transition:all .2s;
  box-shadow:0 0 30px rgba(0,230,118,.2);
}
.btn-big:hover{background:rgba(0,230,118,.1);box-shadow:0 0 50px rgba(0,230,118,.4);transform:scale(1.02)}
.btn-big:disabled{opacity:.4;cursor:not-allowed;transform:none}
.spin{width:36px;height:36px;border-radius:50%;border:2px solid var(--dim);border-top-color:var(--cyan);animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* BOTTOM signal strip */
.bottom{flex-shrink:0}
</style>
</head>
<body>

<div id="fallOverlay"><div class="fall-banner">⚠ FALL DETECTED ⚠</div></div>

<!-- IDLE / START screen -->
<div id="idleScreen">
  <div class="idle-title">REALTIME<br><span>PREDICTIONS</span></div>
  <div class="idle-sub">WiFi CSI · 52 Subcarriers · 3-Class Detection</div>
  <div class="spin" id="idleSpin"></div>
  <div class="idle-conn" id="idleConn">Connecting to ESP32...</div>
  <button class="btn-big" id="btnBig" onclick="startSession()" disabled>START</button>
  <div class="idle-sub" style="color:#1a3a5a">Fall · Walking · No Movement</div>
</div>

<div class="app">
  <!-- HEADER -->
  <header>
    <div class="logo">CSI<span>.</span>HAR</div>
    <div class="header-mid">
      <div class="header-title">REALTIME PREDICTIONS</div>
      <div class="session-time" id="sessionTime">00:00:00</div>
    </div>
    <div class="header-right">
      <div class="stat-pill">FPS <span id="fpsVal">0</span></div>
      <div class="stat-pill">FRAMES <span id="frameVal">0</span></div>
      <div class="stat-pill">PREDS <span id="predVal">0</span></div>
      <button class="btn-start stop" id="btnStop" onclick="stopSession()">STOP</button>
    </div>
  </header>

  <!-- MAIN -->
  <div class="main">

    <!-- LEFT -->
    <div class="panel left-panel">
      <div class="plabel">System Status</div>
      <div class="conn-row">
        <div class="dot on" id="connDot"></div>
        <div class="conn-txt" id="connTxt">ESP32 CONNECTED</div>
      </div>
      <div class="conn-row" style="margin-top:2px">
        <div class="tg-dot" id="tgDot"></div>
        <div class="tg-row" id="tgTxt">Telegram: idle</div>
      </div>

      <div class="divider"></div>

      <div class="plabel">Confidence Trend</div>
      <canvas id="sparkCanvas"></canvas>

      <div class="divider"></div>

      <div class="plabel">Activity Duration</div>
      <div>
        <div class="dur-row">
          <span class="dur-icon">🚨</span>
          <div class="dur-bar-bg"><div class="dur-bar fall" id="durFall" style="width:0%"></div></div>
          <div class="dur-sec" id="durFallSec">0s</div>
        </div>
        <div class="dur-row">
          <span class="dur-icon">🚶</span>
          <div class="dur-bar-bg"><div class="dur-bar walking" id="durWalk" style="width:0%"></div></div>
          <div class="dur-sec" id="durWalkSec">0s</div>
        </div>
        <div class="dur-row">
          <span class="dur-icon">🧍</span>
          <div class="dur-bar-bg"><div class="dur-bar no_movement" id="durNomv" style="width:0%"></div></div>
          <div class="dur-sec" id="durNomvSec">0s</div>
        </div>
      </div>

      <div class="divider"></div>
      <div class="plabel">Prediction History</div>
      <div class="hist-wrap" id="histWrap"></div>
    </div>

    <!-- CENTER -->
    <div class="panel center-panel">
      <div class="act-name no_movement" id="actName">NO MOVEMENT</div>
      <div class="pred-meta">
        CONFIDENCE <span id="confNum">0.0</span>% &nbsp;
        <span class="tag" id="srcTag">IDLE</span>
      </div>

      <!-- Ring + Stickman -->
      <div class="ring-wrap">
        <svg class="ring-svg" viewBox="0 0 200 200">
          <circle class="ring-bg" cx="100" cy="100" r="90"/>
          <circle class="ring-fill" id="ringFill" cx="100" cy="100" r="90"
            stroke="var(--nomv)"
            stroke-dasharray="565.5"
            stroke-dashoffset="565.5"/>
        </svg>
        <div class="ring-pct">
          <span id="ringPct">0</span>%
          <span class="ring-pct-label">CONF</span>
        </div>
        <div class="stickman-container">
          <svg class="stick-svg no_movement" id="stickSvg" viewBox="0 0 120 155">
            <g class="wbody">
              <circle class="head" cx="60" cy="22" r="18" stroke-width="2"/>
              <line x1="60" y1="40" x2="60" y2="114" stroke-width="2.5"/>
              <line class="wla" x1="60" y1="62" x2="34" y2="90" stroke-width="2.5"/>
              <line class="wra" x1="60" y1="62" x2="86" y2="90" stroke-width="2.5"/>
              <line class="wll" x1="60" y1="114" x2="38" y2="148" stroke-width="2.5"/>
              <line class="wrl" x1="60" y1="114" x2="82" y2="148" stroke-width="2.5"/>
            </g>
          </svg>
        </div>
      </div>

      <div class="divider" style="width:100%"></div>
      <div class="plabel" style="text-align:center">Live CSI Signal</div>
      <canvas id="sigCanvas" class="bottom"></canvas>
    </div>

    <!-- RIGHT -->
    <div class="panel right-panel">
      <div class="plabel">Class Probabilities</div>
      <div class="prob-arc-wrap">
        <div class="prob-item">
          <div class="prob-header">
            <span class="prob-name fall">FALL</span>
            <span class="prob-pct" id="pFall">0%</span>
          </div>
          <div class="prob-bar-bg">
            <div class="prob-bar-fill fall" id="bFall" style="width:0%"></div>
            <div class="prob-bar-tick" style="left:50%"></div>
          </div>
        </div>
        <div class="prob-item">
          <div class="prob-header">
            <span class="prob-name walking">WALKING</span>
            <span class="prob-pct" id="pWalk">0%</span>
          </div>
          <div class="prob-bar-bg">
            <div class="prob-bar-fill walking" id="bWalk" style="width:0%"></div>
            <div class="prob-bar-tick" style="left:50%"></div>
          </div>
        </div>
        <div class="prob-item">
          <div class="prob-header">
            <span class="prob-name no_movement">NO MOVEMENT</span>
            <span class="prob-pct" id="pNomv">0%</span>
          </div>
          <div class="prob-bar-bg">
            <div class="prob-bar-fill no_movement" id="bNomv" style="width:0%"></div>
            <div class="prob-bar-tick" style="left:50%"></div>
          </div>
        </div>
      </div>

      <div class="divider"></div>
      <div class="plabel">Prediction #</div>
      <div class="metric-box">
        <div class="metric-val" id="bigPred">--</div>
        <div class="metric-sub">TOTAL PREDICTIONS</div>
      </div>

      <div class="divider"></div>
      <div class="plabel">Session Stats</div>
      <div class="metric-box">
        <div class="metric-val" id="bigFrames">0</div>
        <div class="metric-sub">FRAMES COLLECTED</div>
      </div>
    </div>

  </div>
</div>

<script>
// ── STICKMAN TEMPLATES ─────────────────────────────────────────
const STICK = {
  walking:`<g class="wbody">
    <circle class="head" cx="60" cy="22" r="18" stroke-width="2"/>
    <line x1="60" y1="40" x2="60" y2="114" stroke-width="2.5"/>
    <line class="wla" x1="60" y1="62" x2="30" y2="88" stroke-width="2.5"/>
    <line class="wra" x1="60" y1="62" x2="90" y2="88" stroke-width="2.5"/>
    <line class="wll" x1="60" y1="114" x2="34" y2="150" stroke-width="2.5"/>
    <line class="wrl" x1="60" y1="114" x2="86" y2="150" stroke-width="2.5"/>
  </g>`,
  no_movement:`<g class="wbody">
    <circle class="head" cx="60" cy="22" r="18" stroke-width="2"/>
    <line x1="60" y1="40" x2="60" y2="114" stroke-width="2.5"/>
    <line class="wla" x1="60" y1="62" x2="34" y2="90" stroke-width="2.5"/>
    <line class="wra" x1="60" y1="62" x2="86" y2="90" stroke-width="2.5"/>
    <line class="wll" x1="60" y1="114" x2="38" y2="148" stroke-width="2.5"/>
    <line class="wrl" x1="60" y1="114" x2="82" y2="148" stroke-width="2.5"/>
  </g>`,
  fall:`<g class="fall-g">
    <circle class="head" cx="60" cy="22" r="18" stroke-width="2"/>
    <line x1="60" y1="40" x2="60" y2="114" stroke-width="2.5"/>
    <line class="wla" x1="60" y1="70" x2="28" y2="48" stroke-width="2.5"/>
    <line class="wra" x1="60" y1="70" x2="92" y2="48" stroke-width="2.5"/>
    <line class="wll" x1="60" y1="114" x2="26" y2="138" stroke-width="2.5"/>
    <line class="wrl" x1="60" y1="114" x2="94" y2="130" stroke-width="2.5"/>
  </g>`
};
const LABELS={fall:'FALL',walking:'WALKING',no_movement:'NO MOVEMENT'};
const COLVARS={fall:'var(--fall)',walking:'var(--walk)',no_movement:'var(--nomv)'};
const CIRC=565.5; // 2π×90

let cur='no_movement', running=false, fallTimer=null;
const sigBuf=[]; const SIG_MAX=100;
const sparkBuf=[]; const SPARK_MAX=30;

// ── Audio (fall alert beep) ────────────────────────────────────
let audioCtx=null;
function beep(){
  try{
    if(!audioCtx) audioCtx=new(window.AudioContext||window.webkitAudioContext)();
    [880,660,880].forEach((f,i)=>{
      const o=audioCtx.createOscillator(), g=audioCtx.createGain();
      o.connect(g); g.connect(audioCtx.destination);
      o.frequency.value=f; o.type='square';
      g.gain.setValueAtTime(.3,audioCtx.currentTime+i*.15);
      g.gain.exponentialRampToValueAtTime(.001,audioCtx.currentTime+i*.15+.12);
      o.start(audioCtx.currentTime+i*.15);
      o.stop(audioCtx.currentTime+i*.15+.13);
    });
  }catch(e){}
}

// ── START / STOP ───────────────────────────────────────────────
async function startSession(){
  await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'start'})});
  document.getElementById('idleScreen').classList.add('hidden');
  running=true;
}
async function stopSession(){
  await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'stop'})});
  running=false;
  document.getElementById('idleScreen').classList.remove('hidden');
  document.getElementById('idleConn').textContent='Session stopped. Click START to resume.';
  document.getElementById('idleConn').className='idle-conn ok';
  document.getElementById('btnBig').disabled=false;
  document.getElementById('idleSpin').style.display='none';
}

// ── UPDATE UI ─────────────────────────────────────────────────
function fmtTime(s){
  const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),ss=s%60;
  return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(ss).padStart(2,'0')}`;
}

function update(d){
  // Connection
  if(d.connected){
    document.getElementById('idleConn').textContent='ESP32 Connected ✓  Ready!';
    document.getElementById('idleConn').className='idle-conn ok';
    document.getElementById('btnBig').disabled=false;
    document.getElementById('idleSpin').style.display='none';
  }

  if(!d.running) return;

  const act=d.prediction;
  // Stickman + label
  if(act!==cur){
    cur=act;
    const svg=document.getElementById('stickSvg');
    svg.innerHTML=STICK[act]; svg.className='stick-svg '+act;
    const lbl=document.getElementById('actName');
    lbl.textContent=LABELS[act]; lbl.className='act-name '+act;
  }

  // Confidence ring
  const conf=d.confidence;
  const offset=CIRC-(conf/100)*CIRC;
  const rf=document.getElementById('ringFill');
  rf.style.strokeDashoffset=offset;
  rf.style.stroke=COLVARS[act];
  document.getElementById('ringPct').textContent=Math.round(conf);
  document.getElementById('confNum').textContent=conf.toFixed(1);

  // Source tag
  const tag=document.getElementById('srcTag');
  if(d.source==='impact'){tag.textContent='IMPACT';tag.className='tag impact';}
  else if(d.voted){tag.textContent='VOTED';tag.className='tag voted';}
  else{tag.textContent='LIVE';tag.className='tag model';}

  // Probs
  const pr=d.proba;
  ['Fall','Walk','Nomv'].forEach((k,i)=>{
    const key=['fall','walking','no_movement'][i];
    const v=(pr[key]||0).toFixed(0);
    document.getElementById('b'+k).style.width=v+'%';
    document.getElementById('p'+k).textContent=v+'%';
  });

  // Stats
  document.getElementById('fpsVal').textContent=d.fps||0;
  document.getElementById('frameVal').textContent=(d.frame_count||0).toLocaleString();
  document.getElementById('predVal').textContent=d.pred_count||0;
  document.getElementById('bigPred').textContent=d.pred_count||0;
  document.getElementById('bigFrames').textContent=(d.frame_count||0).toLocaleString();
  document.getElementById('sessionTime').textContent=fmtTime(d.session_sec||0);

  // Activity durations
  const durs=d.durations||{}; const total=Object.values(durs).reduce((a,b)=>a+b,1);
  const durMap={fall:'Fall',walking:'Walk',no_movement:'Nomv'};
  Object.entries(durMap).forEach(([k,s])=>{
    const sec=durs[k]||0;
    document.getElementById('dur'+s).style.width=(sec/total*100)+'%';
    document.getElementById('dur'+s+'Sec').textContent=sec+'s';
  });

  // Telegram
  const tgS=d.telegram_status||'idle';
  const td=document.getElementById('tgDot'), tt=document.getElementById('tgTxt');
  td.className='tg-dot '+(tgS==='sent'?'sent':tgS==='sending'?'sending':tgS==='failed'?'failed':'');
  tt.textContent='Telegram: '+tgS;

  // History dots
  const hw=document.getElementById('histWrap');
  hw.innerHTML='';
  const hist=[...Array(Math.max(0,20-(d.history||[]).length)).fill(''),
              ...(d.history||[]).slice(-20)];
  hist.forEach(h=>{
    const dv=document.createElement('div');
    dv.className='hdot'+(h?' '+h:'');
    dv.title=h||''; hw.appendChild(dv);
  });

  // Spark (confidence trend)
  if(d.conf_history) drawSpark(d.conf_history);

  // Signal
  if(d.signal&&d.signal.length===52){
    const avg=d.signal.reduce((a,b)=>a+b,0)/d.signal.length;
    sigBuf.push(avg); if(sigBuf.length>SIG_MAX) sigBuf.shift();
    drawSig(act);
  }

  // Fall overlay + sound
  if(act==='fall'&&cur==='fall'){
    document.getElementById('fallOverlay').classList.add('show');
    if(fallTimer) clearTimeout(fallTimer);
    else beep();
    fallTimer=setTimeout(()=>{
      document.getElementById('fallOverlay').classList.remove('show');
      fallTimer=null;
    },3000);
  }
}

// ── SPARK CANVAS ──────────────────────────────────────────────
function drawSpark(hist){
  const c=document.getElementById('sparkCanvas');
  const W=c.offsetWidth,H=c.offsetHeight||52;
  c.width=W; c.height=H;
  const ctx=c.getContext('2d');
  ctx.clearRect(0,0,W,H);
  if(hist.length<2) return;
  ctx.strokeStyle='var(--cyan)'; ctx.lineWidth=1.5;
  ctx.shadowBlur=8; ctx.shadowColor='var(--cyan)';
  ctx.beginPath();
  hist.slice(-SPARK_MAX).forEach((v,i)=>{
    const x=(i/(Math.min(hist.length,SPARK_MAX)-1))*W;
    const y=H-(v/100)*(H*.85)-H*.05;
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  });
  ctx.stroke();
  ctx.shadowBlur=0;
  ctx.lineTo(W,H); ctx.lineTo(0,H); ctx.closePath();
  ctx.fillStyle='rgba(0,229,255,.06)'; ctx.fill();
}

// ── SIGNAL CANVAS ─────────────────────────────────────────────
function drawSig(act){
  const c=document.getElementById('sigCanvas');
  const W=c.offsetWidth,H=c.offsetHeight||58;
  c.width=W; c.height=H;
  const ctx=c.getContext('2d');
  ctx.clearRect(0,0,W,H);
  // Grid
  ctx.strokeStyle='rgba(13,32,53,.9)'; ctx.lineWidth=1;
  for(let x=0;x<W;x+=50){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke()}
  for(let y=0;y<H;y+=20){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke()}
  if(sigBuf.length<2) return;
  const col=act==='fall'?'#ff1744':act==='walking'?'#00e676':'#2979ff';
  const mn=Math.min(...sigBuf),mx=Math.max(...sigBuf);
  ctx.shadowBlur=12; ctx.shadowColor=col;
  ctx.strokeStyle=col; ctx.lineWidth=2;
  ctx.beginPath();
  sigBuf.forEach((v,i)=>{
    const x=(i/(SIG_MAX-1))*W;
    const y=H-((v-mn)/(mx-mn+1e-6))*(H*.8)-H*.08;
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  });
  ctx.stroke(); ctx.shadowBlur=0;
  ctx.lineTo((sigBuf.length-1)/(SIG_MAX-1)*W,H);
  ctx.lineTo(0,H); ctx.closePath();
  ctx.fillStyle=col+'18'; ctx.fill();
}

// ── POLL ──────────────────────────────────────────────────────
async function poll(){
  try{const r=await fetch('/api/state');if(r.ok)update(await r.json());}catch(e){}
  setTimeout(poll,280);
}
poll();
</script>
</body></html>"""

# ── HTTP HANDLER ──────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/","/index.html"):
            self.send_response(200)
            self.send_header("Content-Type","text/html; charset=utf-8")
            self.end_headers(); self.wfile.write(HTML.encode())
        elif self.path=="/api/state":
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Access-Control-Allow-Origin","*")
            self.end_headers()
            with lock: data=dict(state)
            self.wfile.write(json.dumps(data).encode())
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        global _session_start
        if self.path=="/api/control":
            length=int(self.headers.get("Content-Length",0))
            body=json.loads(self.rfile.read(length))
            action=body.get("action","")
            with lock:
                if action=="start":
                    state["running"]=True
                    if _session_start==0: _session_start=time.time()
                else:
                    state["running"]=False
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.end_headers(); self.wfile.write(b'{"ok":true}')
        else:
            self.send_response(404); self.end_headers()

    def log_message(self,*a): pass

# ── MAIN ──────────────────────────────────────────────────────────
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--port",type=str,default=None)
    args=parser.parse_args()

    print("\n"+"="*52)
    print("  WiFi CSI — 3-Class Advanced Dashboard")
    print("="*52)

    print("\n  Loading model...")
    for p in [MODEL_PATH,SCALER_PATH,TOP_SC_PATH]:
        if not os.path.exists(p):
            print(f"  Not found: {p}\n  Run three_class_classifier.py first!")
            sys.exit(1)
    clf=joblib.load(MODEL_PATH); scaler=joblib.load(SCALER_PATH); top_sc=np.load(TOP_SC_PATH)
    print(f"  Model loaded ✓  Classes: {list(clf.classes_)}")

    port=args.port or find_port()
    if not port:
        print("  No ESP32 port found. Run: python realtime_ui.py --port COM3"); sys.exit(1)

    # Session timer thread
    threading.Thread(target=session_timer,daemon=True).start()
    # ESP32 thread
    threading.Thread(target=esp32_thread,args=(port,clf,scaler,top_sc),daemon=True).start()

    server=HTTPServer(("0.0.0.0",UI_PORT),Handler)
    print(f"\n  Dashboard → http://localhost:{UI_PORT}")
    print(f"  Click START in browser to begin")
    print(f"  Ctrl+C to stop\n")

    try:
        import webbrowser; time.sleep(2); webbrowser.open(f"http://localhost:{UI_PORT}")
    except: pass

    try: server.serve_forever()
    except KeyboardInterrupt: print("\n  Stopped.")