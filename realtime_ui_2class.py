"""
=============================================================================
  WiFi CSI — Advanced Binary Dashboard
  Walking | No Movement
  Run: python realtime_ui_binary.py
       python realtime_ui_binary.py --port COM3
  Open: http://localhost:8766  (auto-opens)
=============================================================================
"""
import threading, json, time, os, sys, argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from collections import deque, Counter
from scipy.ndimage import median_filter
import numpy as np, joblib

# ── CONFIG ────────────────────────────────────────────────────────
BAUD_RATE     = 115200
N_SC          = 52
ACTIVITIES    = ["walking","no_movement"]
WINDOW_SIZE   = 80
PREDICT_EVERY = 43
VOTE_WINDOW   = 3
UI_PORT       = 8766

MODEL_DIR   = "models_binary"
MODEL_PATH  = f"{MODEL_DIR}/voting_clf.pkl"
SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"
TOP_SC_PATH = f"{MODEL_DIR}/top_sc.npy"

# ── SHARED STATE ──────────────────────────────────────────────────
lock  = threading.Lock()
state = {
    "prediction":"no_movement","confidence":0.0,
    "proba":{"walking":0.0,"no_movement":0.0},
    "frame_count":0,"pred_count":0,"fps":0.0,
    "voted":False,"history":[],"signal":[],
    "connected":False,"running":False,"session_sec":0,
    "durations":{"walking":0,"no_movement":0},
    "conf_history":[],
}
_session_start = 0

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
    feats+=[np.sum(window**2),np.mean(np.abs(diff)),np.max(np.abs(diff))]
    psc=np.std(window,axis=0); feats+=[np.mean(psc),float(np.sum(psc>0.05))]
    active=np.where(psc>1e-3)[0]
    if len(active)>2:
        rs=[np.corrcoef(window[:,active[i]],window[:,active[i+1]])[0,1] for i in range(len(active)-1)]
        feats.append(np.mean([r for r in rs if not np.isnan(r)]) if rs else 0.0)
    else: feats.append(0.0)
    return np.array(feats)

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

def session_timer():
    global _session_start
    while True:
        time.sleep(1)
        with lock:
            if state["running"] and _session_start>0:
                state["session_sec"]=int(time.time()-_session_start)
                pred=state["prediction"]
                state["durations"][pred]=state["durations"].get(pred,0)+1

def esp32_thread(port,clf,scaler,top_sc):
    global _session_start
    try: import serial
    except: sys.exit(1)
    try:
        ser=serial.Serial(port,BAUD_RATE,timeout=1); time.sleep(2)
    except Exception as e:
        print(f"  Cannot open: {e}"); sys.exit(1)
    ser.write(b"START\n"); time.sleep(0.3)
    with lock: state["connected"]=True
    print("  ESP32 connected ✓  — waiting for START button")
    rolling=deque(maxlen=WINDOW_SIZE); recent=deque(maxlen=VOTE_WINDOW)
    frames_since=0; fps_buf=deque(maxlen=43); fps_last=time.time()
    prev_running=False
    while True:
        with lock: running=state["running"]
        if not running:
            try: ser.reset_input_buffer()
            except: pass
            time.sleep(0.05); prev_running=False; continue
        if not prev_running:
            try: ser.reset_input_buffer()
            except: pass
            rolling.clear(); recent.clear()
            frames_since=0; fps_buf.clear(); fps_last=time.time()
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
            state["frame_count"]+=1; state["fps"]=round(fps,1); state["signal"]=row.tolist()
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
                    "proba":prob_d,"voted":is_voted,
                    "pred_count":state["pred_count"]+1})
                h=state["history"]; h.append(voted); state["history"]=h[-30:]
                ch=state["conf_history"]; ch.append(conf); state["conf_history"]=ch[-30:]

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Realtime Predictions</title>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Fira+Code:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root{
  --walk:#00e676;--nomv:#2979ff;
  --bg:#02050a;--p1:#060d16;--p2:#0a1628;
  --cyan:#00e5ff;--gold:#ffd600;
  --bd:#0d2035;--txt:#7ab3cc;--dim:#112030;
}
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;overflow:hidden}
body{background:var(--bg);color:var(--txt);font-family:'Fira Code',monospace;font-size:13px}
body::before{
  content:'';position:fixed;inset:0;z-index:0;opacity:.4;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='56' height='100'%3E%3Cpath d='M28 66L0 50V18L28 2l28 16v32zm0 0l28 16v18M28 66v18M0 68l28 16' stroke='%230d2035' stroke-width='1' fill='none'/%3E%3C/svg%3E");
}
body::after{content:'';position:fixed;inset:0;z-index:1;pointer-events:none;
  background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,.06) 3px,rgba(0,0,0,.06) 4px)}
.app{position:relative;z-index:2;height:100vh;display:flex;flex-direction:column;padding:10px;gap:8px}
header{display:flex;align-items:center;justify-content:space-between;
  background:var(--p1);border:1px solid var(--bd);border-radius:4px;padding:10px 20px;flex-shrink:0}
.logo{font-family:'Rajdhani',sans-serif;font-weight:700;font-size:1.3rem;letter-spacing:6px;color:#fff}
.logo span{color:var(--cyan);text-shadow:0 0 20px var(--cyan)}
.header-mid{text-align:center}
.header-title{font-family:'Rajdhani',sans-serif;font-size:1.05rem;font-weight:600;letter-spacing:4px;color:var(--cyan);opacity:.7}
.session-time{font-family:'Rajdhani',sans-serif;font-size:.75rem;font-weight:600;letter-spacing:2px;color:var(--gold)}
.header-right{display:flex;align-items:center;gap:10px}
.stat-pill{background:var(--dim);border:1px solid var(--bd);border-radius:3px;padding:4px 10px;font-size:.65rem;letter-spacing:2px;color:#3a6080}
.stat-pill span{color:var(--cyan)}
.btn-stop{padding:8px 28px;border-radius:3px;border:2px solid #ff1744;
  background:transparent;color:#ff1744;cursor:pointer;
  font-family:'Rajdhani',sans-serif;font-weight:700;font-size:.95rem;letter-spacing:4px}
.btn-stop:hover{background:rgba(255,23,68,.1)}
.main{flex:1;display:grid;grid-template-columns:200px 1fr 200px;gap:8px;min-height:0}
.panel{background:var(--p1);border:1px solid var(--bd);border-radius:4px;padding:14px;
  position:relative;overflow:hidden;display:flex;flex-direction:column;gap:10px}
.panel::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--cyan),transparent);opacity:.2}
.plabel{font-size:.55rem;letter-spacing:3px;color:#1a3a5a;text-transform:uppercase;margin-bottom:2px}
.metric-box{background:var(--dim);border:1px solid var(--bd);border-radius:3px;padding:8px 10px}
.metric-val{font-family:'Rajdhani',sans-serif;font-size:1.6rem;font-weight:700;color:var(--cyan)}
.metric-sub{font-size:.58rem;letter-spacing:2px;color:#1a3a5a;margin-top:1px}
.divider{height:1px;background:linear-gradient(90deg,transparent,var(--bd),transparent)}
.conn-row{display:flex;align-items:center;gap:8px}
.dot{width:8px;height:8px;border-radius:50%;background:#333;flex-shrink:0}
.dot.on{background:var(--walk);box-shadow:0 0 10px var(--walk)}
#sparkCanvas{width:100%;height:52px;display:block}
.dur-row{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.dur-bar-bg{flex:1;height:6px;background:var(--dim);border-radius:3px;overflow:hidden}
.dur-bar{height:100%;border-radius:3px;transition:width .5s}
.dur-bar.walking{background:var(--walk);box-shadow:0 0 6px var(--walk)}
.dur-bar.no_movement{background:var(--nomv);box-shadow:0 0 6px var(--nomv)}
.dur-sec{font-size:.6rem;color:#2a4a6a;min-width:28px;text-align:right}
.hist-wrap{display:flex;gap:4px;flex-wrap:wrap}
.hdot{width:13px;height:13px;border-radius:2px;background:var(--dim);transition:all .3s}
.hdot.walking{background:var(--walk);box-shadow:0 0 5px var(--walk)}
.hdot.no_movement{background:var(--nomv);box-shadow:0 0 5px var(--nomv)}
/* CENTER */
.center-panel{align-items:center;justify-content:center}
.act-name{font-family:'Rajdhani',sans-serif;font-weight:700;font-size:2.4rem;letter-spacing:8px;text-transform:uppercase;transition:all .3s;text-align:center}
.act-name.walking{color:var(--walk);text-shadow:0 0 30px var(--walk),0 0 60px rgba(0,230,118,.3)}
.act-name.no_movement{color:var(--nomv);text-shadow:0 0 30px var(--nomv),0 0 60px rgba(41,121,255,.3)}
.pred-meta{font-size:.65rem;letter-spacing:2px;color:#2a4a6a;margin-top:4px;text-align:center}
.pred-meta span{color:var(--cyan)}
.ring-wrap{position:relative;width:240px;height:240px;flex-shrink:0}
.ring-svg{width:100%;height:100%;transform:rotate(-90deg)}
.ring-bg{fill:none;stroke:var(--dim);stroke-width:6}
.ring-fill{fill:none;stroke-width:6;stroke-linecap:round;transition:stroke-dashoffset .5s,stroke .4s}
.ring-pct{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-family:'Rajdhani',sans-serif;font-weight:700;font-size:2rem;color:#fff;text-align:center;pointer-events:none}
.ring-pct-label{font-size:.6rem;letter-spacing:2px;color:#2a4a6a;display:block;margin-top:-4px}
.stickman-container{position:absolute;inset:30px;display:flex;align-items:center;justify-content:center}
.stick-svg{width:120px;height:150px;filter:drop-shadow(0 0 12px currentColor);transition:color .4s}
.stick-svg.walking{color:var(--walk)}
.stick-svg.no_movement{color:var(--nomv)}
.stick-svg circle,.stick-svg line{stroke:currentColor;fill:none;stroke-linecap:round}
.stick-svg circle.head{fill:currentColor;opacity:.15;stroke-width:2.5}
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
@keyframes breathe{0%,100%{transform:scaleY(1)}50%{transform:scaleY(1.03)}}
.no_movement .wbody{animation:breathe 3.5s ease-in-out infinite;transform-origin:60px 100px}
.tag{display:inline-block;font-size:.55rem;letter-spacing:2px;padding:2px 8px;border-radius:2px;border:1px solid var(--bd);color:#1a3a5a;margin-top:4px}
.tag.voted{border-color:var(--walk);color:var(--walk)}
.tag.model{border-color:var(--cyan);color:var(--cyan)}
#sigCanvas{width:100%;height:58px;display:block}
/* RIGHT */
.prob-item{margin-bottom:14px}
.prob-header{display:flex;justify-content:space-between;margin-bottom:5px}
.prob-name{font-size:.62rem;letter-spacing:2px}
.prob-name.walking{color:var(--walk)}
.prob-name.no_movement{color:var(--nomv)}
.prob-pct{font-family:'Rajdhani',sans-serif;font-size:1rem;font-weight:700;color:#fff}
.prob-bar-bg{height:8px;background:var(--dim);border-radius:4px;overflow:hidden}
.prob-bar-fill{height:100%;border-radius:4px;transition:width .4s}
.prob-bar-fill.walking{background:linear-gradient(90deg,#006035,var(--walk));box-shadow:0 0 10px var(--walk)}
.prob-bar-fill.no_movement{background:linear-gradient(90deg,#002a80,var(--nomv));box-shadow:0 0 10px var(--nomv)}
/* IDLE */
#idleScreen{position:fixed;inset:0;z-index:50;background:rgba(2,5,10,.92);
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:24px;backdrop-filter:blur(4px)}
#idleScreen.hidden{display:none}
.idle-title{font-family:'Rajdhani',sans-serif;font-size:2.2rem;font-weight:700;letter-spacing:8px;color:#fff;text-align:center}
.idle-title span{color:var(--cyan);text-shadow:0 0 30px var(--cyan)}
.idle-sub{font-size:.75rem;letter-spacing:3px;color:#2a4a6a}
.idle-conn{font-size:.7rem;letter-spacing:2px;color:#1a3a5a}
.idle-conn.ok{color:var(--walk)}
.btn-big{padding:14px 48px;border-radius:4px;border:2px solid var(--walk);
  background:transparent;color:var(--walk);cursor:pointer;
  font-family:'Rajdhani',sans-serif;font-weight:700;font-size:1.2rem;letter-spacing:6px;
  transition:all .2s;box-shadow:0 0 30px rgba(0,230,118,.2)}
.btn-big:hover{background:rgba(0,230,118,.1);box-shadow:0 0 50px rgba(0,230,118,.4)}
.btn-big:disabled{opacity:.4;cursor:not-allowed}
.spin{width:36px;height:36px;border-radius:50%;border:2px solid var(--dim);border-top-color:var(--cyan);animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>

<div id="idleScreen">
  <div class="idle-title">REALTIME<br><span>PREDICTIONS</span></div>
  <div class="idle-sub">WiFi CSI · 52 Subcarriers · Binary Detection</div>
  <div class="spin" id="idleSpin"></div>
  <div class="idle-conn" id="idleConn">Connecting to ESP32...</div>
  <button class="btn-big" id="btnBig" onclick="startSession()" disabled>START</button>
  <div class="idle-sub" style="color:#1a3a5a">Walking · No Movement</div>
</div>

<div class="app">
  <header>
    <div class="logo">CSI<span>.</span>BINARY</div>
    <div class="header-mid">
      <div class="header-title">REALTIME PREDICTIONS</div>
      <div class="session-time" id="sessionTime">00:00:00</div>
    </div>
    <div class="header-right">
      <div class="stat-pill">FPS <span id="fpsVal">0</span></div>
      <div class="stat-pill">FRAMES <span id="frameVal">0</span></div>
      <div class="stat-pill">PREDS <span id="predVal">0</span></div>
      <button class="btn-stop" onclick="stopSession()">STOP</button>
    </div>
  </header>

  <div class="main">
    <!-- LEFT -->
    <div class="panel">
      <div class="plabel">Status</div>
      <div class="conn-row">
        <div class="dot on"></div>
        <span style="font-size:.65rem;letter-spacing:1px">ESP32 CONNECTED</span>
      </div>
      <div class="divider"></div>
      <div class="plabel">Confidence Trend</div>
      <canvas id="sparkCanvas"></canvas>
      <div class="divider"></div>
      <div class="plabel">Activity Duration</div>
      <div>
        <div class="dur-row">
          <span style="font-size:.85rem">🚶</span>
          <div class="dur-bar-bg"><div class="dur-bar walking" id="durWalk" style="width:0%"></div></div>
          <div class="dur-sec" id="durWalkSec">0s</div>
        </div>
        <div class="dur-row">
          <span style="font-size:.85rem">🧍</span>
          <div class="dur-bar-bg"><div class="dur-bar no_movement" id="durNomv" style="width:0%"></div></div>
          <div class="dur-sec" id="durNomvSec">0s</div>
        </div>
      </div>
      <div class="divider"></div>
      <div class="plabel">History</div>
      <div class="hist-wrap" id="histWrap"></div>
    </div>

    <!-- CENTER -->
    <div class="panel center-panel">
      <div class="act-name no_movement" id="actName">NO MOVEMENT</div>
      <div class="pred-meta">CONFIDENCE <span id="confNum">0.0</span>% &nbsp;
        <span class="tag" id="srcTag">IDLE</span></div>
      <div class="ring-wrap">
        <svg class="ring-svg" viewBox="0 0 200 200">
          <circle class="ring-bg" cx="100" cy="100" r="90"/>
          <circle class="ring-fill" id="ringFill" cx="100" cy="100" r="90"
            stroke="var(--nomv)" stroke-dasharray="565.5" stroke-dashoffset="565.5"/>
        </svg>
        <div class="ring-pct"><span id="ringPct">0</span>%<span class="ring-pct-label">CONF</span></div>
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
      <canvas id="sigCanvas"></canvas>
    </div>

    <!-- RIGHT -->
    <div class="panel">
      <div class="plabel">Class Probabilities</div>
      <div class="prob-item">
        <div class="prob-header">
          <span class="prob-name walking">WALKING</span>
          <span class="prob-pct" id="pWalk">0%</span>
        </div>
        <div class="prob-bar-bg"><div class="prob-bar-fill walking" id="bWalk" style="width:0%"></div></div>
      </div>
      <div class="prob-item">
        <div class="prob-header">
          <span class="prob-name no_movement">NO MOVEMENT</span>
          <span class="prob-pct" id="pNomv">0%</span>
        </div>
        <div class="prob-bar-bg"><div class="prob-bar-fill no_movement" id="bNomv" style="width:0%"></div></div>
      </div>
      <div class="divider"></div>
      <div class="plabel">Predictions</div>
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
const STICK={
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
  </g>`
};
const COLVARS={walking:'var(--walk)',no_movement:'var(--nomv)'};
const LABELS={walking:'WALKING',no_movement:'NO MOVEMENT'};
const CIRC=565.5;
let cur='no_movement'; const sigBuf=[]; const SIG_MAX=100;

async function startSession(){
  await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'start'})});
  document.getElementById('idleScreen').classList.add('hidden');
}
async function stopSession(){
  await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'stop'})});
  document.getElementById('idleScreen').classList.remove('hidden');
  document.getElementById('idleConn').textContent='Session stopped. Click START to resume.';
  document.getElementById('idleConn').className='idle-conn ok';
  document.getElementById('btnBig').disabled=false;
  document.getElementById('idleSpin').style.display='none';
}
function fmtTime(s){const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),ss=s%60;return`${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(ss).padStart(2,'0')}`}

function update(d){
  if(d.connected){
    document.getElementById('idleConn').textContent='ESP32 Connected ✓  Ready!';
    document.getElementById('idleConn').className='idle-conn ok';
    document.getElementById('btnBig').disabled=false;
    document.getElementById('idleSpin').style.display='none';
  }
  if(!d.running) return;
  const act=d.prediction;
  if(act!==cur){
    cur=act;
    const svg=document.getElementById('stickSvg');
    svg.innerHTML=STICK[act]; svg.className='stick-svg '+act;
    const lbl=document.getElementById('actName');
    lbl.textContent=LABELS[act]; lbl.className='act-name '+act;
  }
  const conf=d.confidence;
  const rf=document.getElementById('ringFill');
  rf.style.strokeDashoffset=CIRC-(conf/100)*CIRC;
  rf.style.stroke=COLVARS[act];
  document.getElementById('ringPct').textContent=Math.round(conf);
  document.getElementById('confNum').textContent=conf.toFixed(1);
  const tag=document.getElementById('srcTag');
  tag.textContent=d.voted?'VOTED':'LIVE';
  tag.className='tag '+(d.voted?'voted':'model');
  const pr=d.proba;
  const pw=(pr['walking']||0).toFixed(0),pn=(pr['no_movement']||0).toFixed(0);
  document.getElementById('bWalk').style.width=pw+'%'; document.getElementById('pWalk').textContent=pw+'%';
  document.getElementById('bNomv').style.width=pn+'%'; document.getElementById('pNomv').textContent=pn+'%';
  document.getElementById('fpsVal').textContent=d.fps||0;
  document.getElementById('frameVal').textContent=(d.frame_count||0).toLocaleString();
  document.getElementById('predVal').textContent=d.pred_count||0;
  document.getElementById('bigPred').textContent=d.pred_count||0;
  document.getElementById('bigFrames').textContent=(d.frame_count||0).toLocaleString();
  document.getElementById('sessionTime').textContent=fmtTime(d.session_sec||0);
  const durs=d.durations||{}; const total=Object.values(durs).reduce((a,b)=>a+b,1);
  document.getElementById('durWalk').style.width=((durs['walking']||0)/total*100)+'%';
  document.getElementById('durWalkSec').textContent=(durs['walking']||0)+'s';
  document.getElementById('durNomv').style.width=((durs['no_movement']||0)/total*100)+'%';
  document.getElementById('durNomvSec').textContent=(durs['no_movement']||0)+'s';
  const hw=document.getElementById('histWrap'); hw.innerHTML='';
  const hist=[...Array(Math.max(0,20-(d.history||[]).length)).fill(''),...(d.history||[]).slice(-20)];
  hist.forEach(h=>{const dv=document.createElement('div');dv.className='hdot'+(h?' '+h:'');hw.appendChild(dv)});
  if(d.conf_history&&d.conf_history.length>1) drawSpark(d.conf_history);
  if(d.signal&&d.signal.length===52){
    const avg=d.signal.reduce((a,b)=>a+b,0)/d.signal.length;
    sigBuf.push(avg); if(sigBuf.length>SIG_MAX) sigBuf.shift(); drawSig(act);
  }
}
function drawSpark(hist){
  const c=document.getElementById('sparkCanvas'),W=c.offsetWidth,H=c.offsetHeight||52;
  c.width=W;c.height=H;const ctx=c.getContext('2d');ctx.clearRect(0,0,W,H);
  if(hist.length<2)return;
  ctx.strokeStyle='var(--cyan)';ctx.lineWidth=1.5;ctx.shadowBlur=8;ctx.shadowColor='var(--cyan)';
  ctx.beginPath();
  hist.slice(-30).forEach((v,i)=>{const x=(i/29)*W,y=H-(v/100)*(H*.85)-H*.05;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y)});
  ctx.stroke();ctx.shadowBlur=0;
  ctx.lineTo(W,H);ctx.lineTo(0,H);ctx.closePath();ctx.fillStyle='rgba(0,229,255,.06)';ctx.fill();
}
function drawSig(act){
  const c=document.getElementById('sigCanvas'),W=c.offsetWidth,H=c.offsetHeight||58;
  c.width=W;c.height=H;const ctx=c.getContext('2d');ctx.clearRect(0,0,W,H);
  ctx.strokeStyle='rgba(13,32,53,.9)';ctx.lineWidth=1;
  for(let x=0;x<W;x+=50){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke()}
  for(let y=0;y<H;y+=20){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke()}
  if(sigBuf.length<2)return;
  const col=act==='walking'?'#00e676':'#2979ff';
  const mn=Math.min(...sigBuf),mx=Math.max(...sigBuf);
  ctx.shadowBlur=12;ctx.shadowColor=col;ctx.strokeStyle=col;ctx.lineWidth=2;ctx.beginPath();
  sigBuf.forEach((v,i)=>{const x=(i/(SIG_MAX-1))*W,y=H-((v-mn)/(mx-mn+1e-6))*(H*.8)-H*.08;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y)});
  ctx.stroke();ctx.shadowBlur=0;
  ctx.lineTo((sigBuf.length-1)/(SIG_MAX-1)*W,H);ctx.lineTo(0,H);ctx.closePath();ctx.fillStyle=col+'18';ctx.fill();
}
async function poll(){
  try{const r=await fetch('/api/state');if(r.ok)update(await r.json());}catch(e){}
  setTimeout(poll,280);
}
poll();
</script>
</body></html>"""

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/","/index.html"):
            self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8")
            self.end_headers(); self.wfile.write(HTML.encode())
        elif self.path=="/api/state":
            self.send_response(200); self.send_header("Content-Type","application/json")
            self.send_header("Access-Control-Allow-Origin","*"); self.end_headers()
            with lock: data=dict(state)
            self.wfile.write(json.dumps(data).encode())
        else: self.send_response(404); self.end_headers()
    def do_POST(self):
        global _session_start
        if self.path=="/api/control":
            length=int(self.headers.get("Content-Length",0))
            body=json.loads(self.rfile.read(length))
            with lock:
                if body.get("action")=="start":
                    state["running"]=True
                    if _session_start==0: _session_start=time.time()
                else: state["running"]=False
            self.send_response(200); self.send_header("Content-Type","application/json")
            self.end_headers(); self.wfile.write(b'{"ok":true}')
        else: self.send_response(404); self.end_headers()
    def log_message(self,*a): pass

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--port",type=str,default=None)
    args=parser.parse_args()
    print("\n"+"="*52)
    print("  WiFi CSI — Binary Dashboard (Walk vs No-Move)")
    print("="*52)
    print("\n  Loading model...")
    for p in [MODEL_PATH,SCALER_PATH,TOP_SC_PATH]:
        if not os.path.exists(p):
            print(f"  Not found: {p}\n  Run binary_classifier.py --mode train first!"); sys.exit(1)
    clf=joblib.load(MODEL_PATH); scaler=joblib.load(SCALER_PATH); top_sc=np.load(TOP_SC_PATH)
    print(f"  Model loaded ✓  Classes: {list(clf.classes_)}")
    port=args.port or find_port()
    if not port:
        print("  No port found. Run: python realtime_ui_binary.py --port COM3"); sys.exit(1)
    threading.Thread(target=session_timer,daemon=True).start()
    threading.Thread(target=esp32_thread,args=(port,clf,scaler,top_sc),daemon=True).start()
    server=HTTPServer(("0.0.0.0",UI_PORT),Handler)
    print(f"\n  Dashboard → http://localhost:{UI_PORT}")
    print(f"  Click START in browser to begin\n")
    try:
        import webbrowser; time.sleep(2); webbrowser.open(f"http://localhost:{UI_PORT}")
    except: pass
    try: server.serve_forever()
    except KeyboardInterrupt: print("\n  Stopped.")