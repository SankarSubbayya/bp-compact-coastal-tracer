"""Mixed-tide (semidiurnal + diurnal) San Francisco Bay currents on the 200 m CRM
grid, vectorized semi-implicit Casulli (1990) solver. Forced at the Pacific open
boundary by the six dominant constituents (M2,S2,N2,K1,O1,P1) with amplitudes
calibrated, band-by-band, to the NOAA Golden Gate harmonics (station 9414290);
phases are the NOAA Greenwich epochs. San Francisco Bay is a mixed,
predominantly-semidiurnal tide (form factor (K1+O1)/(M2+S2) ~ 0.84)."""
import sys, time, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import splu

G=9.81; DT=120.0; THETA=0.6; HMIN=3.0   # physical estuarine minimum depth (2-3 m)
MANNING_N=0.037   # depth-dependent bottom friction Cd = g n^2 / H^(1/3): Cd~0.003 at the deep Gate
                  # (matches the 2003 calibration), ~0.009 over the shallow flats -> damps the South
                  # Bay resonance so no cell dries at HMIN=3 m
DEG=np.pi/180.0
D="sfbay_fine"
d=np.load(f"{D}/sfbay_fine.npz"); depth,wet,lat,lon=d["depth"],d["wet"],d["lat"],d["lon"]
dx=float(d["dx"]); dy=float(d["dy"]); ny,nx=depth.shape
H=np.where(wet,np.maximum(depth,HMIN),0.0)
cid=-np.ones((ny,nx),int); wj,wi=np.where(wet); N=wj.size; cid[wj,wi]=np.arange(N)
jg,ig=232,96; kg=cid[jg,ig]                      # Golden Gate cell
def deepest(laq,loq,r=0.022):                    # deepest wet cell within ~2 km (the channel)
    idx=np.where((np.abs(lat[wj]-laq)<r)&(np.abs(lon[wi]-loq)<r))[0]
    k=idx[np.argmax(depth[wj[idx],wi[idx]])]; return cid[wj[k],wi[k]]
STATIONS={"Golden Gate":kg, "Oakland":deepest(37.81,-122.34),
          "Richmond":deepest(37.93,-122.42), "San Pablo Bay":deepest(38.05,-122.42)}

# faces + Helmholtz operator (constant in time -> factor once)
um=wet[:,:-1]&wet[:,1:]; vm=wet[:-1,:]&wet[1:,:]
uk1=cid[:,:-1][um]; uk2=cid[:,1:][um]; Hu=(0.5*(H[:,:-1]+H[:,1:]))[um]
vk1=cid[:-1,:][vm]; vk2=cid[1:,:][vm]; Hv=(0.5*(H[:-1,:]+H[1:,:]))[vm]
CDu=G*MANNING_N**2/Hu**(1.0/3.0); CDv=G*MANNING_N**2/Hv**(1.0/3.0)   # Manning drag per face
# Rayleigh sponge over the under-resolved South Bay (south of the Central-Bay study
# region): absorbs its spurious resonance/closed-boundary reflection so the tidal
# range there stays realistic. The Cosco Busan release (37.806 N) and the transport
# domain lie NORTH of LAT_SP, so the demonstration is unaffected.
LAT_SP=37.76; latmin=float(lat.min()); RMAX=6.0e-3
def sponge(cells):
    s=np.clip((LAT_SP-lat[wj[cells]])/(LAT_SP-latmin),0.0,1.0); return RMAX*s**2
ru=sponge(uk1); rv=sponge(vk1)
openm=np.zeros((ny,nx),bool); openm[:,0]=wet[:,0]; isopen=np.zeros(N,bool); isopen[cid[openm&wet]]=True
A=dx*dy; coef=THETA**2*DT**2*G/A; wu=coef*Hu*dy/dx; wv=coef*Hv*dx/dy
rows=np.concatenate([uk1,uk2,vk1,vk2,np.arange(N)]); cols=np.concatenate([uk2,uk1,vk2,vk1,np.arange(N)])
diag=np.ones(N)
for a,b,w in [(uk1,uk2,wu),(vk1,vk2,wv)]: np.add.at(diag,a,w); np.add.at(diag,b,w)
vals=np.concatenate([-wu,-wu,-wv,-wv,diag])
Mx=sp.csr_matrix((vals,(rows,cols)),shape=(N,N)).tolil()
for k in np.where(isopen)[0]: Mx.rows[k]=[k]; Mx.data[k]=[1.0]
lu=splu(Mx.tocsc())

# NOAA 9414290 constituents: name -> (speed deg/h, observed Gate amp m, Greenwich phase deg)
NOAA={"M2":(28.98410,0.576,208.2),"S2":(30.00000,0.137,216.2),"N2":(28.43973,0.122,183.2),
      "K1":(15.04107,0.370,225.4),"O1":(13.94304,0.230,208.4),"P1":(14.95893,0.114,222.1)}
def omega(name): return NOAA[name][0]*DEG/3600.0   # rad/s

def run(forcing, ndays, rec_cells=(), field_window=None):
    """forcing: list of (omega, amp_offshore, phase_rad). Returns dict with time
    series at rec_cells and (optionally) full U,V fields over field_window=(t0,t1,dt_s)."""
    nst=int(ndays*86400/DT); eta=np.zeros(N); uvel=np.zeros(uk1.size); vvel=np.zeros(vk1.size)
    ts=[]; series={k:[] for k in rec_cells}; suv={k:[] for k in rec_cells}
    sU={k:[] for k in rec_cells}; sV={k:[] for k in rec_cells}
    uf_of={k:np.where((uk1==k)|(uk2==k))[0] for k in rec_cells}   # incident faces (cell-centre current)
    vf_of={k:np.where((vk1==k)|(vk2==k))[0] for k in rec_cells}
    Uf=[]; Vf=[]; tf=[]; etamax=0.0; out_argmax=[0]
    nextf=field_window[0] if field_window else None
    for n in range(nst+1):
        t=n*DT
        gEx=(eta[uk2]-eta[uk1])/dx; gNy=(eta[vk2]-eta[vk1])/dy
        pu=uvel-DT*G*(1-THETA)*gEx; pv=vvel-DT*G*(1-THETA)*gNy
        fxu=Hu*dy*(THETA*pu+(1-THETA)*uvel); fxv=Hv*dx*(THETA*pv+(1-THETA)*vvel)
        div=np.zeros(N); np.add.at(div,uk1,fxu); np.add.at(div,uk2,-fxu); np.add.at(div,vk1,fxv); np.add.at(div,vk2,-fxv)
        rhs=eta-(DT/A)*div
        rhs[isopen]=sum(a*np.cos(w*t-p) for (w,a,p) in forcing)
        eta=lu.solve(rhs)
        em=float(np.abs(eta).max())
        if em>etamax: etamax=em; out_argmax[0]=int(np.argmax(np.abs(eta)))
        us=pu-DT*G*THETA*(eta[uk2]-eta[uk1])/dx; vs=pv-DT*G*THETA*(eta[vk2]-eta[vk1])/dy
        uvel=us/(1+DT*CDu*np.abs(us)/Hu+DT*ru); vvel=vs/(1+DT*CDv*np.abs(vs)/Hv+DT*rv)
        if rec_cells:
            ts.append(t)
            for k in rec_cells:
                series[k].append(eta[k])
                uc=uvel[uf_of[k]].mean() if uf_of[k].size else 0.0
                vc=vvel[vf_of[k]].mean() if vf_of[k].size else 0.0
                suv[k].append(np.hypot(uc,vc)); sU[k].append(uc); sV[k].append(vc)
        if field_window and t>=nextf and t<=field_window[1]+1e-6:
            us_=np.zeros(N); uc_=np.zeros(N); vs_=np.zeros(N); vc_=np.zeros(N)
            np.add.at(us_,uk1,uvel); np.add.at(uc_,uk1,1.0); np.add.at(us_,uk2,uvel); np.add.at(uc_,uk2,1.0)
            np.add.at(vs_,vk1,vvel); np.add.at(vc_,vk1,1.0); np.add.at(vs_,vk2,vvel); np.add.at(vc_,vk2,1.0)
            Uf.append(us_/np.maximum(uc_,1)); Vf.append(vs_/np.maximum(vc_,1)); tf.append(t); nextf+=field_window[2]
        if not np.all(np.isfinite(eta)): print("BLEW UP",n); sys.exit(1)
    out={"t":np.array(ts),"etamax":etamax,"argmax":out_argmax[0]}
    for k in rec_cells:
        out[f"eta_{k}"]=np.array(series[k]); out[f"spd_{k}"]=np.array(suv[k])
        out[f"u_{k}"]=np.array(sU[k]); out[f"v_{k}"]=np.array(sV[k])
    if field_window: out["U"]=np.array(Uf); out["V"]=np.array(Vf); out["tf"]=np.array(tf)
    return out

def fit_amp(t,sig,w):
    M=np.column_stack([np.cos(w*t),np.sin(w*t),np.ones_like(t)]); c,*_=np.linalg.lstsq(M,sig,rcond=None)
    return np.hypot(c[0],c[1])

t0=time.time()
# ---- band gains from single-constituent calibration runs (last 2 days fitted) ----
rM=run([(omega("M2"),0.50,0)],4,rec_cells=(kg,)); m=rM["t"]>2*86400
gain_sd=fit_amp(rM["t"][m],rM["eta_%d"%kg][m],omega("M2"))/0.50
rK=run([(omega("K1"),0.30,0)],4,rec_cells=(kg,)); m=rK["t"]>2*86400
gain_di=fit_amp(rK["t"][m],rK["eta_%d"%kg][m],omega("K1"))/0.30
print(f"band gains: semidiurnal={gain_sd:.3f}  diurnal={gain_di:.3f}")

BAND={"M2":"sd","S2":"sd","N2":"sd","K1":"di","O1":"di","P1":"di"}
SD_CORR=1.10   # restore semidiurnal lost to nonlinear (quadratic-drag) coupling with the diurnal currents
DI_CORR=1.00
forcing=[(omega(nm),amp/(gain_sd if BAND[nm]=="sd" else gain_di)*(SD_CORR if BAND[nm]=="sd" else DI_CORR),ph*DEG)
         for nm,(sp_,amp,ph) in NOAA.items()]

# ---- production run: 16 d GG/Oakland series for validation + 26 h field window for figures ----
SPIN=3*86400; WIN=(SPIN, SPIN+26*3600, 1800.0)   # store fields 26 h after spin-up, every 30 min
res=run(forcing, 19, rec_cells=tuple(STATIONS.values()), field_window=WIN)
tg=res["t"]; eg=res["eta_%d"%kg]; mfit=tg>5*86400
amps={nm:fit_amp(tg[mfit],eg[mfit],omega(nm)) for nm in ["M2","S2","N2","K1","O1"]}
print("Gate elevation amplitudes (model vs NOAA):")
for nm in ["M2","S2","N2","K1","O1"]:
    print(f"  {nm}: {amps[nm]:.3f} m  (obs {NOAA[nm][1]:.3f})")
F=(amps["K1"]+amps["O1"])/(amps["M2"]+amps["S2"]); print(f"form factor F = {F:.2f} (mixed if 0.25<F<1.5)")
def fit_M2_current(tt,u,v):   # M2 ellipse semi-major (rectilinear approx) with S2/K1/O1 separated
    cols=[];
    for nm in ["M2","S2","K1","O1"]:
        w=omega(nm); cols+=[np.cos(w*tt),np.sin(w*tt)]
    Mf=np.column_stack(cols+[np.ones_like(tt)])
    cu,*_=np.linalg.lstsq(Mf,u,rcond=None); cv,*_=np.linalg.lstsq(Mf,v,rcond=None)
    return np.hypot(np.hypot(cu[0],cu[1]),np.hypot(cv[0],cv[1]))
print("Station M2 principal current amplitude (model):")
cur_amp={}
for nm,k in STATIONS.items():
    cur_amp[nm]=fit_M2_current(tg[mfit],res["u_%d"%k][mfit],res["v_%d"%k][mfit])
    print(f"  {nm}: {cur_amp[nm]:.2f} m/s")
# Golden Gate current speed series (for diurnal-inequality figure) over the field window
U=res["U"]; V=res["V"]; tf=res["tf"]; spdgg=np.hypot(U[:,kg],V[:,kg])
print(f"field window {len(tf)} snapshots; Gate peak speed {spdgg.max():.2f} m/s; runtime {time.time()-t0:.0f}s")
am=res["argmax"]; jam,iam=wj[am],wi[am]
nnb=int(wet[jam,iam-1])+int(wet[jam,iam+1])+int(wet[jam-1,iam])+int(wet[jam+1,iam]) if 0<jam<ny-1 and 0<iam<nx-1 else 0
print(f"WET/DRY CHECK: max |surface depression| = {res['etamax']:.2f} m ; "
      f"min working column = HMIN - max|eta| = {HMIN-res['etamax']:.2f} m (>0 => no cell dries, stable)")
print(f"  max|eta| cell: lat {lat[jam]:.3f} lon {lon[iam]:.3f}  depth {depth[jam,iam]:.1f} m  wet-neighbours {nnb}/4")

np.savez(f"{D}/currents_fine.npz", U=U, V=V, t=tf, wj=wj, wi=wi, cid=cid, wet=wet,
         lat=lat, lon=lon, dx=dx, dy=dy)
np.savez(f"{D}/sfbay_tides.npz", t=tg, eta_gg=eg, kg=kg,
         amp_names=list(amps.keys()), amp_model=[amps[k] for k in amps],
         amp_obs=[NOAA[k][1] for k in amps])
# ---- station current/elevation time series (for the multi-location figure) ----
st_names=list(STATIONS.keys()); st_k=[STATIONS[n] for n in st_names]
np.savez(f"{D}/sfbay_stations.npz", t=tg, names=st_names,
         eta=np.array([res["eta_%d"%k] for k in st_k]),
         spd=np.array([res["spd_%d"%k] for k in st_k]),
         lat=np.array([lat[wj[k]] for k in st_k]), lon=np.array([lon[wi[k]] for k in st_k]))
# ---- calibration record (constituent forcing & model response) ----
print("\nCALIBRATION TABLE (constituent | NOAA amp,phase | offshore forcing amp | model Gate amp):")
cal_rows=[]
for (nm,(spd_,amp,ph)),(w,a_off,p) in zip(NOAA.items(),forcing):
    mod=amps.get(nm,float('nan'))
    cal_rows.append((nm,amp,ph,a_off,mod))
    print(f"  {nm}: NOAA {amp:.3f} m @ {ph:.1f} deg | offshore {a_off:.3f} m | model {mod:.3f} m")
print(f"  params: Manning n={MANNING_N}, HMIN={HMIN} m, sponge RMAX={RMAX}/s south of {LAT_SP} N, "
      f"band gains sd={gain_sd:.3f} di={gain_di:.3f}, SD_CORR={SD_CORR}")
np.savez(f"{D}/sfbay_calibration.npz", names=[r[0] for r in cal_rows],
         noaa_amp=[r[1] for r in cal_rows], noaa_phase=[r[2] for r in cal_rows],
         offshore_amp=[r[3] for r in cal_rows], model_amp=[r[4] for r in cal_rows],
         manning_n=MANNING_N, hmin=HMIN, gain_sd=gain_sd, gain_di=gain_di)
print("saved currents_fine.npz, sfbay_tides.npz, sfbay_stations.npz, sfbay_calibration.npz")
