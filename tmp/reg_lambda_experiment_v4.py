#!/usr/bin/env python3
"""AX1 reg_lambda 参数网格实验 v4 —— 168 组 × 6 seeds.

修正点：实验基线参数直接复用 AX1 当前配置，避免 `objective` 等基础参数与生产配置漂移。
"""
from __future__ import annotations
import sys
import time, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from scipy import stats

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from skyeye.products.ax1.config import DEFAULT_LGBM_PARAMS

def make_etf_data(n_dates=400, n_assets=30, snr=0.15, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n_dates)
    oids = [f"ETF_{i:03d}" for i in range(n_assets)]
    sector = {o: rng.integers(0,5) for o in oids}
    rows, lrows = [], []
    for d in dates:
        sn = {s: float(rng.normal(0,0.01)) for s in range(5)}
        for o in oids:
            m = float(rng.normal(0,0.025)); v = float(abs(rng.normal(0.018,0.006)))
            dv = float(abs(rng.normal(5e7,1.5e7))); s = sector[o]
            se = float(rng.normal(0,0.005)); st = float(rng.normal(0,0.003))
            sig = 0.3*m+0.4*se+0.2*st; cn = sn[s]
            pn = rng.normal(0,abs(sig)/snr) if abs(sig)>1e-9 else rng.normal(0,0.01)
            r10 = sig+cn+pn; r5 = 0.5*sig+0.5*cn+rng.normal(0,abs(sig*0.5)/snr+0.003)
            r20 = 1.5*sig+0.8*cn+rng.normal(0,abs(sig*1.5)/snr+0.005)
            vl = v*abs(rng.normal(1.0,0.15))+0.008
            rows.append({"date":d,"order_book_id":o,"momentum_2d":m,"volatility_3d":v,
                         "feature_dollar_volume":dv,"sector_exposure":se,"style_exposure":st})
            lrows.append({"date":d,"order_book_id":o,"label_return_5d":r5,"label_return_10d":r10,
                          "label_return_20d":r20,"label_net_return_5d":r5-0.001,
                          "label_net_return_10d":r10-0.001,"label_net_return_20d":r20-0.001,
                          "label_volatility_10d":vl})
    labels = pd.DataFrame(lrows)
    for h in (5,10,20):
        c = f"label_net_return_{h}d"
        labels[f"label_relative_net_return_{h}d"] = labels[c]-labels.groupby("date")[c].transform("mean")
    return pd.DataFrame(rows), labels

def evaluate(features, labels, params, n_folds=5):
    fc = ["momentum_2d","volatility_3d","feature_dollar_volume","sector_exposure","style_exposure"]
    ad = sorted(features["date"].unique()); fs = len(ad)//(n_folds+1)
    oo,tt = [],[]
    for fold in range(n_folds):
        te,ve = fs*(fold+1), fs*(fold+2)
        if ve>len(ad): break
        t_end,v_end = ad[te-1], ad[min(ve-1,len(ad)-1)]
        tm = features["date"]<=t_end; vm = (features["date"]>t_end)&(features["date"]<=v_end)
        tX,tY = features.loc[tm,fc].fillna(0), labels.loc[tm]
        vX,vY = features.loc[vm,fc].fillna(0), labels.loc[vm]
        if len(tX)<50 or len(vX)<10: continue
        tc = "label_relative_net_return_10d"
        td = lgb.Dataset(tX,label=tY[tc].values)
        vd = lgb.Dataset(vX,label=vY[tc].values,reference=td)
        model = lgb.train(params,td,num_boost_round=params.get("n_estimators",200),
                          valid_sets=[vd],
                          callbacks=[lgb.early_stopping(params.get("early_stopping_rounds",20),verbose=False)])
        tp = model.predict(tX); tt.append(pd.Series(tp).corr(pd.Series(tY[tc].values),method="spearman"))
        vp = model.predict(vX); vdf = vY[["date","order_book_id",tc]].copy(); vdf["pred"]=vp
        dics = []
        for d,g in vdf.groupby("date"):
            if len(g)<5: continue
            ic = g["pred"].corr(g[tc],method="spearman")
            if np.isfinite(ic): dics.append(ic)
        oo.append(float(np.mean(dics)) if dics else 0.0)
    if not oo: return {}
    tic,oic = float(np.mean(tt)), float(np.mean(oo))
    return {"train_ic":tic,"oos_ic":oic,"ic_gap":tic-oic,
            "ic_gap_ratio":(tic-oic)/tic if abs(tic)>1e-9 else 0.0,
            "oos_ic_ir":np.mean(oo)/np.std(oo) if np.std(oo)>1e-9 else 0.0}

def main():
    t0 = time.time()
    L = [0.3,0.5,1.0,1.5,2.0,3.0,4.0]  # 7 λ
    A = [0.2,0.3,0.5,0.8,1.0,1.5]       # 6 α
    M = [30,50,65,80]                     # 4 mcs
    ns = 6
    total = len(L)*len(A)*len(M)*ns
    print(f"网格: {len(L)}λ × {len(A)}α × {len(M)}mcs = {len(L)*len(A)*len(M)} 组, {ns} seeds, 共 {total} 轮")

    bp = dict(DEFAULT_LGBM_PARAMS)
    print(
        "AX1 基线参数: "
        f"objective={bp['objective']}, metric={bp['metric']}, "
        f"reg_lambda={bp['reg_lambda']}, reg_alpha={bp['reg_alpha']}, "
        f"min_child_samples={bp['min_child_samples']}"
    )

    ar,dn = [],0
    for seed in range(ns):
        f,l = make_etf_data(n_dates=400,n_assets=30,snr=0.15,seed=seed*100+7)
        for lam in L:
            for alp in A:
                for mcs in M:
                    dn+=1
                    if dn%100==0:
                        el=time.time()-t0; r=dn/el; et=(total-dn)/r if r>0 else 0
                        print(f"  {dn}/{total} ({dn/total:.0%}) el={el:.0f}s eta={et:.0f}s")
                    p={**bp,"reg_lambda":lam,"reg_alpha":alp,"min_child_samples":mcs,
                       "seed":seed*100+7,"feature_fraction_seed":seed*100+7,
                       "bagging_seed":seed*100+7,"data_random_seed":seed*100+7,"drop_seed":seed*100+7}
                    r=evaluate(f,l,p,5)
                    if r: ar.append({"seed":seed,"lam":lam,"alp":alp,"mcs":mcs,**r})
    el=time.time()-t0; print(f"\n训练完成, {el:.0f}s, {len(ar)} 条")

    df = pd.DataFrame(ar)
    # 重命名方便分析
    df = df.rename(columns={"lam":"reg_lambda","alp":"reg_alpha"})

    s = df.groupby(["reg_lambda","reg_alpha","mcs"]).agg(
        oos_ic=("oos_ic","mean"), ic_gap=("ic_gap","mean"),
        ic_gap_ratio=("ic_gap_ratio","mean"), oos_ir=("oos_ic_ir","mean"),
        train_ic=("train_ic","mean")).reset_index()

    s["ic_rk"]=s["oos_ic"].rank(ascending=False)
    s["ir_rk"]=s["oos_ir"].rank(ascending=False)
    s["gap_rk"]=s["ic_gap"].rank(ascending=True)
    s["cmp"]=(s["ic_rk"]+s["ir_rk"]+s["gap_rk"])/3

    print(f"\n{'='*115}")
    print("Pareto Top 25 (OOS_IC + OOS_IR + IC_gap 综合排名)")
    print(f"{'='*115}")
    print(f"{'λ':>5} {'α':>5} {'mcs':>5} {'TrainIC':>9} {'OOS_IC':>9} {'IC_gap':>9} {'Gap%':>8} {'OOS_IR':>8} {'IC_rk':>6} {'IR_rk':>6} {'GapRk':>6} {'Cmp':>6}")
    print("-"*115)
    for _,r in s.nsmallest(25,"cmp").iterrows():
        print(f"{r['reg_lambda']:>5.1f} {r['reg_alpha']:>5.1f} {r['mcs']:>5.0f} "
              f"{r['train_ic']:>9.4f} {r['oos_ic']:>9.4f} {r['ic_gap']:>9.4f} "
              f"{r['ic_gap_ratio']:>8.1%} {r['oos_ir']:>8.4f} "
              f"{r['ic_rk']:>6.0f} {r['ir_rk']:>6.0f} {r['gap_rk']:>6.0f} {r['cmp']:>6.1f}")

    # 热力图 mcs=50
    m50 = s[s["mcs"]==50]
    for metric,label in [("oos_ir","OOS IR"),("oos_ic","OOS IC"),("ic_gap_ratio","IC_gap%"),("ic_gap","IC_gap")]:
        pv = m50.pivot_table(index="reg_lambda",columns="reg_alpha",values=metric)
        print(f"\n{label} (mcs=50):")
        hdr = f"{'λ\\\\α':>5}" + "".join(f" {c:>6.1f}" for c in pv.columns)
        print(hdr)
        for idx,row in pv.iterrows():
            line = f"{idx:>5.1f}"
            for v in row:
                if pd.notna(v):
                    if metric=="ic_gap_ratio": line += f" {v:>5.0%} "
                    elif metric in ("oos_ir","oos_ic","ic_gap"): line += f" {v:>6.4f}"
                    else: line += f" {v:>6.3f}"
                else: line += f"    N/A"
            print(line)

    # 热力图 mcs=65
    m65 = s[s["mcs"]==65]
    for metric,label in [("oos_ir","OOS IR"),("ic_gap_ratio","IC_gap%")]:
        pv = m65.pivot_table(index="reg_lambda",columns="reg_alpha",values=metric)
        print(f"\n{label} (mcs=65):")
        hdr = f"{'λ\\\\α':>5}" + "".join(f" {c:>6.1f}" for c in pv.columns)
        print(hdr)
        for idx,row in pv.iterrows():
            line = f"{idx:>5.1f}"
            for v in row:
                if pd.notna(v):
                    if metric=="ic_gap_ratio": line += f" {v:>5.0%} "
                    else: line += f" {v:>6.4f}"
                else: line += f"    N/A"
            print(line)

    # 配对 t 检验
    print(f"\n{'='*115}")
    print(
        "配对 t 检验 "
        f"(vs AX1 当前: λ={bp['reg_lambda']}, α={bp['reg_alpha']}, mcs={bp['min_child_samples']})"
    )
    print(f"{'='*115}")
    cur = df[
        (df["reg_lambda"] == float(bp["reg_lambda"]))
        & (df["reg_alpha"] == float(bp["reg_alpha"]))
        & (df["mcs"] == int(bp["min_child_samples"]))
    ]
    co = cur.set_index("seed")["oos_ic"]; cg = cur.set_index("seed")["ic_gap"]
    tr = []
    for lam in L:
        for alp in A:
            for mcs in M:
                if (
                    lam == float(bp["reg_lambda"])
                    and alp == float(bp["reg_alpha"])
                    and mcs == int(bp["min_child_samples"])
                ):
                    continue
                c = df[(df["reg_lambda"]==lam)&(df["reg_alpha"]==alp)&(df["mcs"]==mcs)]
                if len(c)<3: continue
                co2 = c.set_index("seed")["oos_ic"]; cg2 = c.set_index("seed")["ic_gap"]
                common = co.index.intersection(co2.index)
                if len(common)<3: continue
                t1,p1 = stats.ttest_rel(co2[common].values,co[common].values)
                t2,p2 = stats.ttest_rel(cg2[common].values,cg[common].values)
                tr.append({"λ":lam,"α":alp,"mcs":mcs,
                           "d_oic":co2[common].mean()-co[common].mean(),
                           "d_gap":cg2[common].mean()-cg[common].mean(),
                           "p_ic":p1,"p_gap":p2})
    dt = pd.DataFrame(tr)

    # 显著改善 OOS IC
    imp = dt[(dt["d_oic"]>0)&(dt["p_ic"]<0.15)].sort_values("p_ic")
    print(f"\n显著改善 OOS IC (p<0.15):")
    print(f"{'λ':>5} {'α':>5} {'mcs':>5} {'ΔOOS_IC':>9} {'ΔIC_gap':>9} {'p(IC)':>8} {'p(gap)':>8}")
    print("-"*60)
    for _,r in imp.head(20).iterrows():
        sg = "***" if r["p_ic"]<0.01 else "**" if r["p_ic"]<0.05 else "*" if r["p_ic"]<0.1 else ""
        print(f"{r['λ']:>5.1f} {r['α']:>5.1f} {r['mcs']:>5.0f} {r['d_oic']:>+9.4f} {r['d_gap']:>+9.4f} {r['p_ic']:>8.4f}{sg:<3} {r['p_gap']:>8.4f}")

    # 显著减小 IC_gap
    gi = dt[(dt["d_gap"]<0)&(dt["p_gap"]<0.15)].sort_values("p_gap")
    print(f"\n显著减小 IC_gap (p<0.15):")
    print(f"{'λ':>5} {'α':>5} {'mcs':>5} {'ΔOOS_IC':>9} {'ΔIC_gap':>9} {'p(IC)':>8} {'p(gap)':>8}")
    print("-"*60)
    for _,r in gi.head(20).iterrows():
        sg = "***" if r["p_gap"]<0.01 else "**" if r["p_gap"]<0.05 else "*" if r["p_gap"]<0.1 else ""
        print(f"{r['λ']:>5.1f} {r['α']:>5.1f} {r['mcs']:>5.0f} {r['d_oic']:>+9.4f} {r['d_gap']:>+9.4f} {r['p_ic']:>8.4f} {r['p_gap']:>8.4f}{sg:<3}")

    # 双重改善
    both = dt[(dt["d_oic"]>0)&(dt["d_gap"]<0)].sort_values("p_ic")
    print(f"\n双重改善 (OOS_IC更高 AND IC_gap更小):")
    print(f"{'λ':>5} {'α':>5} {'mcs':>5} {'ΔOOS_IC':>9} {'ΔIC_gap':>9} {'p(IC)':>8} {'p(gap)':>8}")
    print("-"*60)
    for _,r in both.head(15).iterrows():
        print(f"{r['λ']:>5.1f} {r['α']:>5.1f} {r['mcs']:>5.0f} {r['d_oic']:>+9.4f} {r['d_gap']:>+9.4f} {r['p_ic']:>8.4f} {r['p_gap']:>8.4f}")

    df.to_csv("/home/tiger/rqalpha/tmp/reg_lambda_grid_results.csv",index=False)
    print(f"\n结果已保存 ({len(df)} 行), {time.time()-t0:.0f}s")

if __name__=="__main__":
    main()
