"""共享盘任务队列 worker（每台机器一个常驻进程，管一张空闲 GPU）。

目录（QDIR）：pending/ 任务定义（json，永不移动）；claims/<job>/ 认领目录（mkdir 排他，含 owner/pid/hb/rc）；
              done/<job>.json、failed/<job>.json 终态；logs/<job>.log 任务日志；STOP 文件存在则不再领新任务。

任务 json：{"name","cmd","mem_gb","prio","needs":[路径...],"est_min","timeout_min","machines":["a","b","c"]}
cmd 里可以用 $PY（mmopsd python）、$D（项目根）、$MACHINE。

调度规则：
- 只领「依赖文件齐全、没有 done/failed/有效认领、机器允许、预计跑完不晚于硬停」的任务；
- 优先级：按 prio 升序；排头放不下就等它，但允许 EASY 回填：排在后面的任务只有「按预计时长能在排头腾出预算前跑完」才可插队，或只占用排头起跑后仍富余的显存，不饿死排头；
- 认领用 os.mkdir 排他；worker 每 30s 刷新 hb，超过 STALE 秒无心跳视为失效，可被回收；
- worker 重启会「收养」本机仍在运行的任务（pid 存活），不会重复跑；
- 任务命令被包成 (cmd); echo $? > claims/<job>/rc，退出码以 rc 文件为准；
- 到 HARD_STOP：不再领任务，杀掉运行中任务并退出。
"""

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time

B = "/data/ibu-bi-ai-share-disk"
D = f"{B}/zk/project/rose-distill"
PY = f"{B}/zk/envs/mmopsd/bin/python"
STALE = 300


def now():
    return time.time()


def jload(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def pid_alive(pid):
    return os.path.exists(f"/proc/{pid}") and "Z" not in (open(f"/proc/{pid}/stat").read().split(") ")[-1][:1])


def free_mem_gb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=20).stdout.strip().split(",")
        return (float(out[1]) - float(out[0])) / 1024.0
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", required=True)
    ap.add_argument("--qdir", default=f"{D}/queue")
    ap.add_argument("--budget-gb", type=float, default=42.0)
    ap.add_argument("--max-jobs", type=int, default=3)
    ap.add_argument("--hard-stop", required=True, help="本地时间 YYYY-MM-DD HH:MM")
    args = ap.parse_args()

    q = args.qdir
    for sub in ("pending", "claims", "done", "failed", "logs", "workers"):
        os.makedirs(f"{q}/{sub}", exist_ok=True)
    hard_stop = dt.datetime.strptime(args.hard_stop, "%Y-%m-%d %H:%M").timestamp()
    me = f"{args.machine}-{os.getpid()}"
    running = {}  # name -> dict(pid, proc|None, mem, start, timeout)
    wlog = open(f"{q}/workers/{args.machine}.log", "a")

    def log(msg):
        wlog.write(f"[{dt.datetime.now():%m-%d %H:%M:%S}] {msg}\n")
        wlog.flush()

    def kill(name, sig):
        try:
            os.killpg(running[name]["pid"], sig)
        except Exception:
            pass

    def finish(name, state, note=""):
        r = running.pop(name, None)
        rec = {"name": name, "machine": args.machine, "state": state, "note": note,
               "start": r["start"] if r else None, "end": now(),
               "minutes": round((now() - r["start"]) / 60, 1) if r else None}
        with open(f"{q}/{'done' if state == 'done' else 'failed'}/{name}.json", "w") as f:
            json.dump(rec, f)
        log(f"{state.upper()} {name} {rec['minutes']}min {note}")

    # 收养本机遗留的运行中任务
    for name in os.listdir(f"{q}/claims"):
        cdir = f"{q}/claims/{name}"
        try:
            owner = open(f"{cdir}/owner").read().strip()
            pid = int(open(f"{cdir}/pid").read().strip())
        except Exception:
            continue
        if owner.split("-")[0] != args.machine:
            continue
        if os.path.exists(f"{q}/done/{name}.json") or os.path.exists(f"{q}/failed/{name}.json"):
            continue
        job = jload(f"{q}/pending/{name}.json") or {}
        if pid_alive(pid) or os.path.exists(f"{cdir}/rc"):  # rc 已写 = 停机期间刚好跑完，交给主循环按 rc 记终态
            running[name] = {"pid": pid, "proc": None, "mem": job.get("mem_gb", 10),
                             "start": os.path.getmtime(f"{cdir}/owner"), "timeout": job.get("timeout_min", 120),
                             "est": job.get("est_min", 10)}
            log(f"ADOPT {name} pid={pid}")

    log(f"worker up me={me} hard_stop={args.hard_stop} budget={args.budget_gb}GB max_jobs={args.max_jobs} "
        f"adopted={list(running)}")
    def shadow(head_mem, t):
        """排头任务最早能起跑的时刻，以及那一刻扣掉排头后仍富余的预算 extra（回填用）。"""
        free = args.budget_gb - sum(r["mem"] for r in running.values())
        if free >= head_mem:
            return t, free - head_mem
        for end, mem in sorted((r["start"] + r["est"] * 60, r["mem"]) for r in running.values()):
            free += mem
            if free >= head_mem:
                return end, free - head_mem
        return float("inf"), 0.0

    last_hb = 0
    while True:
        t = now()
        if t - last_hb > 30:
            for name in running:
                try:
                    open(f"{q}/claims/{name}/hb", "w").write(str(t))
                except Exception:
                    pass
            last_hb = t

        for name, r in list(running.items()):
            alive = (r["proc"].poll() is None) if r["proc"] is not None else pid_alive(r["pid"])
            if not alive:
                rcf = f"{q}/claims/{name}/rc"
                try:
                    rc = int(open(rcf).read().strip())
                except Exception:
                    rc = r["proc"].returncode if r["proc"] is not None else 1
                finish(name, "done" if rc == 0 else "failed", f"rc={rc}")
            elif t - r["start"] > r["timeout"] * 60:
                kill(name, signal.SIGKILL)
                finish(name, "failed", "timeout")

        if t >= hard_stop:
            log("HARD_STOP reached")
            for name in list(running):
                kill(name, signal.SIGTERM)
            time.sleep(8)
            for name in list(running):
                kill(name, signal.SIGKILL)
                finish(name, "failed", "hard_stop")
            break

        if len(running) < args.max_jobs and not os.path.exists(f"{q}/STOP"):
            reserved = sum(r["mem"] for r in running.values())
            cands = []
            for fn in os.listdir(f"{q}/pending"):
                if not fn.endswith(".json"):
                    continue
                name = fn[:-5]
                if os.path.exists(f"{q}/done/{name}.json") or os.path.exists(f"{q}/failed/{name}.json"):
                    continue
                if name in running:
                    continue
                job = jload(f"{q}/pending/{fn}")
                if not job:
                    continue
                if job.get("machines") and args.machine not in job["machines"]:
                    continue
                if job["mem_gb"] > args.budget_gb:
                    continue
                cdir = f"{q}/claims/{name}"
                if os.path.exists(cdir):
                    hb = f"{cdir}/hb"
                    age = t - (os.path.getmtime(hb) if os.path.exists(hb) else os.path.getmtime(cdir))
                    if age < STALE:
                        continue
                if any(not os.path.exists(p) for p in job.get("needs", [])):
                    continue
                if t + job.get("est_min", 10) * 60 > hard_stop:
                    continue
                cands.append((job.get("prio", 50), name, job))
            cands.sort(key=lambda x: (x[0], x[1]))
            pick, fm, sh, extra = None, None, None, 0.0
            for i, (prio, name, job) in enumerate(cands):
                if fm is None:
                    fm = free_mem_gb()
                if reserved + job["mem_gb"] > args.budget_gb or fm < job["mem_gb"] + 1.0:
                    if i == 0:
                        sh, extra = shadow(job["mem_gb"], t)
                    continue
                if i == 0 or t + job.get("est_min", 10) * 60 <= sh or job["mem_gb"] <= extra:
                    pick = (prio, name, job)
                    break
            if pick:
                prio, name, job = pick
                if reserved + job["mem_gb"] <= args.budget_gb:
                    cdir = f"{q}/claims/{name}"
                    ok = True
                    try:
                        if os.path.exists(cdir):  # 只回收真正失效的认领，否则会抢走别人刚建的 claim
                            hb = f"{cdir}/hb"
                            age = now() - (os.path.getmtime(hb) if os.path.exists(hb) else os.path.getmtime(cdir))
                            if age < STALE:
                                ok = False
                        if ok:
                            if os.path.exists(cdir):
                                for x in os.listdir(cdir):
                                    os.remove(f"{cdir}/{x}")
                                os.rmdir(cdir)
                            os.mkdir(cdir)  # 排他认领
                    except FileExistsError:
                        ok = False
                    except Exception as e:
                        log(f"claim error {name}: {e}")
                        ok = False
                    if ok:
                        open(f"{cdir}/owner", "w").write(me)
                        open(f"{cdir}/hb", "w").write(str(now()))
                        cmd = job["cmd"].replace("$PY", PY).replace("$D", D).replace("$MACHINE", args.machine)
                        wrapped = f"({cmd}); echo $? > {cdir}/rc"
                        env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH=f"{D}/src", CUDA_VISIBLE_DEVICES="0",
                                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                                   PYTHONUNBUFFERED="1", JOB_NAME=name)
                        lf = open(f"{q}/logs/{name}.log", "a")
                        proc = subprocess.Popen(["bash", "-c", wrapped], stdout=lf, stderr=subprocess.STDOUT,
                                                cwd=D, env=env, start_new_session=True)
                        open(f"{cdir}/pid", "w").write(str(proc.pid))
                        running[name] = {"pid": proc.pid, "proc": proc, "mem": job["mem_gb"], "start": now(),
                                         "timeout": job.get("timeout_min", 120), "est": job.get("est_min", 10)}
                        reserved += job["mem_gb"]
                        log(f"START {name} prio={prio} mem={job['mem_gb']}GB reserved={reserved:.0f}GB "
                            f"running={len(running)}")
                        time.sleep(25)
        time.sleep(5)
    log("worker exit")


if __name__ == "__main__":
    main()
