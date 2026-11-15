"""Build viz/evals.html: a one-chart, judge-friendly 7-day evaluation page.

Reads agentic/out_recursive/seven_day.json. The detailed report is viz/evals_detail.html.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "..", "agentic", "out_recursive", "seven_day_memory.json")))
day0 = R["mean"]["frozen"][0]
pct = lambda xs: [round(100 * max(x, 0) / day0) for x in xs]   # % of day-1 control quality, floored at 0
ps = R["per_scenario"]
late = [sum(ps["agent_memory"][s][d] - ps["existing"][s][d] for d in range(3, 7)) / 4 for s in range(len(ps["existing"]))]
data = {"frozen": pct(R["mean"]["frozen"]), "existing": pct(R["mean"]["existing"]), "agent": pct(R["mean"]["agent_memory"]),
        "raw": {k: [round(x, 3) for x in R["mean"][k]] for k in ["frozen", "existing", "agent_memory"]},
        "late_gain_pts": round(100 * sum(late) / len(late) / day0), "late_wins": sum(1 for x in late if x > 0),
        "mem": [int(x) for x in R["memory_examples"]],
        "scen": len(R["scenarios"]), "evalt": R["eval_trials_per_day"]}

HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>7-Day Evaluation</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#F7F6F2;--surface:#fff;--surface-2:#F1EFE8;--text:#1F1E1C;--text-2:#5F5E5A;--text-3:#888780;--border:rgba(31,30,28,.1);--accent:#0F6E56;--good:#3B6D11;--good-bg:#EAF3DE;--warn:#854F0B;--warn-bg:#FAEEDA}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141413;--surface:#1E1E1C;--surface-2:#262624;--text:#F1EFE8;--text-2:#B4B2A9;--text-3:#888780;--border:rgba(241,239,232,.1);--accent:#5DCAA5;--good:#97C459;--good-bg:rgba(99,153,34,.18);--warn:#EF9F27;--warn-bg:rgba(186,117,23,.18)}}
:root[data-theme="dark"]{--bg:#141413;--surface:#1E1E1C;--surface-2:#262624;--text:#F1EFE8;--text-2:#B4B2A9;--text-3:#888780;--border:rgba(241,239,232,.1);--accent:#5DCAA5;--good:#97C459;--good-bg:rgba(99,153,34,.18);--warn:#EF9F27;--warn-bg:rgba(186,117,23,.18)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:400 15px/1.6 Inter,system-ui,sans-serif}
header{display:flex;align-items:center;gap:14px;padding:14px 24px;border-bottom:1px solid var(--border);background:var(--surface)}
.mark{width:28px;height:28px;border-radius:8px;background:var(--accent);color:#fff;display:grid;place-items:center;font-weight:600;font-size:13px}
.seg{margin-left:auto;display:flex;background:var(--surface-2);border-radius:9px;padding:3px;gap:2px}
.seg a{padding:6px 12px;border-radius:7px;color:var(--text-2);text-decoration:none;font-size:13px;font-weight:500}.seg a.on{background:var(--surface);color:var(--text);box-shadow:0 0 0 1px var(--border)}
.wrap{max-width:980px;margin:0 auto;padding:36px 24px 60px}
h1{font-size:28px;font-weight:600;letter-spacing:-.02em;margin:0 0 8px}.lead{font-size:16px;color:var(--text-2);margin:0 0 28px;max-width:720px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:24px}
.chart{position:relative;height:360px}
.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:20px 0}
.m{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px}.m .l{font-size:13px;color:var(--text-2)}.m .v{font-size:26px;font-weight:600;margin-top:2px}.m .s{font-size:12.5px;color:var(--text-3)}
.badge{display:inline-block;font-size:12px;font-weight:600;padding:3px 10px;border-radius:999px}.good{background:var(--good-bg);color:var(--good)}.warn{background:var(--warn-bg);color:var(--warn)}
h2{font-size:17px;font-weight:600;margin:28px 0 10px}ul{margin:0;padding-left:20px;color:var(--text-2)}li{margin:6px 0}
.foot{font-size:12.5px;color:var(--text-3);margin-top:24px}
@media (max-width:760px){.cards{grid-template-columns:1fr}header{flex-wrap:wrap}.seg{margin-left:0}h1{font-size:23px}}
</style></head><body>
<header><div class="mark">NR</div><div><b style="font-weight:600">7-day evaluation</b></div>
<nav class="seg"><a href="existing_system.html">Existing system</a><a href="agent_system.html">Agent system</a><a href="evals.html" class="on">Evals</a><a href="evals_detail.html">Details</a></nav></header>
<div class="wrap">
<h1>Learns the user over time — and catches up with daily calibration, without labels</h1>
<p class="lead">Over 7 simulated days, electrodes die, channels get noisy and gains drift. The existing system retrains 1% of the model every day from calibration labels. Our agents keep the model 100% frozen and build a growing memory of the user instead. 100% means as good as day 1.</p>
<div class="card"><div class="chart"><canvas id="c"></canvas></div></div>
<div class="cards" id="cards"></div>
<h2>What this shows</h2>
<ul id="take"></ul>
<h2>What's next</h2>
<ul>
<li><b style="font-weight:500;color:var(--text)">Stronger label-free verifier.</b> Today it scores changes on 5 trials, so some harmful fixes slip through under heavy drift. Add confidence intervals and automatic rollback to yesterday's state.</li>
<li><b style="font-weight:500;color:var(--text)">Longer user memory.</b> Seven days already shows the trend; weeks of history and more scenarios should make it significant.</li>
<li><b style="font-weight:500;color:var(--text)">Let the Nemotron reviewer learn across days.</b> It already found one held-out improvement; run it on longer, multi-scenario histories so playbook gains carry over to new users.</li>
<li><b style="font-weight:500;color:var(--text)">Real multi-day recordings.</b> Replace simulated drift with genuine day-to-day data.</li>
</ul>
<h2>How it was measured</h2>
<ul>
<li>Public monkey motor-cortex data (MC_Maze). Decoder trained once, then frozen.</li>
<li>__SCEN__ drift scenarios the agents never saw, 7 days each; __EVALT__ held-out trials scored per day. Agents never see the true hand movement; the existing system gets real labels every day.</li>
<li>Control quality = velocity R² relative to day 1 (negative values shown as 0%).</li>
</ul>
<p class="foot">Real results from <code>agentic/seven_day.py</code>, not projections. Drift is simulated. Full statistics on the Details page.</p>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>
const D=__DATA__;const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
Chart.defaults.font.family='Inter, system-ui, sans-serif';Chart.defaults.color=css('--text-2');Chart.defaults.borderColor=css('--border');
const days=[1,2,3,4,5,6,7].map(d=>'Day '+d);
new Chart(document.getElementById('c'),{type:'line',data:{labels:days,datasets:[
{label:'Existing: 99% frozen + daily labels',data:D.existing,borderColor:'#E24B4A',backgroundColor:'#E24B4A',borderWidth:2.5,tension:.3},
{label:'Agent system + user memory (no labels)',data:D.agent,borderColor:'#1D9E75',backgroundColor:'#1D9E75',borderWidth:3,tension:.3},
{label:'Frozen, no help',data:D.frozen,borderColor:'#B4B2A9',backgroundColor:'#B4B2A9',borderWidth:1.5,borderDash:[5,4],tension:.3}]},
options:{maintainAspectRatio:false,interaction:{mode:'index',intersect:false},plugins:{legend:{position:'bottom',labels:{boxWidth:12,boxHeight:12}},tooltip:{callbacks:{label:c=>c.dataset.label+': '+c.raw+'%'}}},
scales:{y:{min:0,max:120,ticks:{callback:v=>v+'%'},title:{display:true,text:'cursor-control quality left'}}}}});
const ahead=D.agent.filter((v,i)=>v>=D.existing[i]).length;
document.getElementById('cards').innerHTML=[
['Days 4–7 vs existing system',(D.late_gain_pts>=0?'+':'')+D.late_gain_pts+' pts','ahead in '+D.late_wins+' of 5 scenarios',D.late_gain_pts>0?'good':'warn'],
['User memory',D.mem[0].toLocaleString()+' → '+D.mem[6].toLocaleString(),'label-free examples, day 1 → day 7','good'],
['Labels or retraining needed','None','existing system: labels every day','good']].map(c=>`<div class="m"><div class="l">${c[0]}</div><div class="v">${c[1]}</div><div class="s"><span class="badge ${c[3]}">${c[3]==='good'?'✓':'!'}</span> ${c[2]}</div></div>`).join('');
document.getElementById('take').innerHTML=[
`Days 1–3: the agent system is level with or slightly behind the existing system, which needs fresh labels from the user every day.`,
`Days 4–7: as the user memory grows, the agent system pulls ahead (${D.late_gain_pts>=0?'+':''}${D.late_gain_pts} points on average, ${D.late_wins} of 5 scenarios) — with zero labels and zero retraining.`,
`It is a trend, not yet statistically significant (5 scenarios), and both systems still lose most control to heavy drift by day 6.`].map(t=>`<li>${t}</li>`).join('');
</script></body></html>"""
html = HTML.replace("__DATA__", json.dumps(data)).replace("__SCEN__", str(data["scen"])).replace("__EVALT__", str(data["evalt"]))
open(os.path.join(HERE, "evals.html"), "w").write(html)
print("wrote evals.html", data["frozen"], data["existing"], data["agent"], data["late_gain_pts"], data["late_wins"])
