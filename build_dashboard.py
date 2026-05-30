"""Run from the ohm repo root: python build_dashboard.py"""
import zipfile, pathlib

ZIP = pathlib.Path(r'C:\Users\Sasha\Downloads\soh_prediction-handoff.zip')
OUT = pathlib.Path(__file__).parent / 'dashboard.html'

files_order = [
    'soh-data.js', 'tweaks-panel.jsx', 'soh-charts.jsx', 'soh-panels.jsx',
    'soh-explainer.jsx', 'soh-report.jsx', 'soh-flow.jsx', 'soh-landing.jsx', 'soh-stages.jsx',
]
with zipfile.ZipFile(ZIP) as z:
    src = {f: z.read('soh-prediction/project/' + f).decode('utf-8') for f in files_order}

# ── new soh-landing.jsx ───────────────────────────────────────────────────────
NEW_LANDING = """\
/* soh-landing.jsx */
(function () {
  function seedModels(data) {
    const all = data.cellIds;
    return [
      { id: 'm_nmc_v3', name: 'NMC-pouch · v3', typeKey: 'nmc', type: 'Graphite / NMC', form: 'Oxford pouch', cells: all.length, cellIds: all, r2: 0.987, rmse: 0.0094, mae: 0.0071, date: '2026-05-21', seed: true },
      { id: 'm_nmc_v2', name: 'NMC-pouch · v2', typeKey: 'nmc', type: 'Graphite / NMC', form: 'Oxford pouch', cells: 5, cellIds: all.slice(0, 5), r2: 0.981, rmse: 0.0112, mae: 0.0085, date: '2026-04-08', seed: true },
      { id: 'm_lfp_v1', name: 'LFP-cyl · baseline', typeKey: 'lfp', type: 'LFP / graphite', form: 'Severson cylindrical', cells: 8, cellIds: [], r2: 0.972, rmse: 0.0140, mae: 0.0102, date: '2026-03-02', seed: true, locked: true },
    ];
  }

  function ModelCard({ m, onUse }) {
    return (
      <div style={{ background: 'var(--panel)', border: '1px solid var(--border)', borderRadius: 11, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <div style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 11 }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
            <div style={{ flex: '1 1 auto', minWidth: 0 }}>
              <div style={{ font: '600 13.5px/1.2 "IBM Plex Sans", sans-serif', color: 'var(--text)' }}>{m.name}</div>
              <div style={{ font: '400 10px/1.3 "IBM Plex Mono", monospace', color: 'var(--faint)', marginTop: 3 }}>trained {m.date} · {m.cells} cells</div>
            </div>
            {m.locked && <span style={{ font: '500 9px/1 "IBM Plex Mono", monospace', color: 'var(--faint)', border: '1px solid var(--border)', borderRadius: 999, padding: '3px 7px' }}>demo only</span>}
          </div>
          <div style={{ display: 'flex', gap: 16 }}>
            <StatTile label="R²" value={m.r2.toFixed(3)} tone="accent" />
            <StatTile label="RMSE" value={m.rmse.toFixed(4)} />
            <StatTile label="MAE" value={m.mae.toFixed(4)} />
          </div>
          <button disabled={m.locked} onClick={() => !m.locked && onUse(m)}
            style={{ marginTop: 2, font: '600 11.5px/1 "IBM Plex Sans", sans-serif', color: m.locked ? 'var(--faint)' : 'var(--accent)', background: 'transparent', border: '1px solid ' + (m.locked ? 'var(--border)' : 'var(--accent)'), borderRadius: 8, padding: '8px 12px', cursor: m.locked ? 'default' : 'pointer', textAlign: 'left' }}>
            {m.locked ? 'No demo data available' : 'Use this model → predict on a new cell'}
          </button>
        </div>
      </div>
    );
  }

  function LandingStage({ pal, models, onTrainNew, onUseModel }) {
    const groups = {};
    for (const m of models) (groups[m.type] = groups[m.type] || []).push(m);
    const pipeline = ['ICA dQ/dV', 'ΔQ(V) features', 'ElasticNet LOCO', 'SOH · LLI · LAM · R↑'];
    return (
      <div style={{ height: '100%', overflow: 'auto' }}>
        <div style={{ width: 'min(980px, 100%)', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 24 }}>

          {/* hero */}
          <div style={{ padding: '36px 0 28px', borderBottom: '1px solid var(--border)' }}>
            <div style={{ font: '700 54px/1 "IBM Plex Mono", monospace', color: 'var(--text)', letterSpacing: '-.025em' }}>
              dQ/dV <span style={{ color: 'var(--accent)' }}>→</span> SOH
            </div>
            <div style={{ marginTop: 14, font: '400 15px/1.6 "IBM Plex Sans", sans-serif', color: 'var(--dim)', maxWidth: 620 }}>
              Upload battery cycle data, train a physics-aware ElasticNet on incremental capacity signatures, and find out exactly which degradation mechanism is limiting your cells.
            </div>
            <div style={{ marginTop: 18, display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
              {pipeline.map((s, i, arr) => (
                <React.Fragment key={s}>
                  <span style={{ font: '500 11.5px/1 "IBM Plex Mono", monospace', color: i === arr.length - 1 ? 'var(--accent)' : 'var(--dim)', whiteSpace: 'nowrap' }}>{s}</span>
                  {i < arr.length - 1 && <span style={{ color: 'var(--faint)', fontSize: 12 }}>→</span>}
                </React.Fragment>
              ))}
            </div>
          </div>

          {/* train new */}
          <button onClick={onTrainNew} style={{ display: 'flex', alignItems: 'center', gap: 14, textAlign: 'left', border: '1.5px dashed var(--accent)', background: 'var(--accent-ghost)', borderRadius: 12, padding: '16px 18px', cursor: 'pointer' }}>
            <div style={{ width: 38, height: 38, borderRadius: 10, background: 'var(--accent)', color: '#06120f', display: 'grid', placeItems: 'center', font: '300 22px/1 system-ui', flex: '0 0 auto' }}>+</div>
            <div>
              <div style={{ font: '600 13.5px/1.2 "IBM Plex Sans", sans-serif', color: 'var(--text)' }}>Train a new model</div>
              <div style={{ font: '400 11px/1.3 "IBM Plex Sans", sans-serif', color: 'var(--dim)', marginTop: 3 }}>Upload cycle CSVs for a batch of cells → leave-one-cell-out cross-validation</div>
            </div>
            <span style={{ marginLeft: 'auto', font: '600 18px/1 system-ui', color: 'var(--accent)' }}>→</span>
          </button>

          {/* model groups */}
          {Object.entries(groups).map(([type, ms]) => (
            <div key={type} style={{ display: 'flex', flexDirection: 'column', gap: 11 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
                <span style={{ font: '600 11px/1 "IBM Plex Sans", sans-serif', letterSpacing: '.05em', textTransform: 'uppercase', color: 'var(--dim)' }}>{type}</span>
                <span style={{ font: '400 10px/1 "IBM Plex Mono", monospace', color: 'var(--faint)' }}>{ms[0].form} · {ms.length} model{ms.length > 1 ? 's' : ''}</span>
                <span style={{ flex: '1 1 auto', height: 1, background: 'var(--border)' }} />
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
                {ms.map((m) => <ModelCard key={m.id} m={m} onUse={onUseModel} />)}
              </div>
            </div>
          ))}
          <div style={{ height: 8 }} />
        </div>
      </div>
    );
  }

  function Dropzone({ title, sub, onClick }) {
    return (
      <div onClick={onClick} style={{ border: '1.5px dashed var(--border)', borderRadius: 12, padding: '22px 20px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 7, background: 'var(--panel)', cursor: onClick ? 'pointer' : 'default' }}>
        <div style={{ width: 38, height: 38, borderRadius: 10, background: 'var(--accent-ghost)', display: 'grid', placeItems: 'center', color: 'var(--accent)', font: '300 21px/1 system-ui' }}>↥</div>
        <div style={{ font: '600 12.5px/1.3 "IBM Plex Sans", sans-serif', color: 'var(--text)' }}>{title}</div>
        <div style={{ font: '400 10.5px/1.3 "IBM Plex Mono", monospace', color: 'var(--faint)' }}>{sub}</div>
      </div>
    );
  }

  function UploadTrainStage({ pal, onBack, onTrain }) {
    const data = window.SOHData;
    const fileRef = React.useRef(null);
    const [files, setFiles] = React.useState([]);
    const [nomCap, setNomCap] = React.useState('5.0');
    const [demoSel, setDemoSel] = React.useState(() => new Set(data.cellIds));
    const onFiles = (e) => { const picked = Array.from(e.target.files || []); if (picked.length) setFiles(picked); };
    const toggleDemo = (id) => setDemoSel((s) => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n; });
    const hasReal = files.length > 0;
    const canTrain = hasReal || demoSel.size >= 2;
    const go = () => {
      if (hasReal) onTrain(files.map(f => f.name.replace(/\\.csv$/i, '')), files, parseFloat(nomCap) || 5.0);
      else onTrain([...demoSel], null, null);
    };
    return (
      <div style={{ height: '100%', overflow: 'auto' }}>
        <div style={{ width: 'min(820px, 100%)', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 15 }}>
          <BackLink onBack={onBack} label="Model library" crumb="Train new model" />
          <input ref={fileRef} type="file" multiple accept=".csv,.mat" style={{ display: 'none' }} onChange={onFiles} />
          <Dropzone title={hasReal ? files.length + ' file' + (files.length > 1 ? 's' : '') + ' selected' : 'Click to upload cycle data'} sub={hasReal ? files.map(f => f.name).join(', ') : 'Oxford .mat or per-cell CSVs (time, voltage, current)'} onClick={() => fileRef.current && fileRef.current.click()} />
          {hasReal && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, background: 'var(--panel)', border: '1px solid var(--border)', borderRadius: 9, padding: '10px 14px' }}>
              <span style={{ font: '500 11.5px/1 "IBM Plex Sans", sans-serif', color: 'var(--dim)', flex: '0 0 auto' }}>Nominal capacity (Ah)</span>
              <input type="number" step="0.1" min="0.1" value={nomCap} onChange={e => setNomCap(e.target.value)}
                style={{ width: 72, height: 28, borderRadius: 6, border: '1px solid var(--border)', background: 'var(--panel-2)', color: 'var(--text)', font: '500 13px/1 "IBM Plex Mono", monospace', textAlign: 'right', padding: '0 8px', outline: 'none' }} />
            </div>
          )}
          {!hasReal && (
            <Panel title="Or use demo batch" sub="select cells for cross-validation" right={<span style={{ font: '500 10.5px/1 "IBM Plex Mono", monospace', color: 'var(--accent)' }}>{demoSel.size} selected</span>} flush>
              <div>
                {data.cells.map((c, i) => {
                  const on = demoSel.has(c.id);
                  return (
                    <button key={c.id} onClick={() => toggleDemo(c.id)} style={rowBtn(i)}>
                      <Check on={on} />
                      <span style={{ font: '600 12.5px/1.2 "IBM Plex Mono", monospace', color: 'var(--text)', flex: '0 0 92px' }}>{c.id}</span>
                      <span style={{ font: '400 11px/1.2 "IBM Plex Mono", monospace', color: 'var(--dim)', flex: '1 1 auto' }}>{c.id.toLowerCase()}_cycle.csv</span>
                      <span style={{ font: '400 11px/1.2 "IBM Plex Mono", monospace', color: 'var(--faint)' }}>{fmt.cyc(c.maxCycle)} cyc · {c.tempC}°C</span>
                    </button>
                  );
                })}
              </div>
            </Panel>
          )}
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            <PrimaryBtn disabled={!canTrain} onClick={go}>Train model →</PrimaryBtn>
            <span style={{ font: '400 11px/1.4 "IBM Plex Sans", sans-serif', color: 'var(--faint)' }}>
              {hasReal ? 'Upload ' + files.length + ' cell' + (files.length > 1 ? 's' : '') + ' → LOCO cross-validation' : demoSel.size < 2 ? 'Select at least 2 cells' : 'Demo mode · LOCO CV across ' + demoSel.size + ' cells'}
            </span>
          </div>
          <div style={{ height: 8 }} />
        </div>
      </div>
    );
  }

  function UploadPredictStage({ pal, model, onBack, onPredict }) {
    const data = window.SOHData;
    const fileRef = React.useRef(null);
    const [file, setFile] = React.useState(null);
    const [nomCap, setNomCap] = React.useState('5.0');
    const compatible = model && model.cellIds && model.cellIds.length ? model.cellIds : data.cellIds;
    const [demoPick, setDemoPick] = React.useState(compatible[0]);
    const onFile = (e) => { const f = e.target.files && e.target.files[0]; if (f) setFile(f); };
    const hasReal = !!file && !!model && !model.seed;
    const go = () => {
      if (hasReal) onPredict(file.name.replace(/\\.csv$/i, ''), file, parseFloat(nomCap) || 5.0);
      else onPredict(demoPick, null, null);
    };
    return (
      <div style={{ height: '100%', overflow: 'auto' }}>
        <div style={{ width: 'min(820px, 100%)', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 15 }}>
          <BackLink onBack={onBack} label="Model library" crumb={model ? model.name + ' · predict' : 'predict'} />
          {model && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 14, background: 'var(--panel)', border: '1px solid var(--border)', borderRadius: 11, padding: '13px 16px' }}>
              <div style={{ flex: '1 1 auto' }}>
                <div style={{ font: '600 13px/1.2 "IBM Plex Sans", sans-serif', color: 'var(--text)' }}>{model.name}</div>
                <div style={{ font: '400 10.5px/1.3 "IBM Plex Mono", monospace', color: 'var(--faint)', marginTop: 3 }}>{model.type} · {model.form} · trained on {model.cells} cells</div>
              </div>
              <StatTile label="R²" value={model.r2.toFixed(3)} tone="accent" />
              <StatTile label="RMSE" value={model.rmse.toFixed(4)} />
            </div>
          )}
          <input ref={fileRef} type="file" accept=".csv,.mat" style={{ display: 'none' }} onChange={onFile} />
          <Dropzone title={file ? file.name : 'Click to upload a cell to score'} sub={file ? (file.size / 1024).toFixed(0) + ' KB · click to change' : (model && !model.seed ? 'Oxford .mat or cycle CSV · same chemistry as training' : 'Upload a real cell or pick a demo below')} onClick={() => fileRef.current && fileRef.current.click()} />
          {file && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, background: 'var(--panel)', border: '1px solid var(--border)', borderRadius: 9, padding: '10px 14px' }}>
              <span style={{ font: '500 11.5px/1 "IBM Plex Sans", sans-serif', color: 'var(--dim)', flex: '0 0 auto' }}>Nominal capacity (Ah)</span>
              <input type="number" step="0.1" min="0.1" value={nomCap} onChange={e => setNomCap(e.target.value)}
                style={{ width: 72, height: 28, borderRadius: 6, border: '1px solid var(--border)', background: 'var(--panel-2)', color: 'var(--text)', font: '500 13px/1 "IBM Plex Mono", monospace', textAlign: 'right', padding: '0 8px', outline: 'none' }} />
            </div>
          )}
          {!file && (
            <Panel title="Or score a demo cell" sub="pick one" flush>
              <div>
                {compatible.map((id, i) => {
                  const c = data.byId[id]; if (!c) return null;
                  const on = id === demoPick;
                  return (
                    <button key={id} onClick={() => setDemoPick(id)} style={rowBtn(i)}>
                      <Radio on={on} />
                      <span style={{ font: '600 12.5px/1.2 "IBM Plex Mono", monospace', color: 'var(--text)', flex: '0 0 92px' }}>{c.id}</span>
                      <span style={{ font: '400 11px/1.2 "IBM Plex Mono", monospace', color: 'var(--dim)', flex: '1 1 auto' }}>{c.id.toLowerCase()}_cycle.csv</span>
                      <span style={{ font: '400 11px/1.2 "IBM Plex Mono", monospace', color: 'var(--faint)' }}>{fmt.cyc(c.maxCycle)} cyc · {c.tempC}°C</span>
                    </button>
                  );
                })}
              </div>
            </Panel>
          )}
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            <PrimaryBtn onClick={go}>Run prediction →</PrimaryBtn>
            <span style={{ font: '400 11px/1.4 "IBM Plex Sans", sans-serif', color: 'var(--faint)' }}>
              {hasReal ? 'Extract dQ/dV → ΔQ(V) features → SOH + mechanism report' : 'Demo mode · uses synthetic data'}
            </span>
          </div>
          <div style={{ height: 8 }} />
        </div>
      </div>
    );
  }

  function BackLink({ onBack, label, crumb }) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, font: '500 11.5px/1 "IBM Plex Sans", sans-serif' }}>
        <button onClick={onBack} style={{ color: 'var(--accent)', background: 'none', border: 'none', cursor: 'pointer', font: 'inherit', padding: 0 }}>← {label}</button>
        <span style={{ color: 'var(--faint)' }}>/</span>
        <span style={{ color: 'var(--dim)' }}>{crumb}</span>
      </div>
    );
  }
  function PrimaryBtn({ children, onClick, disabled }) {
    return <button onClick={() => !disabled && onClick()} disabled={disabled} style={{ font: '600 13px/1 "IBM Plex Sans", sans-serif', color: '#06120f', background: disabled ? 'var(--track)' : 'var(--accent)', border: 'none', borderRadius: 9, padding: '12px 20px', cursor: disabled ? 'default' : 'pointer', boxShadow: disabled ? 'none' : '0 4px 16px -6px var(--accent)' }}>{children}</button>;
  }
  function Check({ on }) { return <span style={{ width: 17, height: 17, borderRadius: 5, border: '1.5px solid ' + (on ? 'var(--accent)' : 'var(--faint)'), background: on ? 'var(--accent)' : 'transparent', color: '#06120f', display: 'grid', placeItems: 'center', font: '700 11px/1 system-ui', flex: '0 0 auto' }}>{on ? '✓' : ''}</span>; }
  function Radio({ on }) { return <span style={{ width: 16, height: 16, borderRadius: '50%', border: '1.5px solid ' + (on ? 'var(--accent)' : 'var(--faint)'), display: 'grid', placeItems: 'center', flex: '0 0 auto' }}>{on && <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--accent)' }} />}</span>; }
  const rowBtn = (i) => ({ display: 'flex', alignItems: 'center', gap: 12, width: '100%', textAlign: 'left', padding: '11px 16px', cursor: 'pointer', background: 'transparent', border: 'none', borderTop: i ? '1px solid var(--border)' : 'none' });

  window.SOHLanding = { seedModels, LandingStage, UploadTrainStage, UploadPredictStage };
})();
"""

# ── new soh-stages.jsx ────────────────────────────────────────────────────────
NEW_STAGES = """\
/* soh-stages.jsx */
(function () {
  const P = () => window.SOHFlowParts;
  const L = () => window.SOHLanding;

  function TrainingStage({ cellIds, jobId, onDone }) {
    const data = window.SOHData;
    const [prog, setProg] = React.useState(0);
    const [fold, setFold] = React.useState(0);
    const [currentStage, setCurrentStage] = React.useState('');
    const [errMsg, setErrMsg] = React.useState(null);
    React.useEffect(() => {
      if (jobId) {
        let alive = true;
        const poll = async () => {
          try {
            const res = await fetch('/api/jobs/' + jobId);
            const d = await res.json();
            if (!alive) return;
            if (!res.ok || d.error === 'server_restarted') {
              setErrMsg('Server restarted mid-training — please try again'); return;
            }
            if (d.error) { setErrMsg(d.error); return; }
            const p = d.progress || 0;
            setProg(p);
            setFold(Math.min(cellIds.length, Math.floor(p * cellIds.length) + 1));
            if (d.current_stage) setCurrentStage(d.current_stage);
            if (d.status === 'done') { setTimeout(() => onDone(d), 400); return; }
            if (d.status === 'failed') { setErrMsg(d.error || 'Pipeline failed'); return; }
            setTimeout(poll, 700);
          } catch (_) { if (alive) setTimeout(poll, 1500); }
        };
        poll();
        return () => { alive = false; };
      } else {
        const total = 2200, t0 = performance.now(); let timer;
        const tick = () => {
          const p = Math.min(1, (performance.now() - t0) / total);
          setProg(p); setFold(Math.min(cellIds.length, Math.floor(p * cellIds.length) + 1));
          if (p < 1) timer = setTimeout(tick, 60); else setTimeout(() => onDone(null), 480);
        };
        tick(); return () => clearTimeout(timer);
      }
    }, [jobId]);
    return (
      <Centered>
        <div style={{ textAlign: 'center' }}>
          <div style={{ font: '600 16px/1.2 "IBM Plex Sans", sans-serif', color: 'var(--text)' }}>Training SOH model</div>
          <div style={{ font: '400 11.5px/1.4 "IBM Plex Mono", monospace', color: 'var(--dim)', marginTop: 5 }}>ElasticNet · leave-one-cell-out cross-validation</div>
        </div>
        {errMsg ? <div style={{ font: '400 12px/1.5 "IBM Plex Mono", monospace', color: 'var(--danger)', textAlign: 'center', padding: '0 20px' }}>Error: {errMsg}</div> : <Bar prog={prog} />}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
          {cellIds.map((id, i) => {
            const st = prog >= 1 || i < fold - 1 ? 'done' : i === fold - 1 ? 'run' : 'wait';
            return (
              <div key={id} style={{ display: 'flex', alignItems: 'center', gap: 10, font: '400 11.5px/1 "IBM Plex Mono", monospace', color: st === 'wait' ? 'var(--faint)' : 'var(--dim)' }}>
                <Dot st={st} /><span>fold {i + 1} — hold out {id}</span>{st === 'run' && <span style={{ color: 'var(--accent)' }}>training…</span>}
              </div>
            );
          })}
        </div>
        {currentStage && prog < 1 && (
          <div style={{ font: '400 10.5px/1 "IBM Plex Mono", monospace', color: 'var(--faint)', textAlign: 'center' }}>{currentStage}</div>
        )}
        <div style={{ textAlign: 'center', font: '600 12px/1 "IBM Plex Mono", monospace', color: 'var(--accent)' }}>
          {prog >= 1 ? 'R² ' + data.overall.r2.toFixed(3) + ' · RMSE ' + data.overall.rmse.toFixed(4) : Math.round(prog * 100) + '%'}
        </div>
      </Centered>
    );
  }

  function PredictingStage({ cellId, predictPromise, onDone }) {
    const steps = ['Extract dQ/dV ICA curves', 'Compute ΔQ(V) features', 'Predict SOH (ElasticNet)', 'Interpret degradation mechanisms'];
    const [prog, setProg] = React.useState(0);
    const [errMsg, setErrMsg] = React.useState(null);
    React.useEffect(() => {
      const total = 1500, t0 = performance.now(); let timer;
      const cap = predictPromise ? 0.88 : 1.0;
      const tick = () => {
        const p = Math.min(cap, (performance.now() - t0) / total);
        setProg(p);
        if (p < cap) timer = setTimeout(tick, 55);
        else if (!predictPromise) setTimeout(() => onDone(null), 380);
      };
      tick();
      if (predictPromise) {
        predictPromise
          .then(d => { clearTimeout(timer); setProg(1); setTimeout(() => onDone(d), 380); })
          .catch(e => { clearTimeout(timer); setErrMsg((e && e.message) || 'Prediction failed'); });
      }
      return () => clearTimeout(timer);
    }, []);
    const active = Math.min(steps.length, Math.floor(prog * steps.length) + 1);
    return (
      <Centered>
        <div style={{ textAlign: 'center' }}>
          <div style={{ font: '600 16px/1.2 "IBM Plex Sans", sans-serif', color: 'var(--text)' }}>Scoring {cellId}</div>
          <div style={{ font: '400 11.5px/1.4 "IBM Plex Mono", monospace', color: 'var(--dim)', marginTop: 5 }}>running the interpretable SOH pipeline</div>
        </div>
        {errMsg ? <div style={{ font: '400 12px/1.5 "IBM Plex Mono", monospace', color: 'var(--danger)', textAlign: 'center' }}>Error: {errMsg}</div> : <Bar prog={prog} />}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {steps.map((s, i) => { const st = prog >= 1 || i < active - 1 ? 'done' : i === active - 1 ? 'run' : 'wait';
            return <div key={s} style={{ display: 'flex', alignItems: 'center', gap: 10, font: '400 12px/1 "IBM Plex Sans", sans-serif', color: st === 'wait' ? 'var(--faint)' : 'var(--dim)' }}><Dot st={st} /><span>{s}</span></div>;
          })}
        </div>
      </Centered>
    );
  }

  function ReportStage({ cell, selIdx, setIdx, setCell, pal, showBand, showGuide, setGuide, model, scopeIds, mode, batchCell }) {
    const dominant = cell.rows[selIdx].phys.dominant, conf = cell.rows[selIdx].phys.conf;
    const titlePrefix = cell.isBatch ? 'Batch average' : cell.id;
    const MainCol = (
      <div style={{ display: 'grid', gridTemplateRows: mode === 'single' ? 'auto auto 1fr auto auto' : 'auto 1fr auto auto', gap: 12, minHeight: 0 }}>
        {mode === 'single' && <SingleSummary cell={cell} model={model} selIdx={selIdx} pal={pal} />}
        <Explainer pal={pal} accent={pal.accent} collapsed={!showGuide} onToggle={() => setGuide(!showGuide)} />
        <Panel title={titlePrefix + ' · State of Health'} sub={cell.isBatch ? 'mean predicted trajectory' : 'predicted vs. measured'}
          right={<div style={{ display: 'flex', alignItems: 'center', gap: 14 }}><MechBadge dominant={dominant} confidence={conf} /><Scrubber cell={cell} selIdx={selIdx} onSelect={setIdx} pal={pal} /></div>}
          bodyStyle={{ padding: 8 }} style={{ minHeight: 200 }}>
          <SOHLine cell={cell} selIdx={selIdx} onSelect={setIdx} pal={pal} showBand={showBand} showPhases />
        </Panel>
        <Panel title="Mechanism timeline" sub={cell.isBatch ? 'mean dominant pathway over life' : 'dominant pathway over cycle life'} right={<TimelineLegend />} bodyStyle={{ padding: '8px 8px 4px' }} style={{ flex: '0 0 auto' }}>
          <div style={{ height: 78 }}><MechTimeline cell={cell} selIdx={selIdx} onSelect={setIdx} pal={pal} /></div>
        </Panel>
        <div style={{ flex: '0 0 auto' }}><MechReport cell={cell} selIdx={selIdx} /></div>
      </div>
    );
    if (mode === 'single') return <div style={{ height: '100%', minHeight: 0 }}>{MainCol}</div>;
    return (
      <div style={{ display: 'grid', gridTemplateColumns: '252px 1fr', gap: 12, height: '100%', minHeight: 0 }}>
        <LeftRail cell={cell} selIdx={selIdx} onSelectCell={setCell} pal={pal} model={model} scopeIds={scopeIds} batchCell={batchCell} />
        {MainCol}
      </div>
    );
  }

  function Centered({ children }) { return <div style={{ display: 'grid', placeItems: 'center', height: '100%' }}><div style={{ width: 'min(560px, 100%)', display: 'flex', flexDirection: 'column', gap: 20 }}>{children}</div></div>; }
  function Bar({ prog }) { return <div style={{ height: 7, borderRadius: 4, background: 'var(--track)', overflow: 'hidden' }}><div style={{ height: '100%', width: prog * 100 + '%', background: 'var(--accent)', borderRadius: 4, boxShadow: '0 0 10px var(--accent)', transition: 'width .08s linear' }} /></div>; }
  function Dot({ st }) { return <span style={{ width: 14, height: 14, borderRadius: '50%', display: 'grid', placeItems: 'center', font: '700 9px/1 system-ui', flex: '0 0 auto', background: st === 'done' ? 'var(--accent)' : 'transparent', border: st === 'done' ? 'none' : '1.5px solid ' + (st === 'run' ? 'var(--accent)' : 'var(--faint)'), color: '#06120f' }}>{st === 'done' ? '✓' : ''}</span>; }

  function SOHFlow({ theme = 'dark', accent = 'teal', density = 'comfortable', showBand = true }) {
    const { ACCENTS, buildPal, rootVars } = P();
    const { seedModels, LandingStage, UploadTrainStage, UploadPredictStage } = L();
    const data = window.SOHData;
    const [stage, setStage] = React.useState('landing');
    const [models, setModels] = React.useState(() => seedModels(data));
    const [activeModel, setActiveModel] = React.useState(null);
    const [scopeIds, setScopeIds] = React.useState(data.cellIds);
    const [trainIds, setTrainIds] = React.useState(data.cellIds);
    const [reportMode, setReportMode] = React.useState('batch');
    const [batchCell, setBatchCell] = React.useState(null);
    const [viewBatch, setViewBatch] = React.useState(false);
    const [cellId, setCellId] = React.useState('Cell_01');
    const [selIdx, setSelIdx] = React.useState(() => Math.round(data.byId['Cell_01'].nCycles * 0.82));
    const [showGuide, setShowGuide] = React.useState(true);
    const [jobId, setJobId] = React.useState(null);
    const predictPromiseRef = React.useRef(null);
    const displayed = (reportMode === 'batch' && viewBatch && batchCell) ? batchCell : (data.byId[cellId] || data.byId[data.cellIds[0]]);
    const cell = displayed;
    const si = Math.min(selIdx, cell.nCycles - 1);

    React.useEffect(() => {
      fetch('/api/models').then(r => r.json()).then(apiModels => {
        if (Array.isArray(apiModels) && apiModels.length > 0) {
          setModels(prev => {
            const realIds = new Set(apiModels.map(m => m.id));
            return [...apiModels, ...prev.filter(m => m.seed && !realIds.has(m.id))];
          });
        }
      }).catch(() => {});
    }, []);

    const setCell = (id) => {
      if (id === 'BATCH') { setViewBatch(true); setSelIdx(p => Math.min(p, (batchCell ? batchCell.nCycles : 81) - 1)); return; }
      const c = data.byId[id]; if (!c) return;
      setViewBatch(false); setCellId(id); setSelIdx(p => Math.min(p, c.nCycles - 1));
    };
    const goTrain = () => setStage('uploadTrain');
    const startTrain = (ids, files, nomCap) => {
      const useIds = ids && ids.length ? ids : data.cellIds;
      setTrainIds(useIds); setJobId(null);
      if (files && files.length > 0) {
        const fd = new FormData();
        for (const f of files) fd.append('files', f);
        fd.append('nominal_capacity_ah', String(nomCap || 5.0));
        fetch('/api/train', { method: 'POST', body: fd })
          .then(r => r.json()).then(res => {
            if (res.job_id) setJobId(res.job_id);
            if (res.cell_ids && res.cell_ids.length > 0) setTrainIds(res.cell_ids);
          }).catch(() => {});
      }
      setStage('training');
    };
    const finishTrain = (jobData) => {
      if (jobData && jobData.cells && Object.keys(jobData.cells).length > 0) {
        const realIds = Object.keys(jobData.cells);
        for (const [id, cellData] of Object.entries(jobData.cells)) {
          data.byId[id] = cellData;
          if (!data.cellIds.includes(id)) data.cellIds.push(id);
        }
        const nm = jobData.model;
        if (nm) { setModels(m => [nm, ...m.filter(x => x.id !== nm.id)]); setActiveModel(nm); }
        setScopeIds(realIds);
        const bc = jobData.batch;
        if (bc) { bc.nCells = realIds.length; setBatchCell(bc); }
        setViewBatch(!!bc); setReportMode(bc ? 'batch' : 'single');
        if (realIds[0] && data.byId[realIds[0]]) { setCellId(realIds[0]); setSelIdx(0); }
      } else {
        const ids = trainIds;
        const nextV = Math.max(3, ...models.filter(m => m.typeKey === 'nmc').map(m => +(m.name.match(/v(\\d+)/) || [0, 0])[1])) + 1;
        const nm = { id: 'm_' + Date.now(), name: 'NMC-pouch · v' + nextV, typeKey: 'nmc', type: 'Graphite / NMC', form: 'Oxford pouch', cells: ids.length, cellIds: ids, r2: data.overall.r2, rmse: data.overall.rmse, mae: data.overall.mae, date: '2026-05-30' };
        setModels(m => [nm, ...m]); setActiveModel(nm); setScopeIds(ids);
        const bc = data.buildBatch(ids); setBatchCell(bc); setViewBatch(true); setReportMode('batch');
        setCellId(ids[0]); setSelIdx(Math.round(bc.nCycles * 0.8));
      }
      setStage('report');
    };
    const useModel = (m) => {
      setActiveModel(m);
      const ids = m.cellIds && m.cellIds.length ? m.cellIds : data.cellIds;
      setScopeIds(ids); setStage('uploadPredict');
    };
    const runPredict = (id, file, nomCap) => {
      predictPromiseRef.current = null;
      const safeId = id || data.cellIds[0];
      setScopeIds([safeId]); setReportMode('single'); setViewBatch(false);
      if (data.byId[safeId]) { setCellId(safeId); setSelIdx(Math.round(data.byId[safeId].nCycles * 0.82)); }
      if (file && activeModel && !activeModel.seed) {
        const fd = new FormData();
        fd.append('file', file);
        fd.append('model_id', activeModel.id);
        fd.append('nominal_capacity_ah', String(nomCap || 5.0));
        predictPromiseRef.current = fetch('/api/predict', { method: 'POST', body: fd }).then(r => {
          if (!r.ok) return r.json().then(e => Promise.reject(new Error(e.error || r.statusText)));
          return r.json();
        });
      }
      setStage('predicting');
    };
    const onPredictDone = (apiData) => {
      if (apiData && apiData.cell) {
        const cd = apiData.cell;
        data.byId[cd.id] = cd; if (!data.cellIds.includes(cd.id)) data.cellIds.push(cd.id);
        setCellId(cd.id); setSelIdx(0); setScopeIds([cd.id]);
      }
      setStage('report');
    };
    const toLibrary = () => setStage('landing');
    const rgb = ACCENTS[accent] || ACCENTS.teal;
    const pal = React.useMemo(() => buildPal(theme, rgb), [theme, accent]);
    const vars = React.useMemo(() => rootVars(theme, rgb), [theme, accent]);
    const c = density === 'compact';
    return (
      <TweakCtx.Provider value={{ density }}>
        <div style={{ ...vars, width: '100%', height: '100%', boxSizing: 'border-box', padding: c ? 14 : 18, fontFamily: '"IBM Plex Sans", system-ui, sans-serif', display: 'flex', flexDirection: 'column', gap: 14, overflow: 'hidden' }}>
          <header style={{ display: 'flex', alignItems: 'center', gap: 16, flex: '0 0 auto' }}>
            <button onClick={toLibrary} style={{ display: 'flex', alignItems: 'center', gap: 10, background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}>
              <div style={{ width: 26, height: 26, borderRadius: 7, background: 'var(--accent)', display: 'grid', placeItems: 'center', color: '#06120f', font: '700 13px/1 "IBM Plex Mono", monospace' }}>S</div>
              <div style={{ textAlign: 'left' }}>
                <div style={{ font: '600 14px/1.1 "IBM Plex Sans", sans-serif', color: 'var(--text)', margin: '0px 0px 10px' }}>SOH Prediction</div>
                <div style={{ font: '400 9.5px/1.2 "IBM Plex Mono", monospace', color: 'var(--faint)', lineHeight: '1.4', margin: '10px 1px 1px' }}>dQ/dV · batch training</div>
              </div>
            </button>
            {stage === 'report' && activeModel && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, font: '500 11.5px/1 "IBM Plex Sans", sans-serif' }}>
                <span style={{ color: 'var(--faint)' }}>/</span>
                <button onClick={toLibrary} style={{ color: 'var(--accent)', background: 'none', border: 'none', cursor: 'pointer', font: 'inherit', padding: 0 }}>{activeModel.name}</button>
                <span style={{ color: 'var(--faint)' }}>/</span>
                <span style={{ color: 'var(--dim)' }}>{reportMode === 'single' ? 'prediction' : 'batch report'}</span>
              </div>
            )}
            {stage === 'report' && (
              <div style={{ marginLeft: 'auto', display: 'flex', gap: 22, alignItems: 'center' }}>
                <StatTile label="Pred SOH" value={fmt.soh(cell.sohPred[si])} tone="accent" />
                <StatTile label="RUL → 80%" value={cell.rul ? fmt.cyc(cell.rul) : '—'} unit={cell.rul ? 'cyc' : ''} />
                <StatTile label={cell.isBatch ? 'Batch end' : 'Dominant'} value={cell.cellDominant === 'EARLY' ? '—' : data.MECH[cell.cellDominant].short} />
              </div>
            )}
          </header>
          <div style={{ flex: '1 1 auto', minHeight: 0 }}>
            {stage === 'landing'       && <LandingStage pal={pal} models={models} onTrainNew={goTrain} onUseModel={useModel} />}
            {stage === 'uploadTrain'   && <UploadTrainStage pal={pal} onBack={toLibrary} onTrain={startTrain} />}
            {stage === 'training'      && <TrainingStage cellIds={trainIds} jobId={jobId} onDone={finishTrain} />}
            {stage === 'uploadPredict' && <UploadPredictStage pal={pal} model={activeModel} onBack={toLibrary} onPredict={runPredict} />}
            {stage === 'predicting'    && <PredictingStage cellId={cellId} predictPromise={predictPromiseRef.current} onDone={onPredictDone} />}
            {stage === 'report'        && <ReportStage cell={cell} selIdx={si} setIdx={setSelIdx} setCell={setCell} pal={pal} showBand={showBand} showGuide={showGuide} setGuide={setShowGuide} model={activeModel} scopeIds={scopeIds} mode={reportMode} batchCell={batchCell} />}
          </div>
        </div>
      </TweakCtx.Provider>
    );
  }

  window.SOHFlow = SOHFlow;
  window.ACCENTS = window.SOHFlowParts.ACCENTS;
})();
"""

replacements = {'soh-landing.jsx': NEW_LANDING, 'soh-stages.jsx': NEW_STAGES}

def script_tag(fname, content):
    tag = 'text/javascript' if fname.endswith('.js') else 'text/babel'
    return f'<script type="{tag}">\n{content}\n</script>'

blocks = [script_tag(f, replacements.get(f, src[f])) for f in files_order]

APP = """\
<script type="text/babel" data-presets="react">
const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "dark": true,
  "accent": "teal",
  "density": "comfortable",
  "showBand": true
}/*EDITMODE-END*/;

function App() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);
  const theme = t.dark ? 'dark' : 'light';
  return (
    <React.Fragment>
      <SOHFlow theme={theme} accent={t.accent} density={t.density} showBand={t.showBand} />
      <TweaksPanel>
        <TweakSection label="Appearance" />
        <TweakToggle label="Dark theme" value={t.dark} onChange={(v) => setTweak('dark', v)} />
        <TweakColor label="Accent" value={'rgb(' + (ACCENTS[t.accent] || ACCENTS.teal).join(',') + ')'}
          options={Object.keys(ACCENTS).map((k) => 'rgb(' + ACCENTS[k].join(',') + ')')}
          onChange={(v) => { const key = Object.keys(ACCENTS).find((k) => 'rgb(' + ACCENTS[k].join(',') + ')' === v); if (key) setTweak('accent', key); }} />
        <TweakRadio label="Density" value={t.density} options={['comfortable', 'compact']} onChange={(v) => setTweak('density', v)} />
        <TweakSection label="Charts" />
        <TweakToggle label="Uncertainty band" value={t.showBand} onChange={(v) => setTweak('showBand', v)} />
      </TweaksPanel>
    </React.Fragment>
  );
}

document.getElementById('loading').remove();
ReactDOM.createRoot(document.getElementById('root')).render(<App />);
</script>"""

html_parts = [
    '<!DOCTYPE html>',
    '<html lang="en">',
    '<head>',
    '<meta charset="UTF-8" />',
    '<meta name="viewport" content="width=device-width, initial-scale=1.0" />',
    '<title>Interpretable SOH · Batch Flow</title>',
    '<link rel="preconnect" href="https://fonts.googleapis.com" />',
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />',
    '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet" />',
    '<style>',
    '  * { box-sizing: border-box; }',
    '  html, body { margin: 0; padding: 0; height: 100%; background: #0b0c0f; }',
    '  body { font-family: "IBM Plex Sans", system-ui, sans-serif; -webkit-font-smoothing: antialiased; }',
    '  #root { height: 100vh; }',
    '  input[type="range"] { -webkit-appearance: none; appearance: none; background: transparent; height: 14px; }',
    '  input[type="range"]::-webkit-slider-runnable-track { height: 4px; border-radius: 2px; background: var(--track, rgba(255,255,255,.12)); }',
    '  input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 13px; height: 13px; border-radius: 50%; background: var(--accent, #28c4b2); margin-top: -4.5px; cursor: pointer; box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent,#28c4b2) 22%, transparent); }',
    '  input[type="range"]::-moz-range-track { height: 4px; border-radius: 2px; background: var(--track, rgba(255,255,255,.12)); }',
    '  input[type="range"]::-moz-range-thumb { width: 13px; height: 13px; border: none; border-radius: 50%; background: var(--accent, #28c4b2); cursor: pointer; }',
    '  ::-webkit-scrollbar { width: 9px; height: 9px; }',
    '  ::-webkit-scrollbar-thumb { background: rgba(255,255,255,.14); border-radius: 5px; }',
    '  #loading { position: fixed; inset: 0; display: flex; align-items: center; justify-content: center; color: #28c4b2; font: 500 13px/1 "IBM Plex Mono", monospace; letter-spacing: .1em; }',
    '</style>',
    '</head>',
    '<body>',
    '<div id="loading">LOADING…</div>',
    '<div id="root"></div>',
    '',
    '<script src="https://unpkg.com/react@18.3.1/umd/react.development.js" integrity="sha384-hD6/rw4ppMLGNu3tX5cjIb+uRZ7UkRJ6BPkLpg4hAu/6onKUg4lLsHAs9EBPT82L" crossorigin="anonymous"></script>',
    '<script src="https://unpkg.com/react-dom@18.3.1/umd/react-dom.development.js" integrity="sha384-u6aeetuaXnQ38mYT8rp6sbXaQe3NL9t+IBXmnYxwkUI2Hw4bsp2Wvmx4yRQF1uAm" crossorigin="anonymous"></script>',
    '<script src="https://unpkg.com/@babel/standalone@7.29.0/babel.min.js" integrity="sha384-m08KidiNqLdpJqLq95G/LEi8Qvjl/xUYll3QILypMoQ65QorJ9Lvtp2RXYGBFj1y" crossorigin="anonymous"></script>',
    '',
]

html_parts.extend(blocks)
html_parts.append('')
html_parts.append(APP)
html_parts.append('</body>')
html_parts.append('</html>')

html = '\n'.join(html_parts)
OUT.write_text(html, encoding='utf-8')
print(f'Written {len(html):,} chars to {OUT}')
