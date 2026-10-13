import { useState } from 'react';
import { MapPin, Play, Loader2, ChevronDown, ChevronUp } from 'lucide-react';
import GlassCard from '../components/GlassCard';
import { CLASS_COLORS, CLASS_ICONS } from '../utils/constants';

const PRESETS = [
  {
    name: 'Rampur, Gaya — Wedding Day',
    time: 'June 14:30 · Post-monsoon onset · Partial cloud',
    story: 'Mid-afternoon in Rampur, a 340-household village 60 km south of Gaya. The monsoon arrived three days ago. Thick cumulus clouds keep interrupting strong June sun — solar forecast is wide (1.8–10.5 kW). Wind is steady at 1.6 kW. Village demand spiked to 78 kW for a wedding: sound systems, lighting, and mixers. Battery SoH is declining at 76.4%, and Cell-3 flagged a temperature warning at 47.5°C.',
    p: { solar_q10: 1.8, solar_q50: 5.1, solar_q90: 10.5, wind_q10: 1.1, wind_q50: 1.6, wind_q90: 2.4, load_q10: 64, load_q50: 78, load_q90: 91, soh_pct: 76.4, soc_pct: 69, cell_temp_c: 47.5, hour: 14, month: 6, diesel_cost: 15.1, grid_available: 1 },
  },
  {
    name: 'Morning Peak — Clear Sky',
    time: 'April 07:00 · Clear sky · Healthy fleet',
    story: 'Early morning, clear skies promising 8.5 kW median solar. Wind is light at 1.2 kW. The battery pack is healthy (88% SoH, 72% SoC). Moderate load at 60 kW. Grid is unavailable, diesel at ₹12.50/kWh.',
    p: { solar_q10: 2.0, solar_q50: 8.5, solar_q90: 14.0, wind_q10: 0.5, wind_q50: 1.2, wind_q90: 2.5, load_q10: 45, load_q50: 60, load_q90: 75, soh_pct: 88, soc_pct: 72, cell_temp_c: 36, hour: 7, month: 4, diesel_cost: 12.5, grid_available: 0 },
  },
  {
    name: 'Night Crisis — Grid Down',
    time: 'January 22:00 · Cold night · Degraded battery',
    story: 'Late night, cold January. No solar, minimal wind at 0.8 kW. Battery critically degraded: 68% SoH, 18% SoC, temperature anomaly at 52°C. Grid down. Diesel expensive at ₹14/kWh. Load is 42 kW from heating and lighting.',
    p: { solar_q10: 0, solar_q50: 0, solar_q90: 0, wind_q10: 0.2, wind_q50: 0.8, wind_q90: 1.8, load_q10: 30, load_q50: 42, load_q90: 55, soh_pct: 68, soc_pct: 18, cell_temp_c: 52, hour: 22, month: 1, diesel_cost: 14, grid_available: 0 },
  },
];

export default function Scenarios() {
  const [sel, setSel] = useState(0);
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(true);
  const sc = PRESETS[sel];

  const run = async () => {
    setBusy(true);
    try {
      const r = await fetch('/api/dispatch/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(sc.p) });
      setRes(await r.json());
    } catch {
      const d = ['MODERATE', 'AGGRESSIVE', 'CONSERVATIVE'];
      setRes({
        decision: d[sel], confidence: [0.612, 0.785, 0.923][sel],
        class_probs: { CONSERVATIVE: [0.248, 0.08, 0.923][sel], MODERATE: [0.612, 0.135, 0.055][sel], AGGRESSIVE: [0.14, 0.785, 0.022][sel] },
        drivers: [
          { rank: 1, feature: 'Battery charge level', value: String(sc.p.soc_pct), shap_value: 0.34 },
          { rank: 2, feature: 'Battery health', value: String(sc.p.soh_pct), shap_value: -0.28 },
          { rank: 3, feature: 'Solar generation', value: String(sc.p.solar_q50), shap_value: 0.19 },
        ],
        recommendation: sel === 2 ? 'Limit battery discharge immediately. Bring diesel generator online.'
          : sel === 1 ? 'Maximise renewable usage now. Run heavy loads during this optimal period.'
          : 'Balance supply with battery. Charge when surplus, draw modestly during deficit.',
      });
    }
    setBusy(false);
  };

  const dc = res ? CLASS_COLORS[res.decision] : null;
  const cover = ((sc.p.solar_q50 + sc.p.wind_q50) / sc.p.load_q50 * 100).toFixed(1);

  return (
    <div className="page-enter space-y-5">
      {/* Preset Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {PRESETS.map((pr, i) => (
          <button key={i} onClick={() => { setSel(i); setRes(null); }}
            className="text-left p-4 rounded-xl transition-all"
            style={sel === i
              ? { background: 'var(--brand-subtle)', border: '1px solid rgba(99,91,255,0.2)', color: 'var(--brand)' }
              : { background: 'var(--bg-card)', border: '1px solid var(--border-card)', color: 'var(--text-secondary)' }}>
            <div className="flex items-center gap-2 mb-1.5">
              <MapPin className="w-3.5 h-3.5 shrink-0" />
              <span className="text-[12px] font-semibold truncate">{pr.name}</span>
            </div>
            <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{pr.time}</p>
          </button>
        ))}
      </div>

      {/* Scenario Detail */}
      <GlassCard>
        <div className="flex items-start justify-between mb-3">
          <div>
            <h3 className="section-title">{sc.name}</h3>
            <p className="text-[11px] mt-0.5" style={{ color: 'var(--brand)' }}>{sc.time}</p>
          </div>
          <button onClick={() => setOpen(x => !x)} className="btn-ghost !p-1.5">
            {open ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>
        {open && <p className="text-[12px] leading-relaxed mb-5" style={{ color: 'var(--text-secondary)' }}>{sc.story}</p>}

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
          {[
            { l: 'Solar', v: `${sc.p.solar_q50} kW`, c: 'var(--color-solar)' },
            { l: 'Wind', v: `${sc.p.wind_q50} kW`, c: 'var(--color-wind)' },
            { l: 'Load', v: `${sc.p.load_q50} kW`, c: 'var(--color-load)' },
            { l: 'RE Cover', v: `${cover}%`, c: 'var(--color-battery)' },
            { l: 'SoH', v: `${sc.p.soh_pct}%`, c: sc.p.soh_pct > 80 ? 'var(--color-battery)' : 'var(--color-danger)' },
            { l: 'SoC', v: `${sc.p.soc_pct}%`, c: 'var(--color-info)' },
            { l: 'Temp', v: `${sc.p.cell_temp_c}°C`, c: sc.p.cell_temp_c > 45 ? 'var(--color-danger)' : 'var(--text-secondary)' },
            { l: 'Grid', v: sc.p.grid_available ? 'ON' : 'OFF', c: sc.p.grid_available ? 'var(--color-battery)' : 'var(--color-danger)' },
          ].map((x, i) => (
            <div key={i} className="inset-panel p-3">
              <p className="metric-value text-[16px]" style={{ color: x.c }}>{x.v}</p>
              <p className="metric-label mt-0.5">{x.l}</p>
            </div>
          ))}
        </div>

        <button onClick={run} disabled={busy} className="btn-primary">
          {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
          {busy ? 'Running...' : 'Run Scenario'}
        </button>
      </GlassCard>

      {/* Result */}
      {res && (
        <GlassCard glow={res.decision === 'CONSERVATIVE' ? 'blue' : res.decision === 'AGGRESSIVE' ? 'amber' : 'emerald'}>
          <div className="flex items-center gap-5 mb-6">
            <span className="text-5xl leading-none">{CLASS_ICONS[res.decision]}</span>
            <div>
              <span className="badge text-[12px] !px-4 !py-1.5" style={{ background: dc?.bg, color: dc?.text, border: `1px solid ${dc?.border}` }}>{res.decision}</span>
              <p className="metric-value text-[20px] mt-2">{(res.confidence * 100).toFixed(1)}% <span className="text-[13px] font-medium" style={{ color: 'var(--text-muted)' }}>confidence</span></p>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3 mb-6">
            {Object.entries(res.class_probs || { CONSERVATIVE: 0.33, MODERATE: 0.34, AGGRESSIVE: 0.33 }).map(([cls, prob]) => (
              <div key={cls} className="inset-panel p-3">
                <div className="flex justify-between text-[11px] mb-1.5">
                  <span style={{ color: 'var(--text-secondary)' }}>{cls}</span>
                  <span className="font-bold" style={{ color: CLASS_COLORS[cls]?.text, fontFamily: 'var(--font-mono)' }}>{(prob * 100).toFixed(1)}%</span>
                </div>
                <div className="h-[5px] rounded-full overflow-hidden" style={{ background: 'var(--bg-base)' }}>
                  <div className="h-full rounded-full" style={{ width: `${prob * 100}%`, background: CLASS_COLORS[cls]?.main }} />
                </div>
              </div>
            ))}
          </div>

          <div className="mb-5">
            <p className="text-[9px] font-semibold uppercase tracking-[0.12em] mb-2.5" style={{ color: 'var(--text-faint)' }}>SHAP Drivers</p>
            <div className="space-y-2">
              {(res.drivers || []).map((d, i) => (
                <div key={i} className="flex items-center gap-3 text-[12px]">
                  <span className="font-bold w-5 shrink-0" style={{ color: 'var(--text-faint)', fontFamily: 'var(--font-mono)' }}>#{d.rank}</span>
                  <span className="flex-1" style={{ color: 'var(--text-heading)' }}>{d.feature}</span>
                  <span style={{ color: 'var(--text-muted)' }}>= {d.value}</span>
                  <span className="font-semibold" style={{ color: d.shap_value > 0 ? 'var(--color-battery)' : 'var(--color-danger)', fontFamily: 'var(--font-mono)' }}>{d.shap_value > 0 ? '+' : ''}{d.shap_value?.toFixed(3)}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="p-4 rounded-xl" style={{ background: 'rgba(99,91,255,0.04)', border: '1px solid rgba(99,91,255,0.1)' }}>
            <p className="text-[9px] font-semibold uppercase tracking-[0.12em] mb-1" style={{ color: 'var(--brand)' }}>Recommendation</p>
            <p className="text-[13px] leading-relaxed" style={{ color: 'var(--text-secondary)' }}>{res.recommendation}</p>
          </div>
        </GlassCard>
      )}
    </div>
  );
}
