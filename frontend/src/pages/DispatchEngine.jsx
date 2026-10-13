import { useState } from 'react';
import { Zap, Send, RotateCcw } from 'lucide-react';
import GlassCard from '../components/GlassCard';
import { CLASS_COLORS, CLASS_ICONS } from '../utils/constants';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const TT = { background: 'rgba(255,255,255,0.96)', border: '1px solid rgba(0,0,0,0.06)', borderRadius: '10px', fontSize: '11px', boxShadow: '0 8px 24px rgba(0,0,0,0.08)' };

const INIT = {
  solar_q10: 1.8, solar_q50: 5.1, solar_q90: 10.5, wind_q10: 1.1, wind_q50: 1.6, wind_q90: 2.4,
  load_q10: 64, load_q50: 78, load_q90: 91, soh_pct: 76.4, soc_pct: 69, cell_temp_c: 47.5,
  hour: 14, month: 6, diesel_cost: 15.1, grid_available: 1,
};

const GROUPS = [
  { title: 'Solar Forecast', fields: [['solar_q10','q10',0.1],['solar_q50','q50',0.1],['solar_q90','q90',0.1]] },
  { title: 'Wind Forecast', fields: [['wind_q10','q10',0.1],['wind_q50','q50',0.1],['wind_q90','q90',0.1]] },
  { title: 'Load Demand', fields: [['load_q10','Low',1],['load_q50','Median',1],['load_q90','High',1]] },
  { title: 'Battery', fields: [['soh_pct','SoH %',1],['soc_pct','SoC %',1],['cell_temp_c','Temp °C',0.5]] },
  { title: 'Context', fields: [['hour','Hour',1],['month','Month',1],['diesel_cost','Diesel ₹/kWh',0.1],['grid_available','Grid',1]] },
];

export default function DispatchEngine() {
  const [p, setP] = useState(INIT);
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);

  const run = async () => {
    setBusy(true);
    try {
      const r = await fetch('/api/dispatch/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(p) });
      setRes(await r.json());
    } catch {
      setRes({
        decision: 'MODERATE', confidence: 0.612,
        class_probs: { CONSERVATIVE: 0.248, MODERATE: 0.612, AGGRESSIVE: 0.140 },
        drivers: [
          { rank: 1, feature: 'Battery charge level', value: '69.00', shap_value: 0.3421 },
          { rank: 2, feature: 'Battery health', value: '76.40', shap_value: -0.2815 },
          { rank: 3, feature: 'Solar generation', value: '5.10', shap_value: 0.1982 },
          { rank: 4, feature: 'Load demand', value: '78.00', shap_value: -0.1543 },
          { rank: 5, feature: 'Cell temperature', value: '47.50', shap_value: -0.1201 },
        ],
        recommendation: 'Balance renewable supply with battery support. Charge during surplus, draw modestly during deficit.',
      });
    }
    setBusy(false);
  };

  const probs = res ? Object.entries(res.class_probs || { CONSERVATIVE: 0.33, MODERATE: 0.34, AGGRESSIVE: 0.33 }).map(([n, v]) => ({ name: n, value: v * 100, fill: CLASS_COLORS[n]?.main || '#666' })) : [];
  const dc = res ? CLASS_COLORS[res.decision] : null;

  return (
    <div className="page-enter">
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-5">
        {/* ── Input Panel ── */}
        <div className="xl:col-span-4">
          <GlassCard className="!p-5">
            <div className="flex items-center justify-between mb-4">
              <h3 className="section-title">Sensor Inputs</h3>
              <button onClick={() => { setP(INIT); setRes(null); }} className="btn-ghost !py-1 !px-2.5 !text-[10px]">
                <RotateCcw className="w-3 h-3" /> Reset
              </button>
            </div>
            <div className="space-y-4 max-h-[58vh] overflow-y-auto pr-1">
              {GROUPS.map((g, gi) => (
                <div key={gi}>
                  <p className="text-[9px] font-semibold uppercase tracking-[0.12em] mb-2" style={{ color: 'var(--text-faint)' }}>{g.title}</p>
                  <div className="space-y-1.5">
                    {g.fields.map(([key, label, step]) => (
                      <div key={key} className="flex items-center gap-2">
                        <label className="text-[11px] w-24 shrink-0" style={{ color: 'var(--text-secondary)' }}>{label}</label>
                        <input type="number" value={p[key]} step={step}
                          onChange={e => setP(x => ({ ...x, [key]: parseFloat(e.target.value) || 0 }))}
                          className="input-field flex-1 text-right !py-1.5 text-[12px]" />
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
            <button onClick={run} disabled={busy} className="btn-primary w-full mt-5">
              <Send className="w-4 h-4" />{busy ? 'Running...' : 'Run Dispatch Engine'}
            </button>
          </GlassCard>
        </div>

        {/* ── Results ── */}
        <div className="xl:col-span-8 space-y-4">
          {res ? (<>
            {/* Decision Banner */}
            <GlassCard glow={res.decision === 'CONSERVATIVE' ? 'blue' : res.decision === 'AGGRESSIVE' ? 'amber' : 'emerald'}>
              <div className="flex items-center gap-5 mb-5">
                <span className="text-5xl leading-none">{CLASS_ICONS[res.decision]}</span>
                <div>
                  <span className="badge text-[12px] !px-4 !py-1.5" style={{ background: dc?.bg, color: dc?.text, border: `1px solid ${dc?.border}` }}>{res.decision}</span>
                  <p className="text-[13px] mt-2" style={{ color: 'var(--text-secondary)' }}>
                    Confidence: <span className="font-bold" style={{ color: 'var(--text-heading)', fontFamily: 'var(--font-mono)' }}>{(res.confidence * 100).toFixed(1)}%</span>
                  </p>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={76}>
                <BarChart data={probs} layout="vertical" margin={{ top: 0, right: 10, left: 0, bottom: 0 }}>
                  <XAxis type="number" domain={[0, 100]} hide />
                  <YAxis type="category" dataKey="name" tick={{ fill: '#64748b', fontSize: 11, fontWeight: 500 }} width={110} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={TT} formatter={v => [`${v.toFixed(1)}%`]} />
                  <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={14}>
                    {probs.map((e, i) => <Cell key={i} fill={e.fill} fillOpacity={0.8} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </GlassCard>

            {/* SHAP Drivers */}
            <GlassCard>
              <h3 className="section-title mb-1">SHAP Explanation</h3>
              <p className="section-subtitle mb-4">Top features driving this decision</p>
              <div className="space-y-2">
                {(res.drivers || []).map((d, i) => (
                  <div key={i} className="flex items-center gap-3 p-3.5 rounded-xl" style={{ background: 'var(--bg-inset)', border: '1px solid var(--border-subtle)' }}>
                    <span className="text-[11px] font-bold w-5 shrink-0" style={{ color: 'var(--text-faint)', fontFamily: 'var(--font-mono)' }}>#{d.rank}</span>
                    <div className="flex-1 min-w-0">
                      <p className="text-[12px] font-semibold" style={{ color: 'var(--text-heading)' }}>{d.feature}</p>
                      <p className="text-[10px] mt-0.5" style={{ color: 'var(--text-muted)' }}>
                        Value: <span style={{ color: 'var(--text-secondary)' }}>{d.value}</span>
                        <span className="mx-1.5">·</span>
                        SHAP: <span className="font-semibold" style={{ color: d.shap_value > 0 ? 'var(--color-battery)' : 'var(--color-danger)', fontFamily: 'var(--font-mono)' }}>{d.shap_value > 0 ? '+' : ''}{d.shap_value?.toFixed(4)}</span>
                      </p>
                    </div>
                    <div className="w-24 h-[6px] rounded-full overflow-hidden shrink-0" style={{ background: 'var(--bg-base)' }}>
                      <div className="h-full rounded-full" style={{ width: `${Math.min(Math.abs(d.shap_value) * 200, 100)}%`, background: d.shap_value > 0 ? 'var(--color-battery)' : 'var(--color-danger)' }} />
                    </div>
                  </div>
                ))}
              </div>
            </GlassCard>

            {/* Recommendation */}
            <GlassCard glow="brand">
              <h3 className="section-title mb-2">Operator Recommendation</h3>
              <p className="text-[13px] leading-relaxed" style={{ color: 'var(--text-secondary)' }}>{res.recommendation}</p>
            </GlassCard>
          </>) : (
            <GlassCard className="!py-20 text-center">
              <Zap className="w-14 h-14 mx-auto mb-5" style={{ color: 'var(--text-faint)' }} />
              <h3 className="text-[16px] font-bold" style={{ fontFamily: 'var(--font-heading)', color: 'var(--text-secondary)' }}>XGBoost Dispatch Engine</h3>
              <p className="text-[12px] mt-2 max-w-sm mx-auto" style={{ color: 'var(--text-muted)' }}>
                Configure sensor inputs and click <strong>Run Dispatch Engine</strong> for an AI-powered dispatch decision with full SHAP explainability.
              </p>
            </GlassCard>
          )}
        </div>
      </div>
    </div>
  );
}
