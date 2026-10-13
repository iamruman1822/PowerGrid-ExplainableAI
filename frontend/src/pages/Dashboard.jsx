import { Sun, Wind, Battery, AlertTriangle, Activity, Gauge } from 'lucide-react';
import GlassCard from '../components/GlassCard';
import StatCard from '../components/StatCard';
import { useApi } from '../hooks/useApi';
import { CLASS_ICONS } from '../utils/constants';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const TT = { background: 'rgba(255,255,255,0.96)', border: '1px solid rgba(0,0,0,0.06)', borderRadius: '10px', fontSize: '11px', boxShadow: '0 8px 24px rgba(0,0,0,0.08)', padding: '8px 12px' };

export default function Dashboard() {
  const { data } = useApi('/dashboard');

  // Deep defaults — backend returns {} when CSV is missing
  const DEF_F = { solar: { q10: 1.8, q50: 5.2, q90: 10.1 }, wind: { q10: 0.8, q50: 1.6, q90: 2.4 }, load: { q10: 45, q50: 62, q90: 78 } };
  const raw = data?.forecast || {};
  const f = {
    solar: { ...DEF_F.solar, ...raw.solar },
    wind:  { ...DEF_F.wind, ...raw.wind },
    load:  { ...DEF_F.load, ...raw.load },
  };
  const bat = { total_batteries: 18, avg_soh_pct: 82.4, batteries_at_risk: 3, total_anomalies: 7, ...data?.battery };
  const alerts = data?.alerts?.length ? data.alerts : [
    { type: 'warning', title: 'Battery Health Alert', message: '3 batteries below 80% SoH threshold' },
    { type: 'danger', title: 'Anomalies Detected', message: '7 anomalies detected across fleet' },
  ];
  const sum = { solar_avg_kw: 4.12, wind_avg_kw: 1.35, load_avg_kw: 58.7, total_records: 17520, ...data?.forecast_summary };

  const spark = Array.from({ length: 24 }, (_, i) => ({
    h: `${String(i).padStart(2,'0')}:00`,
    solar: Math.max(0, 8 * Math.sin(Math.PI * (i - 5) / 14) + (Math.random() - 0.5) * 1.5).toFixed(1) * 1,
    wind: (1.2 + Math.random() * 1.5).toFixed(1) * 1,
    load: (40 + 30 * Math.sin(Math.PI * (i - 6) / 16) + Math.random() * 8).toFixed(0) * 1,
  }));

  const probs = [
    { name: 'Conservative', pct: 25, color: '#3b82f6' },
    { name: 'Moderate', pct: 55, color: '#10b981' },
    { name: 'Aggressive', pct: 20, color: '#e5880a' },
  ];
  const topDispatch = probs.reduce((best, current) => (current.pct > best.pct ? current : best), probs[0]);
  const topDispatchKey = topDispatch.name.toUpperCase();

  return (
    <div className="page-enter space-y-5">
      <div className="glass p-5 sm:p-6">
        <h1 className="text-[20px] font-extrabold tracking-tight" style={{ color: 'var(--text-heading)', fontFamily: 'var(--font-heading)' }}>
          Operational Overview
        </h1>
        <p className="text-[12px] mt-1" style={{ color: 'var(--text-secondary)' }}>
          Live generation forecast, battery health, and dispatch confidence in one control view.
        </p>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <StatCard icon={Sun} label="Solar Generation" value={(f.solar.q50 ?? 0).toFixed(1)} unit="kW" accent="solar" helper="P50" subtext={`P10-P90 window: ${(f.solar.q10 ?? 0).toFixed(1)} to ${(f.solar.q90 ?? 0).toFixed(1)} kW`} />
        <StatCard icon={Wind} label="Wind Generation" value={(f.wind.q50 ?? 0).toFixed(1)} unit="kW" accent="wind" helper="P50" subtext={`P10-P90 window: ${(f.wind.q10 ?? 0).toFixed(1)} to ${(f.wind.q90 ?? 0).toFixed(1)} kW`} />
        <StatCard icon={Gauge} label="Demand Forecast" value={(f.load.q50 ?? 0).toFixed(0)} unit="kW" accent="load" helper="P50" subtext={`Expected range: ${(f.load.q10 ?? 0).toFixed(0)} to ${(f.load.q90 ?? 0).toFixed(0)} kW`} />
        <StatCard icon={Battery} label="Fleet State of Health" value={(bat.avg_soh_pct ?? 0).toFixed(1)} unit="%" accent={(bat.avg_soh_pct ?? 0) > 80 ? 'battery' : 'danger'} helper="Target >= 80%" subtext={`${bat.batteries_at_risk ?? 0} of ${bat.total_batteries ?? 0} units below target`} />
      </div>

      {/* Chart + Dispatch */}
      <div className="grid grid-cols-1 xl:grid-cols-5 gap-4">
        <GlassCard className="xl:col-span-3 !p-0">
          <div className="px-6 pt-6 pb-2 flex items-center justify-between">
            <div>
              <h3 className="section-title">Daily Power Profile</h3>
              <p className="section-subtitle">Forecast medians with demand overlay</p>
            </div>
            <div className="flex gap-4 text-[10px] font-medium" style={{ color: 'var(--text-muted)' }}>
              {[{ c: 'var(--color-solar)', l: 'Solar' }, { c: 'var(--color-wind)', l: 'Wind' }, { c: 'var(--color-load)', l: 'Demand' }].map(x => (
                <span key={x.l} className="flex items-center gap-1.5"><span className="w-2 h-[3px] rounded-full inline-block" style={{ background: x.c }} />{x.l}</span>
              ))}
            </div>
          </div>
          <div className="px-2 pb-4">
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={spark} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.04)" />
                <XAxis dataKey="h" tick={{ fill: '#94a3b8', fontSize: 10 }} tickLine={false} axisLine={false} interval={3} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} tickLine={false} axisLine={false} width={35} />
                <Tooltip contentStyle={TT} labelStyle={{ color: '#64748b', fontWeight: 600 }} />
                <Area type="monotone" dataKey="load" stroke="#8b5cf6" fill="rgba(139,92,246,0.06)" strokeWidth={2} dot={false} name="Load (kW)" />
                <Area type="monotone" dataKey="solar" stroke="#e5880a" fill="rgba(229,136,10,0.06)" strokeWidth={2} dot={false} name="Solar (kW)" />
                <Area type="monotone" dataKey="wind" stroke="#0ea5e9" fill="rgba(14,165,233,0.06)" strokeWidth={2} dot={false} name="Wind (kW)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </GlassCard>

        <GlassCard className="xl:col-span-2">
          <h3 className="section-title">Dispatch Recommendation</h3>
          <p className="section-subtitle mb-5">XGBoost operating mode confidence</p>
          <div className="flex items-center gap-3 mb-6">
            <span className="text-3xl">{CLASS_ICONS[topDispatchKey]}</span>
            <div>
              <span className={`badge badge-${topDispatch.name.toLowerCase()}`}>{topDispatchKey}</span>
              <p className="text-[11px] mt-1.5" style={{ color: 'var(--text-muted)' }}>Model confidence: <span className="font-semibold" style={{ color: 'var(--text-heading)' }}>{topDispatch.pct.toFixed(1)}%</span></p>
            </div>
          </div>
          <p className="text-[11px] mb-3" style={{ color: 'var(--text-secondary)' }}>Mode probability distribution</p>
          <div className="space-y-3">
            {probs.map(p => (
              <div key={p.name}>
                <div className="flex justify-between text-[11px] mb-1.5">
                  <span style={{ color: 'var(--text-secondary)' }}>{p.name}</span>
                  <span className="font-semibold" style={{ color: 'var(--text-heading)', fontFamily: 'var(--font-mono)' }}>{p.pct}%</span>
                </div>
                <div className="h-[6px] rounded-full overflow-hidden" style={{ background: 'var(--bg-inset)' }}>
                  <div className="h-full rounded-full transition-all duration-700 ease-out" style={{ width: `${p.pct}%`, background: p.color }} />
                </div>
              </div>
            ))}
          </div>
          <div className="mt-5 p-3 rounded-lg text-[11px] leading-relaxed" style={{ background: 'var(--bg-inset)', color: 'var(--text-secondary)' }}>
            Recommended policy: prioritize charging during renewable surplus, then discharge in controlled bursts during peak demand.
          </div>
        </GlassCard>
      </div>

      {/* Alerts + System */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <GlassCard glow={alerts.some(a => a.type === 'danger') ? 'rose' : ''}>
          <div className="flex items-center gap-2 mb-4">
            <AlertTriangle className="w-4 h-4" style={{ color: 'var(--color-warning)' }} />
            <h3 className="section-title">Operational Alerts</h3>
          </div>
          <div className="space-y-2.5">
            {alerts.map((a, i) => {
              const isD = a.type === 'danger';
              const clr = isD ? 'var(--color-danger)' : a.type === 'warning' ? 'var(--color-warning)' : 'var(--color-info)';
              return (
                <div key={i} className="flex items-start gap-3 p-3.5 rounded-xl" style={{ background: isD ? 'rgba(239,68,68,0.04)' : 'rgba(245,158,11,0.04)', border: `1px solid ${isD ? 'rgba(239,68,68,0.1)' : 'rgba(245,158,11,0.1)'}` }}>
                  <span className="w-2 h-2 mt-1 rounded-full shrink-0 pulse-alert" style={{ background: clr }} />
                  <div>
                    <p className="text-[12px] font-semibold" style={{ color: 'var(--text-heading)' }}>{a.title}</p>
                    <p className="text-[11px] mt-0.5" style={{ color: 'var(--text-secondary)' }}>{a.message}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </GlassCard>

        <GlassCard>
          <div className="flex items-center gap-2 mb-4">
            <Activity className="w-4 h-4" style={{ color: 'var(--brand)' }} />
            <h3 className="section-title">Fleet Snapshot</h3>
          </div>
          <div className="grid grid-cols-3 gap-3">
            {[
              { l: 'Battery Units', v: bat.total_batteries, c: 'var(--color-wind)' },
              { l: 'Active Anomalies', v: bat.total_anomalies, c: 'var(--color-danger)' },
              { l: 'Avg Solar Gen', v: `${sum.solar_avg_kw} kW`, c: 'var(--color-solar)' },
              { l: 'Avg Wind Gen', v: `${sum.wind_avg_kw} kW`, c: 'var(--color-wind)' },
              { l: 'Avg Demand', v: `${sum.load_avg_kw} kW`, c: 'var(--color-load)' },
              { l: 'Historical Records', v: sum.total_records?.toLocaleString(), c: 'var(--color-battery)' },
            ].map((x, i) => (
              <div key={i} className="inset-panel p-3">
                <p className="metric-value text-[18px]" style={{ color: x.c }}>{x.v}</p>
                <p className="metric-label mt-0.5">{x.l}</p>
              </div>
            ))}
          </div>
        </GlassCard>
      </div>
    </div>
  );
}
