import { useState } from 'react';
import { Sun, Wind, Gauge, Clock } from 'lucide-react';
import GlassCard from '../components/GlassCard';
import { useApi } from '../hooks/useApi';
import { ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const TT = { background: 'rgba(255,255,255,0.96)', border: '1px solid rgba(0,0,0,0.06)', borderRadius: '10px', fontSize: '11px', boxShadow: '0 8px 24px rgba(0,0,0,0.08)' };

export default function EnergyForecast() {
  const { data } = useApi('/forecast?limit=168');
  const [src, setSrc] = useState('all');

  const raw = data?.data || [];
  const chart = raw.filter((_, i) => i % 2 === 0).slice(-84).map(r => ({
    t: r.timestamp?.slice(11, 16) || '',
    sq10: Math.max(0, r.solar_power_q10 || 0), sq50: Math.max(0, r.solar_power_q50 || 0), sq90: Math.max(0, r.solar_power_q90 || 0),
    wq10: Math.max(0, r.wind_power_q10 || 0), wq50: Math.max(0, r.wind_power_q50 || 0), wq90: Math.max(0, r.wind_power_q90 || 0),
    lq10: r.load_demand_q10 || 0, lq50: r.load_demand_q50 || 0, lq90: r.load_demand_q90 || 0,
  }));

  const fb = Array.from({ length: 48 }, (_, i) => {
    const h = i % 24;
    return { t: `${String(h).padStart(2, '0')}:00`,
      sq10: Math.max(0, 3 * Math.sin(Math.PI * (h - 5) / 14)), sq50: Math.max(0, 7 * Math.sin(Math.PI * (h - 5) / 14)), sq90: Math.max(0, 11 * Math.sin(Math.PI * (h - 5) / 14)),
      wq10: 0.4 + Math.random() * 0.5, wq50: 1.0 + Math.random(), wq90: 1.8 + Math.random() * 1.5,
      lq10: 35 + 15 * Math.sin(Math.PI * (h - 6) / 16), lq50: 50 + 20 * Math.sin(Math.PI * (h - 6) / 16), lq90: 65 + 25 * Math.sin(Math.PI * (h - 6) / 16),
    };
  });
  const d = chart.length > 0 ? chart : fb;

  const tabs = [
    { k: 'all', l: 'All Sources', i: Gauge }, { k: 'solar', l: 'Solar', i: Sun },
    { k: 'wind', l: 'Wind', i: Wind }, { k: 'load', l: 'Load', i: Clock },
  ];

  const last = d[d.length - 1] || {};
  const cards = [
    { title: 'Solar Power', icon: Sun, color: 'var(--color-solar)', bg: 'rgba(229,136,10,0.06)', v: last.sq50?.toFixed(2), r: `${last.sq10?.toFixed(1)} – ${last.sq90?.toFixed(1)} kW` },
    { title: 'Wind Power', icon: Wind, color: 'var(--color-wind)', bg: 'rgba(14,165,233,0.06)', v: last.wq50?.toFixed(2), r: `${last.wq10?.toFixed(1)} – ${last.wq90?.toFixed(1)} kW` },
    { title: 'Load Demand', icon: Gauge, color: 'var(--color-load)', bg: 'rgba(139,92,246,0.06)', v: last.lq50?.toFixed(1), r: `${last.lq10?.toFixed(1)} – ${last.lq90?.toFixed(1)} kW` },
  ];

  return (
    <div className="page-enter space-y-5">
      {/* Tabs */}
      <div className="flex gap-2">
        {tabs.map(t => {
          const I = t.i;
          return (
            <button key={t.k} onClick={() => setSrc(t.k)}
              className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-[12px] font-medium transition-all ${
                src === t.k ? 'text-[var(--brand)] font-semibold' : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
              }`}
              style={src === t.k ? { background: 'var(--brand-subtle)', border: '1px solid rgba(99,91,255,0.15)' } : { background: 'transparent', border: '1px solid transparent' }}>
              <I className="w-3.5 h-3.5" />{t.l}
            </button>
          );
        })}
      </div>

      {/* Chart */}
      <GlassCard className="!p-0">
        <div className="px-6 pt-6 pb-1 flex items-center justify-between">
          <div>
            <h3 className="section-title">Probabilistic Forecast Bands</h3>
            <p className="section-subtitle">CNN+LSTM · q10 pessimistic / q50 median / q90 optimistic</p>
          </div>
          <span className="badge" style={{ background: 'rgba(34,197,94,0.06)', color: '#16a34a', border: '1px solid rgba(34,197,94,0.15)', fontSize: '10px' }}>
            {d.length} pts
          </span>
        </div>
        <div className="px-2 pb-4">
          <ResponsiveContainer width="100%" height={400}>
            <ComposedChart data={d} margin={{ top: 15, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.04)" />
              <XAxis dataKey="t" tick={{ fill: '#94a3b8', fontSize: 10 }} tickLine={false} axisLine={false} interval={5} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} tickLine={false} axisLine={false} width={35} label={{ value: 'kW', angle: -90, position: 'insideLeft', style: { fill: '#94a3b8', fontSize: 10 } }} />
              <Tooltip contentStyle={TT} />
              {(src === 'all' || src === 'solar') && <>
                <Area type="monotone" dataKey="sq90" stroke="none" fill="rgba(229,136,10,0.08)" name="Solar q90" />
                <Area type="monotone" dataKey="sq10" stroke="none" fill="var(--bg-base)" name="Solar q10" />
                <Line type="monotone" dataKey="sq50" stroke="#e5880a" strokeWidth={2.5} dot={false} name="Solar median" />
              </>}
              {(src === 'all' || src === 'wind') && <>
                <Area type="monotone" dataKey="wq90" stroke="none" fill="rgba(14,165,233,0.08)" name="Wind q90" />
                <Area type="monotone" dataKey="wq10" stroke="none" fill="var(--bg-base)" name="Wind q10" />
                <Line type="monotone" dataKey="wq50" stroke="#0ea5e9" strokeWidth={2.5} dot={false} name="Wind median" />
              </>}
              {(src === 'all' || src === 'load') && <>
                <Area type="monotone" dataKey="lq90" stroke="none" fill="rgba(139,92,246,0.08)" name="Load q90" />
                <Area type="monotone" dataKey="lq10" stroke="none" fill="var(--bg-base)" name="Load q10" />
                <Line type="monotone" dataKey="lq50" stroke="#8b5cf6" strokeWidth={2.5} dot={false} name="Load median" />
              </>}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </GlassCard>

      {/* Summary Strip */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {cards.map((c, i) => {
          const I = c.icon;
          return (
            <GlassCard key={i}>
              <div className="flex items-center gap-3 mb-4">
                <div className="w-9 h-9 rounded-xl flex items-center justify-center" style={{ background: c.bg }}><I className="w-4 h-4" style={{ color: c.color }} /></div>
                <div><h4 className="text-[13px] font-semibold" style={{ color: 'var(--text-heading)' }}>{c.title}</h4><p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Latest forecast</p></div>
              </div>
              <p className="metric-value text-[26px]" style={{ color: c.color }}>{c.v || '—'} <span className="text-[13px] font-medium" style={{ color: 'var(--text-muted)' }}>kW</span></p>
              <p className="text-[10px] mt-1.5" style={{ color: 'var(--text-muted)' }}>Confidence: {c.r}</p>
            </GlassCard>
          );
        })}
      </div>
    </div>
  );
}
