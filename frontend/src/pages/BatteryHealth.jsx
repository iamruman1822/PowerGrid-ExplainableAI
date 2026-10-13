import { Battery as BatteryIcon, Heart, AlertTriangle, TrendingDown } from 'lucide-react';
import GlassCard from '../components/GlassCard';
import { useApi } from '../hooks/useApi';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';

const TT = { background: 'rgba(255,255,255,0.96)', border: '1px solid rgba(0,0,0,0.06)', borderRadius: '10px', fontSize: '11px', boxShadow: '0 8px 24px rgba(0,0,0,0.08)' };

export default function BatteryHealth() {
  const { data: sd } = useApi('/battery');
  const { data: fd } = useApi('/battery/fleet');

  const DEMO_BATTERIES = [
    { battery: 'B0025', total_cycles: 168, soh_actual_pct: 87.2, rul_predicted_cycles: 312, anomaly_count: 2, mae_Ahr: 0.0184 },
    { battery: 'B0026', total_cycles: 168, soh_actual_pct: 85.1, rul_predicted_cycles: 280, anomaly_count: 1, mae_Ahr: 0.0156 },
    { battery: 'B0027', total_cycles: 168, soh_actual_pct: 72.3, rul_predicted_cycles: 45, anomaly_count: 5, mae_Ahr: 0.0221 },
    { battery: 'B0028', total_cycles: 168, soh_actual_pct: 91.0, rul_predicted_cycles: 450, anomaly_count: 0, mae_Ahr: 0.0098 },
    { battery: 'B0029', total_cycles: 158, soh_actual_pct: 68.5, rul_predicted_cycles: 0, anomaly_count: 8, mae_Ahr: 0.0310 },
    { battery: 'B0030', total_cycles: 158, soh_actual_pct: 82.7, rul_predicted_cycles: 210, anomaly_count: 3, mae_Ahr: 0.0175 },
  ];
  const batteries = sd?.data?.length ? sd.data : DEMO_BATTERIES;
  const DEF_FLEET = { total_batteries: batteries.length, avg_soh_pct: 81.1, min_soh_pct: 68.5, batteries_at_risk: 2, total_anomalies: 19 };
  const fleet = fd && Object.keys(fd).length > 0 ? { ...DEF_FLEET, ...fd } : DEF_FLEET;

  const sohColor = v => v >= 85 ? '#10b981' : v >= 75 ? '#f59e0b' : '#ef4444';
  const sohLabel = v => v >= 85 ? 'Healthy' : v >= 75 ? 'Aging' : 'Critical';

  const barData = batteries.map(b => ({
    id: b.battery,
    soh: parseFloat(b['soh_actual_%'] || b.soh_actual_pct || 80),
  }));

  return (
    <div className="page-enter space-y-5">
      {/* Fleet KPIs */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { icon: BatteryIcon, label: 'Total Fleet', value: fleet.total_batteries, color: 'var(--color-wind)' },
          { icon: Heart, label: 'Average SoH', value: `${fleet.avg_soh_pct}%`, color: fleet.avg_soh_pct > 80 ? 'var(--color-battery)' : 'var(--color-warning)' },
          { icon: AlertTriangle, label: 'At Risk (<80%)', value: fleet.batteries_at_risk, color: 'var(--color-danger)' },
          { icon: TrendingDown, label: 'Total Anomalies', value: fleet.total_anomalies, color: 'var(--color-warning)' },
        ].map((k, i) => {
          const I = k.icon;
          return (
            <GlassCard key={i}>
              <div className="flex items-center gap-2.5 mb-3">
                <I className="w-4 h-4" style={{ color: k.color }} />
                <span className="metric-label">{k.label}</span>
              </div>
              <p className="metric-value text-[24px]">{k.value}</p>
            </GlassCard>
          );
        })}
      </div>

      {/* SoH Chart */}
      <GlassCard className="!p-0">
        <div className="px-6 pt-6 pb-2">
          <h3 className="section-title">State of Health — Fleet Overview</h3>
          <p className="section-subtitle">LSTM predicted SoH per battery · Dashed line = 80% EOL threshold</p>
        </div>
        <div className="px-2 pb-4">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={barData} margin={{ top: 15, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.04)" />
              <XAxis dataKey="id" tick={{ fill: '#64748b', fontSize: 11, fontWeight: 500 }} tickLine={false} axisLine={false} />
              <YAxis domain={[0, 100]} tick={{ fill: '#94a3b8', fontSize: 10 }} tickLine={false} axisLine={false} width={35} />
              <Tooltip contentStyle={TT} formatter={v => [`${v.toFixed(1)}%`, 'SoH']} />
              <ReferenceLine y={80} stroke="#ef4444" strokeDasharray="6 4" strokeWidth={1.5} label={{ value: '80% EOL', position: 'right', fill: '#ef4444', fontSize: 10, fontWeight: 600 }} />
              <Bar dataKey="soh" radius={[8, 8, 0, 0]} barSize={40}>
                {barData.map((e, i) => <Cell key={i} fill={sohColor(e.soh)} fillOpacity={0.75} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </GlassCard>

      {/* Battery Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
        {batteries.map((b, i) => {
          const soh = parseFloat(b['soh_actual_%'] || b.soh_actual_pct || 80);
          const clr = sohColor(soh);
          const anom = b.anomaly_count || 0;
          return (
            <GlassCard key={i} glow={soh < 75 ? 'rose' : ''}>
              {/* Header */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: `${clr}10` }}>
                    <BatteryIcon className="w-4 h-4" style={{ color: clr }} />
                  </div>
                  <div>
                    <h4 className="text-[13px] font-bold" style={{ color: 'var(--text-heading)' }}>{b.battery}</h4>
                    <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{b.total_cycles} charge cycles</p>
                  </div>
                </div>
                <span className="badge" style={{ background: `${clr}0c`, color: clr, border: `1px solid ${clr}25` }}>{sohLabel(soh)}</span>
              </div>

              {/* SoH bar */}
              <div className="mb-4">
                <div className="flex justify-between text-[11px] mb-1.5">
                  <span style={{ color: 'var(--text-secondary)' }}>State of Health</span>
                  <span className="font-bold" style={{ color: clr, fontFamily: 'var(--font-mono)' }}>{soh.toFixed(1)}%</span>
                </div>
                <div className="h-2 rounded-full overflow-hidden" style={{ background: 'var(--bg-inset)' }}>
                  <div className="h-full rounded-full transition-all duration-700" style={{ width: `${soh}%`, background: clr }} />
                </div>
              </div>

              {/* Metrics Grid */}
              <div className="grid grid-cols-3 gap-2">
                {[
                  { l: 'RUL', v: typeof b.rul_predicted_cycles === 'number' ? b.rul_predicted_cycles : '—', u: 'cycles' },
                  { l: 'Anomalies', v: anom, u: '', warn: anom > 3 },
                  { l: 'MAE', v: (b.mae_Ahr || 0.015).toFixed(4), u: 'Ahr' },
                ].map((m, j) => (
                  <div key={j} className="inset-panel p-2.5 text-center">
                    <p className="text-[12px] font-bold" style={{ fontFamily: 'var(--font-heading)', color: m.warn ? 'var(--color-danger)' : 'var(--text-heading)' }}>{m.v}</p>
                    <p className="text-[9px] mt-0.5" style={{ color: 'var(--text-muted)' }}>{m.l} {m.u && `(${m.u})`}</p>
                  </div>
                ))}
              </div>
            </GlassCard>
          );
        })}
      </div>
    </div>
  );
}
