export default function StatCard({ icon: Icon, label, value, unit = '', accent = 'brand', subtext = '', helper = '' }) {
  const COLORS = {
    brand:   { bg: 'rgba(99,91,255,0.07)',   icon: '#635bff' },
    solar:   { bg: 'rgba(229,136,10,0.07)',   icon: '#e5880a' },
    wind:    { bg: 'rgba(14,165,233,0.07)',   icon: '#0ea5e9' },
    load:    { bg: 'rgba(139,92,246,0.07)',   icon: '#8b5cf6' },
    battery: { bg: 'rgba(16,185,129,0.07)',   icon: '#10b981' },
    danger:  { bg: 'rgba(239,68,68,0.07)',    icon: '#ef4444' },
    warning: { bg: 'rgba(245,158,11,0.07)',   icon: '#f59e0b' },
  };
  const c = COLORS[accent] || COLORS.brand;

  return (
    <div className={`glass stat-card stat-card-refined accent-${accent} p-5`}>
      <div className="flex items-start justify-between gap-3 mb-4">
        <div className="flex items-center gap-3 min-w-0">
          <div className="stat-card-icon w-9 h-9 rounded-xl flex items-center justify-center" style={{ background: c.bg }}>
            {Icon && <Icon className="w-[18px] h-[18px]" style={{ color: c.icon }} />}
          </div>
          <span className="stat-card-label truncate">{label}</span>
        </div>
        {helper && <span className="stat-chip">{helper}</span>}
      </div>
      <div className="metric-value text-[30px] leading-none tracking-tight">
        <span>{value}</span>
        {unit && <span className="text-[14px] font-medium ml-1.5" style={{ color: 'var(--text-muted)' }}>{unit}</span>}
      </div>
      {subtext && <p className="stat-card-subtext mt-2.5">{subtext}</p>}
    </div>
  );
}
