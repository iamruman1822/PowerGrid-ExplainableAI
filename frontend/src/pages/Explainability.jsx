import { useState } from 'react';
import { BrainCircuit, Image as ImageIcon, ExternalLink } from 'lucide-react';
import GlassCard from '../components/GlassCard';
import { useApi } from '../hooks/useApi';
import { API_BASE } from '../utils/constants';

export default function Explainability() {
  const { data } = useApi('/explain/available');
  const [sel, setSel] = useState(null);

  const DEMO_PLOTS = [
    { filename: 'shap_feature_importance.png', module: 'dispatch_engine', url: '/api/explain/images/shap_feature_importance.png', type: 'image' },
    { filename: 'shap_beeswarm_conservative.png', module: 'dispatch_engine', url: '/api/explain/images/shap_beeswarm_conservative.png', type: 'image' },
    { filename: 'shap_beeswarm_moderate.png', module: 'dispatch_engine', url: '/api/explain/images/shap_beeswarm_moderate.png', type: 'image' },
    { filename: 'confusion_matrix.png', module: 'dispatch_engine', url: '/api/explain/images/confusion_matrix.png', type: 'image' },
    { filename: 'loss_curve.png', module: 'forecaster', url: '/api/explain/images/loss_curve.png', type: 'image' },
    { filename: 'sample_forecast.png', module: 'forecaster', url: '/api/explain/images/sample_forecast.png', type: 'image' },
  ];
  const plots = data?.plots?.length ? data.plots : DEMO_PLOTS;

  const imgs = plots.filter(p => p.type === 'image');
  const htmls = plots.filter(p => p.type === 'html');
  const grouped = {};
  imgs.forEach(p => { (grouped[p.module] ??= []).push(p); });
  const labels = { dispatch_engine: 'Dispatch — SHAP', forecaster: 'CNN+LSTM Forecaster' };
  const fmt = f => f.replace(/\.(png|html)$/, '').replace(/_/g, ' ').replace(/shap /i, 'SHAP ').replace(/\b\w/g, c => c.toUpperCase());

  return (
    <div className="page-enter space-y-5">
      {/* Header */}
      <GlassCard className="flex items-center gap-4">
        <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ background: 'rgba(139,92,246,0.07)' }}>
          <BrainCircuit className="w-5 h-5" style={{ color: 'var(--color-load)' }} />
        </div>
        <div className="flex-1">
          <h3 className="section-title">Model Explainability</h3>
          <p className="section-subtitle">SHAP beeswarm, waterfall, feature importance & model performance</p>
        </div>
        <span className="badge" style={{ background: 'rgba(139,92,246,0.06)', color: '#7c3aed', border: '1px solid rgba(139,92,246,0.15)' }}>{plots.length} plots</span>
      </GlassCard>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-5">
        {/* Sidebar */}
        <div className="xl:col-span-3 space-y-4">
          {Object.entries(grouped).map(([mod, list]) => (
            <GlassCard key={mod} className="!p-4">
              <p className="text-[9px] font-semibold uppercase tracking-[0.12em] mb-2.5" style={{ color: 'var(--text-faint)' }}>{labels[mod] || mod}</p>
              <div className="space-y-1">
                {list.map((pl, i) => (
                  <button key={i} onClick={() => setSel(pl)}
                    className={`w-full text-left flex items-center gap-2 px-3 py-2 rounded-lg text-[11px] font-medium transition-all ${
                      sel?.filename === pl.filename ? 'font-semibold' : ''
                    }`}
                    style={sel?.filename === pl.filename
                      ? { background: 'var(--brand-subtle)', color: 'var(--brand)', border: '1px solid rgba(99,91,255,0.15)' }
                      : { color: 'var(--text-secondary)', border: '1px solid transparent' }}>
                    <ImageIcon className="w-3.5 h-3.5 shrink-0" /><span className="truncate">{fmt(pl.filename)}</span>
                  </button>
                ))}
              </div>
            </GlassCard>
          ))}
          {htmls.length > 0 && (
            <GlassCard className="!p-4">
              <p className="text-[9px] font-semibold uppercase tracking-[0.12em] mb-2.5" style={{ color: 'var(--text-faint)' }}>Interactive</p>
              {htmls.map((pl, i) => (
                <a key={i} href={`${API_BASE}${pl.url.replace('/api','')}`} target="_blank" rel="noopener noreferrer"
                  className="flex items-center gap-2 px-3 py-2 rounded-lg text-[11px] transition-all" style={{ color: 'var(--text-secondary)' }}>
                  <ExternalLink className="w-3.5 h-3.5 shrink-0" /><span className="truncate">{fmt(pl.filename)}</span>
                </a>
              ))}
            </GlassCard>
          )}
        </div>

        {/* Viewer */}
        <div className="xl:col-span-9">
          <GlassCard className="min-h-[520px]">
            {sel ? (<>
              <div className="flex items-center justify-between mb-5">
                <h3 className="section-title">{fmt(sel.filename)}</h3>
                <span className="badge" style={{ background: 'var(--bg-inset)', color: 'var(--text-muted)', border: '1px solid var(--border-subtle)', fontSize: '10px' }}>{sel.module}</span>
              </div>
              <div className="flex items-center justify-center rounded-xl p-6 min-h-[400px]" style={{ background: 'var(--bg-inset)' }}>
                <img src={`${API_BASE}${sel.url.replace('/api','')}`} alt={sel.filename}
                  className="max-w-full max-h-[480px] object-contain rounded-lg"
                  onError={e => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }} />
                <div className="hidden flex-col items-center">
                  <ImageIcon className="w-10 h-10 mb-3" style={{ color: 'var(--text-faint)' }} />
                  <p className="text-[12px]" style={{ color: 'var(--text-muted)' }}>Start the Flask backend to view plots</p>
                </div>
              </div>
            </>) : (
              <div className="flex flex-col items-center justify-center min-h-[440px]">
                <BrainCircuit className="w-14 h-14 mb-4" style={{ color: 'var(--text-faint)' }} />
                <h3 className="text-[14px] font-semibold" style={{ color: 'var(--text-secondary)' }}>Select a Visualization</h3>
                <p className="text-[11px] mt-1" style={{ color: 'var(--text-muted)' }}>Choose a plot from the sidebar</p>
              </div>
            )}
          </GlassCard>
        </div>
      </div>
    </div>
  );
}
