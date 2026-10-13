import { useState, useEffect } from 'react';
import { Settings as SettingsIcon, Save, RotateCcw, Server, Database, Cpu, Palette, CheckCircle2 } from 'lucide-react';
import GlassCard from '../components/GlassCard';

export default function Settings() {
  const [cfg, setCfg] = useState({
    seq_len: 48, horizon: 24, batch_size: 32, lstm_hidden: 128, physics_penalty: 10.0,
    eol_threshold: 80, anomaly_z_score: 2.0, nominal_capacity: 2.0,
    rotor_area: 50.0, panel_area: 200.0, panel_efficiency: 0.18,
    api_url: 'http://localhost:5000', refresh_interval: 30,
  });
  const [saving, setSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);

  useEffect(() => {
    const token = localStorage.getItem('xplainable_ai_token');
    fetch('/api/settings', {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    })
      .then(res => res.json())
      .then(data => {
        if (data.settings) {
          setCfg(prev => ({ ...prev, ...data.settings }));
        }
      })
      .catch(err => console.warn('Failed to load settings:', err));
  }, []);

  const handleSave = async () => {
    setSaving(true);
    setSaveSuccess(false);
    try {
      const token = localStorage.getItem('xplainable_ai_token');
      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ settings: cfg }),
      });
      if (res.ok) {
        setSaveSuccess(true);
        setTimeout(() => setSaveSuccess(false), 3000);
      }
    } catch (err) {
      console.error('Failed to save settings:', err);
    } finally {
      setSaving(false);
    }
  };

  const reset = () => setCfg({
    seq_len: 48, horizon: 24, batch_size: 32, lstm_hidden: 128, physics_penalty: 10.0,
    eol_threshold: 80, anomaly_z_score: 2.0, nominal_capacity: 2.0,
    rotor_area: 50.0, panel_area: 200.0, panel_efficiency: 0.18,
    api_url: 'http://localhost:5000', refresh_interval: 30
  });

  const sections = [
    { title: 'CNN+LSTM Forecaster', icon: Cpu, color: 'var(--color-solar)', fields: [
      ['seq_len','Sequence Length','number',1], ['horizon','Forecast Horizon','number',1], ['batch_size','Batch Size','number',1], ['lstm_hidden','LSTM Hidden','number',1], ['physics_penalty','Physics Penalty','number',0.1],
    ]},
    { title: 'Battery Degradation', icon: Database, color: 'var(--color-wind)', fields: [
      ['eol_threshold','EOL SoH %','number',1], ['anomaly_z_score','Anomaly Z-Score','number',0.1], ['nominal_capacity','Capacity (Ahr)','number',0.1],
    ]},
    { title: 'Dispatch Engine', icon: Palette, color: 'var(--color-load)', fields: [
      ['rotor_area','Rotor Area (m²)','number',1], ['panel_area','Panel Area (m²)','number',1], ['panel_efficiency','Panel η','number',0.01],
    ]},
    { title: 'Backend & Database', icon: Server, color: 'var(--color-battery)', fields: [
      ['api_url','API URL','text'], ['refresh_interval','Refresh (sec)','number',5],
    ]},
  ];

  return (
    <div className="page-enter space-y-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ background: 'var(--bg-inset)' }}>
            <SettingsIcon className="w-5 h-5" style={{ color: 'var(--text-secondary)' }} />
          </div>
          <div>
            <h3 className="section-title text-[16px]">Configuration</h3>
            <p className="section-subtitle">Model parameters, thresholds & PostgreSQL backend</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {saveSuccess && (
            <span className="flex items-center gap-1.5 text-[12px] text-emerald-400 font-medium mr-2">
              <CheckCircle2 className="w-3.5 h-3.5" /> Saved to PostgreSQL
            </span>
          )}
          <button onClick={reset} className="btn-ghost"><RotateCcw className="w-3.5 h-3.5" /> Reset</button>
          <button onClick={handleSave} className="btn-primary" disabled={saving}>
            <Save className="w-3.5 h-3.5" /> {saving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {sections.map((s, si) => {
          const I = s.icon;
          return (
            <GlassCard key={si}>
              <div className="flex items-center gap-2.5 mb-5">
                <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: `color-mix(in srgb, ${s.color} 8%, transparent)` }}>
                  <I className="w-4 h-4" style={{ color: s.color }} />
                </div>
                <h4 className="section-title">{s.title}</h4>
              </div>
              <div className="space-y-3">
                {s.fields.map(([key, label, type, step]) => (
                  <div key={key} className="flex items-center justify-between gap-4">
                    <label className="text-[11px] font-medium" style={{ color: 'var(--text-secondary)' }}>{label}</label>
                    <input type={type} value={cfg[key]} step={step}
                      onChange={e => setCfg(c => ({ ...c, [key]: type === 'number' ? parseFloat(e.target.value) || 0 : e.target.value }))}
                      className="input-field w-32 text-right !py-1.5 text-[12px]" />
                  </div>
                ))}
              </div>
            </GlassCard>
          );
        })}
      </div>

      <GlassCard>
        <h4 className="section-title mb-4">System Stack</h4>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {[
            { l: 'Forecast', v: 'CNN-LSTM · PyTorch', c: 'var(--color-solar)' },
            { l: 'Battery', v: 'LSTM · Keras', c: 'var(--color-wind)' },
            { l: 'Dispatch', v: 'XGBoost + SHAP', c: 'var(--color-load)' },
            { l: 'Backend', v: 'Flask 3.1', c: 'var(--color-battery)' },
            { l: 'Database', v: 'PostgreSQL 18', c: '#38bdf8' },
          ].map((x, i) => (
            <div key={i} className="inset-panel p-3.5">
              <p className="text-[13px] font-semibold" style={{ color: x.c }}>{x.v}</p>
              <p className="metric-label mt-0.5">{x.l}</p>
            </div>
          ))}
        </div>
      </GlassCard>
    </div>
  );
}
