/**
 * Landing Page
 * ============
 * Premium landing page with hero, features, and auth modal.
 */

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthProvider';
import {
  Zap, Sun, Battery, BrainCircuit, ArrowRight,
  Mail, Lock, Eye, EyeOff, Loader2, CheckCircle2,
  X, ChevronRight, Shield, TrendingUp, Activity
} from 'lucide-react';

export default function Landing() {
  const { signIn, signUp, demoSignIn } = useAuth();
  const navigate = useNavigate();

  const [showAuth, setShowAuth] = useState(false);
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPwd, setShowPwd] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleDemoLogin = async () => {
    setError('');
    setLoading(true);
    try {
      await demoSignIn();
      navigate('/');
    } catch (err) {
      setError(err.message || 'Demo login failed');
    }
    setLoading(false);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setLoading(true);
    try {
      if (isLogin) {
        await signIn(email, password);
        navigate('/');
      } else {
        await signUp(email, password);
        setSuccess('Account created successfully in PostgreSQL! You can now sign in.');
        setIsLogin(true);
        setPassword('');
      }
    } catch (err) {
      setError(err.message || 'Authentication failed');
    }
    setLoading(false);
  };

  const features = [
    {
      icon: Sun, color: '#e5880a', bg: 'rgba(229,136,10,0.08)',
      title: 'CNN+LSTM Energy Forecast',
      desc: 'Physics-constrained deep learning model for 24-hour probabilistic solar, wind & load predictions with q10/q50/q90 uncertainty bands.',
    },
    {
      icon: Battery, color: '#0ea5e9', bg: 'rgba(14,165,233,0.08)',
      title: 'Battery Degradation LSTM',
      desc: 'NASA battery dataset-trained LSTM predicting State of Health, Remaining Useful Life, and real-time anomaly detection across 18-cell fleet.',
    },
    {
      icon: Zap, color: '#8b5cf6', bg: 'rgba(139,92,246,0.08)',
      title: 'XGBoost Dispatch Engine',
      desc: 'Intelligent microgrid dispatch decisions — Conservative, Moderate, or Aggressive — based on 34 real-time features with live inference.',
    },
    {
      icon: BrainCircuit, color: '#10b981', bg: 'rgba(16,185,129,0.08)',
      title: 'SHAP Explainability',
      desc: 'Every AI decision is mathematically explained using SHAP values. See exactly which sensor readings drove each dispatch recommendation.',
    },
  ];

  const techStack = [
    { name: 'PyTorch', color: '#ee4c2c' },
    { name: 'Keras', color: '#d00000' },
    { name: 'XGBoost', color: '#189fdd' },
    { name: 'SHAP', color: '#8b5cf6' },
    { name: 'Flask', color: '#000000' },
    { name: 'React', color: '#61dafb' },
    { name: 'PostgreSQL', color: '#336791' },
  ];

  const stats = [
    { value: '17,520', label: 'Forecast Data Points', icon: TrendingUp },
    { value: '18', label: 'Battery Cells Monitored', icon: Shield },
    { value: '34', label: 'Dispatch Features', icon: Activity },
    { value: '3', label: 'ML Models Integrated', icon: BrainCircuit },
  ];

  return (
    <div className="landing-page">
      {/* ── Animated Background ── */}
      <div className="landing-bg" />
      <div className="landing-particles">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="particle" style={{
            left: `${15 + i * 14}%`,
            animationDelay: `${i * 1.5}s`,
            animationDuration: `${12 + i * 3}s`,
          }} />
        ))}
      </div>

      {/* ── Nav Bar ── */}
      <nav className="landing-nav">
        <div className="landing-nav-inner">
          <div className="landing-logo">
            <div className="landing-logo-icon">
              <Zap className="w-6 h-6" style={{ color: 'white' }} />
            </div>
            <div>
              <span className="landing-logo-text">XplainableAI</span>
              <span className="landing-logo-sub">Microgrid Intelligence</span>
            </div>
          </div>
          <div className="landing-nav-links">
            <a href="#features">Features</a>
            <a href="#tech">Technology</a>
            <button className="btn-primary !py-2.5 !px-5 !text-[13px]" onClick={() => setShowAuth(true)}>
              Get Started <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className="landing-hero">
        <div className="landing-hero-badge">
          <CheckCircle2 className="w-3.5 h-3.5" style={{ color: '#10b981' }} />
          Research-Grade · Bihar Rural Microgrid
        </div>
        <h1 className="landing-hero-title">
          Explainable AI for
          <br />
          <span className="landing-hero-gradient">Smart Microgrids</span>
        </h1>
        <p className="landing-hero-desc">
          Three integrated ML models CNN and LSTM Forecasting, Battery Degradation LSTM,
          and SHAP Explainable XGBoost Dispatch working together to deliver
          transparent, trustworthy energy management for rural India.
        </p>
        <div className="landing-hero-actions">
          <button className="btn-primary !py-3 !px-8 !text-[14px]" onClick={() => setShowAuth(true)}>
            Open Dashboard <ArrowRight className="w-4 h-4" />
          </button>
          <a href="#features" className="btn-ghost !py-3 !px-6 !text-[13px]">
            Explore Features <ChevronRight className="w-4 h-4" />
          </a>
        </div>

        {/* Stats strip */}
        <div className="landing-stats">
          {stats.map((s, i) => {
            const I = s.icon;
            return (
              <div key={i} className="landing-stat-item">
                <I className="w-4 h-4" style={{ color: 'var(--brand)' }} />
                <span className="landing-stat-value">{s.value}</span>
                <span className="landing-stat-label">{s.label}</span>
              </div>
            );
          })}
        </div>
      </section>

      {/* ── Features ── */}
      <section id="features" className="landing-features">
        <div className="landing-section-header">
          <h2 className="landing-section-title">Powered by Three ML Models</h2>
          <p className="landing-section-sub">Each model feeds real predictions into a unified dashboard with full transparency</p>
        </div>
        <div className="landing-features-grid">
          {features.map((f, i) => {
            const I = f.icon;
            return (
              <div key={i} className="landing-feature-card glass">
                <div className="landing-feature-icon" style={{ background: f.bg }}>
                  <I className="w-5 h-5" style={{ color: f.color }} />
                </div>
                <h3 className="landing-feature-title">{f.title}</h3>
                <p className="landing-feature-desc">{f.desc}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* ── Tech Stack ── */}
      <section id="tech" className="landing-tech">
        <div className="landing-section-header">
          <h2 className="landing-section-title">Technology Stack</h2>
          <p className="landing-section-sub">Built with production grade open source ML and web technologies</p>
        </div>
        <div className="landing-tech-badges">
          {techStack.map((t, i) => (
            <span key={i} className="landing-tech-badge" style={{ borderColor: `${t.color}30`, color: t.color }}>
              <span className="landing-tech-dot" style={{ background: t.color }} />
              {t.name}
            </span>
          ))}
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="landing-cta">
        <div className="landing-cta-card glass glow-brand">
          <h2 className="landing-cta-title">Ready to explore the dashboard?</h2>
          <p className="landing-cta-desc">Sign in to access real time forecasts, battery monitoring, and AI-powered dispatch decisions.</p>
          <button className="btn-primary !py-3 !px-8 !text-[14px]" onClick={() => setShowAuth(true)}>
            Sign In <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="landing-footer">
        <p>© 2026 XplainableAI · Zeroday Explorers · Bihar Rural Microgrid Research</p>
      </footer>

      {/* ══════════════ AUTH MODAL ══════════════ */}
      {showAuth && (
        <div className="auth-overlay" onClick={() => setShowAuth(false)}>
          <div className="auth-modal glass" onClick={(e) => e.stopPropagation()}>
            <button className="auth-close" onClick={() => setShowAuth(false)}>
              <X className="w-4 h-4" />
            </button>

            <div className="auth-header">
              <div className="landing-logo-icon" style={{ width: 40, height: 40 }}>
                <Zap className="w-5 h-5" style={{ color: 'white' }} />
              </div>
              <h2 className="auth-title">{isLogin ? 'Welcome back' : 'Create account'}</h2>
              <p className="auth-subtitle">
                {isLogin ? 'Sign in to access your microgrid dashboard' : 'Start monitoring your microgrid today'}
              </p>
            </div>

            {/* Tabs */}
            <div className="auth-tabs">
              <button className={`auth-tab ${isLogin ? 'active' : ''}`} onClick={() => { setIsLogin(true); setError(''); setSuccess(''); }}>
                Sign In
              </button>
              <button className={`auth-tab ${!isLogin ? 'active' : ''}`} onClick={() => { setIsLogin(false); setError(''); setSuccess(''); }}>
                Sign Up
              </button>
            </div>

            {/* Quick Demo Access */}
            <div style={{ marginBottom: '14px' }}>
              <button
                type="button"
                onClick={handleDemoLogin}
                className="btn-ghost w-full !py-2.5 !text-[13px] flex items-center justify-center gap-2"
                style={{ borderColor: 'var(--brand)', color: 'var(--brand)', background: 'rgba(229,136,10,0.06)' }}
                disabled={loading}
              >
                <Zap className="w-4 h-4" />
                <span>Quick Demo Access (1-Click)</span>
              </button>
              <div style={{ display: 'flex', alignItems: 'center', margin: '10px 0', gap: '8px' }}>
                <div style={{ flex: 1, height: '1px', background: 'rgba(255,255,255,0.08)' }} />
                <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>or with PostgreSQL credentials</span>
                <div style={{ flex: 1, height: '1px', background: 'rgba(255,255,255,0.08)' }} />
              </div>
            </div>

            <form onSubmit={handleSubmit} className="auth-form">
              {error && (
                <div className="auth-error">
                  <X className="w-3.5 h-3.5 shrink-0" />
                  {error}
                </div>
              )}
              {success && (
                <div className="auth-success">
                  <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
                  {success}
                </div>
              )}

              <div className="auth-field">
                <label className="auth-label">Email</label>
                <div className="auth-input-wrap">
                  <Mail className="w-4 h-4 auth-input-icon" />
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="you@example.com"
                    className="auth-input"
                    required
                  />
                </div>
              </div>

              <div className="auth-field">
                <label className="auth-label">Password</label>
                <div className="auth-input-wrap">
                  <Lock className="w-4 h-4 auth-input-icon" />
                  <input
                    type={showPwd ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder={isLogin ? '••••••••' : 'Min. 6 characters'}
                    className="auth-input"
                    required
                    minLength={6}
                  />
                  <button type="button" className="auth-pwd-toggle" onClick={() => setShowPwd(!showPwd)}>
                    {showPwd ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>

              <button type="submit" className="btn-primary w-full !py-3 !text-[13px]" disabled={loading}>
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
                {loading ? 'Please wait...' : isLogin ? 'Sign In' : 'Create Account'}
              </button>
            </form>

            <p className="auth-switch">
              {isLogin ? "Don't have an account?" : 'Already have an account?'}{' '}
              <button onClick={() => { setIsLogin(!isLogin); setError(''); setSuccess(''); }}>
                {isLogin ? 'Sign Up' : 'Sign In'}
              </button>
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
