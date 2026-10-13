import { NavLink, useLocation, useNavigate } from 'react-router-dom';
import {
  LayoutDashboard, Sun, Battery, Zap, BrainCircuit,
  MapPin, Settings, ChevronLeft, ChevronRight, Workflow, LogOut
} from 'lucide-react';
import { NAV_ITEMS } from '../utils/constants';
import { useAuth } from '../context/AuthProvider';

const ICON_MAP = { LayoutDashboard, Sun, Battery, Zap, BrainCircuit, MapPin, Settings };

export default function Sidebar({ collapsed, onToggle }) {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, signOut } = useAuth();

  const handleSignOut = async () => {
    await signOut();
    navigate('/landing');
  };

  return (
    <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
      {/* Brand — clean Lucide icon + text */}
      <div className="flex items-center gap-3 px-5 h-16 shrink-0" style={{ borderBottom: '1px solid var(--border-subtle)' }}>
        <div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0" style={{ background: 'var(--brand)' }}>
          <Workflow className="w-4 h-4 text-white" />
        </div>
        {!collapsed && (
          <div>
            <div className="text-[15px] font-extrabold tracking-tight" style={{ fontFamily: 'var(--font-heading)', color: 'var(--text-heading)' }}>XplainableAI</div>
            <div className="text-[10px] font-semibold tracking-[0.08em] uppercase" style={{ color: 'var(--text-muted)' }}>Microgrid Dashboard</div>
          </div>
        )}
      </div>

      {/* Nav — no "Navigation" label */}
      <nav className="flex-1 py-4 overflow-y-auto">
        {NAV_ITEMS.map(item => {
          const Icon = ICON_MAP[item.icon] || LayoutDashboard;
          const isActive = location.pathname === item.path;
          return (
            <NavLink key={item.id} to={item.path} className={`sidebar-nav-item ${isActive ? 'active' : ''}`} title={collapsed ? item.label : undefined}>
              <Icon className="w-[20px] h-[20px] shrink-0" />
              {!collapsed && <span>{item.label}</span>}
            </NavLink>
          );
        })}
      </nav>

      {/* User + Sign out + Collapse */}
      <div className="px-3 pb-3 space-y-2" style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '12px' }}>
        {!collapsed && user && (
          <div className="px-2 mb-1">
            <p className="text-[10px] font-semibold truncate" style={{ color: 'var(--text-muted)' }}>{user.email}</p>
          </div>
        )}
        <button onClick={handleSignOut} className="btn-ghost w-full justify-center" title={collapsed ? 'Sign Out' : undefined}
          style={{ color: 'var(--color-danger)', borderColor: 'rgba(239,68,68,0.15)' }}>
          <LogOut className="w-4 h-4" />
          {!collapsed && <span>Sign Out</span>}
        </button>
        <button onClick={onToggle} className="btn-ghost w-full justify-center">
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <><ChevronLeft className="w-4 h-4" /><span>Collapse</span></>}
        </button>
      </div>
    </aside>
  );
}

