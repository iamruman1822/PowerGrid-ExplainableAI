import { useLocation } from 'react-router-dom';
import { Bell, Search } from 'lucide-react';
import { NAV_ITEMS } from '../utils/constants';

export default function TopBar() {
  const location = useLocation();
  const current = NAV_ITEMS.find(item => item.path === location.pathname) || NAV_ITEMS[0];

  return (
    <header className="flex items-center justify-between gap-4 px-5 sm:px-8 h-16 shrink-0" style={{ borderBottom: '1px solid var(--border-subtle)' }}>
      {/* Page title only — no breadcrumb */}
      <h2 className="text-[18px] font-bold uppercase tracking-wide truncate min-w-0" style={{ fontFamily: 'var(--font-heading)', color: 'var(--text-heading)' }}>
        {current.label}
      </h2>

      {/* Right Actions */}
      <div className="flex items-center gap-2.5 shrink-0">
        <div className="relative hidden md:block">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 pointer-events-none" style={{ color: 'var(--text-muted)' }} />
          <input type="text" placeholder="Search..." className="input-field w-52 lg:w-56 !pl-10 !pr-4 !py-[7px] text-[12px]" />
        </div>
        <button className="relative w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'var(--bg-inset)', border: '1px solid var(--border-card)' }}>
          <Bell className="w-3.5 h-3.5" style={{ color: 'var(--text-secondary)' }} />
          <span className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full" style={{ background: 'var(--color-danger)', border: '2px solid var(--bg-base)' }} />
        </button>
        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg" style={{ background: 'rgba(34,197,94,0.06)', border: '1px solid rgba(34,197,94,0.15)' }}>
          <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: 'var(--color-success)' }} />
          <span className="text-[11px] font-semibold" style={{ color: '#16a34a' }}>Live</span>
        </div>
      </div>
    </header>
  );
}
