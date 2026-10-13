import { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './context/AuthProvider';
import Sidebar from './components/Sidebar';
import TopBar from './components/TopBar';
import Landing from './pages/Landing';
import Dashboard from './pages/Dashboard';
import EnergyForecast from './pages/EnergyForecast';
import BatteryHealth from './pages/BatteryHealth';
import DispatchEngine from './pages/DispatchEngine';
import Explainability from './pages/Explainability';
import Scenarios from './pages/Scenarios';
import Settings from './pages/Settings';

/**
 * ProtectedRoute — redirects to /landing if user is not logged in.
 */
function ProtectedRoute({ children }) {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100vh', fontFamily: 'var(--font-heading)', color: 'var(--text-muted)',
        fontSize: '14px', gap: '10px',
      }}>
        <div className="w-5 h-5 border-2 border-t-transparent rounded-full animate-spin" style={{ borderColor: 'var(--brand)', borderTopColor: 'transparent' }} />
        Loading...
      </div>
    );
  }

  return user ? children : <Navigate to="/landing" replace />;
}

/**
 * DashboardLayout — the sidebar + topbar chrome for authenticated pages.
 */
function DashboardLayout({ children }) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  return (
    <>
      <div className="bg-mesh" />
      <Sidebar
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(prev => !prev)}
      />
      <div className={`main-content ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
        <TopBar />
        <main className="px-6 py-6 lg:px-8">
          {children}
        </main>
      </div>
    </>
  );
}

export default function App() {
  const { user, loading } = useAuth();

  return (
    <Router>
      <Routes>
        {/* Public landing page */}
        <Route
          path="/landing"
          element={user && !loading ? <Navigate to="/" replace /> : <Landing />}
        />

        {/* Protected dashboard routes */}
        <Route path="/" element={
          <ProtectedRoute>
            <DashboardLayout><Dashboard /></DashboardLayout>
          </ProtectedRoute>
        } />
        <Route path="/forecast" element={
          <ProtectedRoute>
            <DashboardLayout><EnergyForecast /></DashboardLayout>
          </ProtectedRoute>
        } />
        <Route path="/battery" element={
          <ProtectedRoute>
            <DashboardLayout><BatteryHealth /></DashboardLayout>
          </ProtectedRoute>
        } />
        <Route path="/dispatch" element={
          <ProtectedRoute>
            <DashboardLayout><DispatchEngine /></DashboardLayout>
          </ProtectedRoute>
        } />
        <Route path="/explainability" element={
          <ProtectedRoute>
            <DashboardLayout><Explainability /></DashboardLayout>
          </ProtectedRoute>
        } />
        <Route path="/scenarios" element={
          <ProtectedRoute>
            <DashboardLayout><Scenarios /></DashboardLayout>
          </ProtectedRoute>
        } />
        <Route path="/settings" element={
          <ProtectedRoute>
            <DashboardLayout><Settings /></DashboardLayout>
          </ProtectedRoute>
        } />

        {/* Fallback */}
        <Route path="*" element={<Navigate to={user ? "/" : "/landing"} replace />} />
      </Routes>
    </Router>
  );
}
