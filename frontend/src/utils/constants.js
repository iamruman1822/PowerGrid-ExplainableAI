export const API_BASE = '/api';

export const CLASS_NAMES = ['CONSERVATIVE', 'MODERATE', 'AGGRESSIVE'];

export const CLASS_COLORS = {
  CONSERVATIVE: { main: '#2563eb', bg: 'rgba(37, 99, 235, 0.10)', border: 'rgba(37, 99, 235, 0.25)', text: '#2563eb' },
  MODERATE:     { main: '#059669', bg: 'rgba(5, 150, 105, 0.10)', border: 'rgba(5, 150, 105, 0.25)', text: '#059669' },
  AGGRESSIVE:   { main: '#d97706', bg: 'rgba(217, 119, 6, 0.10)', border: 'rgba(217, 119, 6, 0.25)', text: '#d97706' },
};

export const CLASS_ICONS = {
  CONSERVATIVE: '🛡️',
  MODERATE:     '⚖️',
  AGGRESSIVE:   '🚀',
};

export const CHART_COLORS = {
  solar:  { stroke: '#d97706', fill: 'rgba(217, 119, 6, 0.12)' },
  wind:   { stroke: '#0891b2', fill: 'rgba(8, 145, 178, 0.12)' },
  load:   { stroke: '#7c3aed', fill: 'rgba(124, 58, 237, 0.12)' },
  actual: { stroke: '#2563eb', fill: 'rgba(37, 99, 235, 0.12)' },
  pred:   { stroke: '#e11d48', fill: 'rgba(225, 29, 72, 0.12)' },
};

export const NAV_ITEMS = [
  { id: 'dashboard',    label: 'Dashboard',        path: '/',               icon: 'LayoutDashboard' },
  { id: 'forecast',     label: 'Energy Forecast',  path: '/forecast',       icon: 'Sun' },
  { id: 'battery',      label: 'Battery Health',   path: '/battery',        icon: 'Battery' },
  { id: 'dispatch',     label: 'Dispatch Engine',  path: '/dispatch',       icon: 'Zap' },
  { id: 'explain',      label: 'Explainability',   path: '/explainability', icon: 'BrainCircuit' },
  { id: 'scenarios',    label: 'Scenarios',         path: '/scenarios',      icon: 'MapPin' },
  { id: 'settings',     label: 'Settings',          path: '/settings',       icon: 'Settings' },
];
