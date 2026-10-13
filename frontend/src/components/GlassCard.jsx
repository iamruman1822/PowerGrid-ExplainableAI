export default function GlassCard({ children, className = '', glow = '', hover = true, ...props }) {
  const glowClass = glow ? `glow-${glow}` : '';
  return (
    <div className={`glass ${glowClass} p-6 ${className}`} {...props}>
      {children}
    </div>
  );
}
