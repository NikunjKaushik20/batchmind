import { NavLink, useLocation } from 'react-router-dom'
import {
    LayoutDashboard, FlaskConical, Zap, Award, Leaf, X
} from 'lucide-react'

const NAV = [
    { to: '/', label: 'Dashboard', icon: LayoutDashboard },
    { to: '/batch', label: 'Batch Explorer', icon: FlaskConical },
    { to: '/optimizer', label: 'Causal Optimizer', icon: Zap },
    { to: '/golden', label: 'Golden Signature', icon: Award },
    { to: '/carbon', label: 'Carbon Ledger', icon: Leaf },
]

export default function Sidebar({ open, onClose }) {
    return (
        <>
            <div className={`sidebar-overlay ${open ? 'visible' : ''}`} onClick={onClose} />
            <aside className={`sidebar ${open ? 'open' : ''}`}>
                <div className="sidebar-logo">
                    <h1>⚡ BatchMind</h1>
                    <span>AI Manufacturing Intelligence</span>
                </div>
                <nav className="sidebar-nav">
                    <div className="nav-section-label">Platform</div>
                    {NAV.map(({ to, label, icon: Icon }) => (
                        <NavLink
                            key={to}
                            to={to}
                            end={to === '/'}
                            className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
                            onClick={onClose}
                        >
                            <Icon className="nav-icon" size={18} />
                            {label}
                        </NavLink>
                    ))}
                </nav>
                <div className="sidebar-footer">
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
                        <div style={{ fontWeight: 700, color: 'var(--text-secondary)', marginBottom: 4 }}>BatchMind v1.0</div>
                        <br />

                    </div>
                </div>
            </aside>
        </>
    )
}
