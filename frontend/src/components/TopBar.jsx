import { useLocation } from 'react-router-dom'
import { useTheme } from '../context/ThemeContext'
import { Sun, Moon, Menu } from 'lucide-react'

const PAGE_TITLES = {
    '/': { title: 'Dashboard', sub: 'Manufacturing Intelligence Overview' },
    '/batch': { title: 'Batch Explorer', sub: 'Time-series analysis & energy fingerprinting' },
    '/optimizer': { title: 'Causal Optimizer', sub: 'Counterfactual reasoning & recommendations' },
    '/golden': { title: 'Golden Signature', sub: 'Bayesian optimal parameter management' },
    '/carbon': { title: 'Carbon Ledger', sub: 'Per-batch carbon accounting & forecasting' },
}

export default function TopBar({ onMenuClick }) {
    const { theme, toggle } = useTheme()
    const loc = useLocation()
    const info = PAGE_TITLES[loc.pathname] || { title: 'BatchMind', sub: '' }

    return (
        <header className="topbar">
            <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                <button className="hamburger" onClick={onMenuClick}>
                    <Menu size={22} />
                </button>
                <div className="topbar-left">
                    <h2>{info.title}</h2>
                    <p>{info.sub}</p>
                </div>
            </div>
            <div className="topbar-right">
                <button className="theme-toggle" onClick={toggle}>
                    {theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}
                    {theme === 'dark' ? 'Light' : 'Dark'}
                </button>
            </div>
        </header>
    )
}
