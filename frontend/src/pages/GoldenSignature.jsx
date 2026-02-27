import { useEffect, useState } from 'react'
import { api } from '../api'
import {
    RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer, Tooltip,
    ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Cell
} from 'recharts'

const PRESET_LABELS = {
    best_quality: '🏆 Best Quality',
    best_energy: '⚡ Best Energy',
    balanced: '⚖️ Balanced',
    sustainability: '🌱 Sustainability',
}

export default function GoldenSignature() {
    const [signatures, setSignatures] = useState([])
    const [pareto, setPareto] = useState([])
    const [selected, setSelected] = useState(null)
    const [loading, setLoading] = useState(true)
    const [creating, setCreating] = useState(false)
    const [form, setForm] = useState({ quality: 0.4, yield_score: 0.3, energy: 0.2, performance: 0.1, label: '' })
    const [updateMsg, setUpdateMsg] = useState('')

    useEffect(() => {
        Promise.all([api.getSignatures(), api.getPareto()])
            .then(([s, p]) => { setSignatures(s); setPareto(p); setSelected(s[0]) })
            .catch(console.error)
            .finally(() => setLoading(false))
    }, [])

    const handleCreate = async () => {
        setCreating(true)
        try {
            const sig = await api.createSignature(form)
            setSignatures(prev => [...prev, sig])
            setSelected(sig)
        } catch (e) { console.error(e) }
        setCreating(false)
    }

    const handleUpdate = async (sigId, batchId) => {
        const res = await api.updateSignature({ sig_id: sigId, new_batch_id: batchId })
        setUpdateMsg(res.updated ? `✅ Signature updated! +${res.improvement_pct}% improvement` : `ℹ️ ${res.reason}`)
        setTimeout(() => setUpdateMsg(''), 3000)
    }

    // Radar chart data from selected signature params
    const radarData = selected ? Object.entries(selected.parameters || {}).map(([key, val]) => ({
        subject: key.replace(/_/g, ' '),
        value: val.value,
        fullMark: Math.max(val.value * 1.5, 20)
    })) : []

    if (loading) return <div className="loading"><div className="spinner" /></div>

    return (
        <div className="page fade-in">
            <div className="page-header">
                <h2>Golden Signature</h2>
                <p>Bayesian optimal parameter management with confidence intervals</p>
            </div>

            <div className="grid-12" style={{ marginBottom: 20 }}>
                {/* Left: Signature List */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                    <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
                        <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border)' }}>
                            <div className="section-title" style={{ margin: 0 }}>Saved Signatures</div>
                        </div>
                        {signatures.map((sig, i) => (
                            <div key={i}
                                onClick={() => setSelected(sig)}
                                style={{
                                    padding: '14px 20px',
                                    cursor: 'pointer',
                                    borderBottom: '1px solid var(--border)',
                                    background: selected?.id === sig.id ? 'var(--accent-glow)' : 'transparent',
                                    transition: 'background 0.15s',
                                }}
                            >
                                <div style={{ fontWeight: 600, fontSize: 14, color: selected?.id === sig.id ? 'var(--accent)' : 'var(--text)' }}>
                                    {PRESET_LABELS[sig.id] || sig.label || sig.id}
                                </div>
                                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 3 }}>
                                    Ref: {sig.reference_batch} · {sig.n_supporting_batches} batches
                                </div>
                                <div style={{ display: 'flex', gap: 6, marginTop: 6, flexWrap: 'wrap' }}>
                                    {Object.entries(sig.objectives || {}).map(([k, v]) => v > 0 && (
                                        <span key={k} className="badge badge-muted">{k}: {(v * 100).toFixed(0)}%</span>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Create New */}
                    <div className="card">
                        <div className="section-title">Create Custom Signature</div>
                        {['quality', 'yield_score', 'energy', 'performance'].map(k => (
                            <div className="slider-group" key={k} style={{ marginBottom: 12 }}>
                                <div className="slider-label">
                                    <span style={{ textTransform: 'capitalize' }}>{k.replace('_', ' ')}</span>
                                    <strong>{(form[k] * 100).toFixed(0)}%</strong>
                                </div>
                                <input type="range" min={0} max={1} step={0.05} value={form[k]}
                                    onChange={e => setForm(f => ({ ...f, [k]: parseFloat(e.target.value) }))} />
                            </div>
                        ))}
                        <input className="form-input" placeholder="Label (optional)"
                            value={form.label} onChange={e => setForm(f => ({ ...f, label: e.target.value }))}
                            style={{ marginBottom: 12 }} />
                        <button className="btn btn-primary" style={{ width: '100%' }}
                            onClick={handleCreate} disabled={creating}>
                            {creating ? 'Computing...' : '⚡ Generate Signature'}
                        </button>
                    </div>
                </div>

                {/* Right: Selected Signature Detail */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                    {selected && (
                        <>
                            {/* Header */}
                            <div className="card">
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 12 }}>
                                    <div>
                                        <div style={{ fontSize: 22, fontWeight: 800, color: 'var(--text)', marginBottom: 4 }}>
                                            {PRESET_LABELS[selected.id] || selected.label || selected.id}
                                        </div>
                                        <span className={`badge ${selected.confidence === 'HIGH' ? 'badge-success' : 'badge-warning'}`}>
                                            {selected.confidence} CONFIDENCE
                                        </span>
                                        <span className="badge badge-muted" style={{ marginLeft: 8 }}>
                                            Ref: {selected.reference_batch}
                                        </span>
                                    </div>
                                    <button className="btn btn-ghost btn-sm"
                                        onClick={() => handleUpdate(selected.id, selected.reference_batch)}>
                                        🔄 Check for Update
                                    </button>
                                </div>
                                {updateMsg && (
                                    <div style={{ marginTop: 12, padding: '10px 14px', background: 'var(--success-bg)', borderRadius: 8, fontSize: 13, color: 'var(--success)' }}>
                                        {updateMsg}
                                    </div>
                                )}
                            </div>

                            {/* Recommended Parameters */}
                            <div className="card">
                                <div className="section-title">Recommended Parameters <small>with 95% confidence intervals</small></div>
                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 10 }}>
                                    {Object.entries(selected.parameters || {}).map(([param, val]) => (
                                        <div key={param} className="param-card">
                                            <div className="param-name">{param.replace(/_/g, ' ')}</div>
                                            <div className="param-value">{val.value?.toFixed(2)}</div>
                                            <div className="param-ci">[{val.ci_low?.toFixed(2)}, {val.ci_high?.toFixed(2)}]</div>
                                            <div className="param-bar">
                                                <div className="param-bar-fill" style={{
                                                    left: `${((val.ci_low) / (val.ci_high * 1.1)) * 100}%`,
                                                    width: `${((val.ci_high - val.ci_low) / (val.ci_high * 1.1)) * 100 + 20}%`,
                                                }} />
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* Expected Outcomes */}
                            <div className="card">
                                <div className="section-title">Expected Outcomes</div>
                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: 10 }}>
                                    {Object.entries(selected.expected_outcomes || {}).slice(0, 6).map(([metric, val]) => (
                                        <div key={metric} className="param-card">
                                            <div className="param-name">{metric.replace(/_/g, ' ')}</div>
                                            <div className="param-value" style={{ fontSize: 16 }}>{val.mean?.toFixed(2)}</div>
                                            <div className="param-ci">CI: [{val.ci_low?.toFixed(2)}, {val.ci_high?.toFixed(2)}]</div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* Pareto scatter */}
                            <div className="card">
                                <div className="section-title">Pareto Frontier — Energy vs Quality</div>
                                <div className="chart-wrapper">
                                    <ResponsiveContainer width="100%" height={260}>
                                        <ScatterChart margin={{ top: 10, right: 20, left: -10, bottom: 20 }}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                                            <XAxis dataKey="energy_kwh" name="Energy (kWh)" tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                                                label={{ value: 'Energy (kWh)', position: 'insideBottom', offset: -10, fill: 'var(--text-muted)', fontSize: 11 }} />
                                            <YAxis dataKey="quality_score" name="Quality" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                                            <Tooltip
                                                content={({ payload }) => {
                                                    if (!payload?.length) return null
                                                    const d = payload[0].payload
                                                    return (
                                                        <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, padding: '10px 14px', fontSize: 12 }}>
                                                            <div style={{ fontWeight: 700, color: 'var(--accent)' }}>{d.batch_id}</div>
                                                            <div>Energy: {d.energy_kwh} kWh</div>
                                                            <div>Quality: {d.quality_score}</div>
                                                        </div>
                                                    )
                                                }}
                                            />
                                            <Scatter data={pareto} fill="var(--accent)" opacity={0.7} r={5} />
                                            {/* Highlight reference batch */}
                                            <Scatter
                                                data={pareto.filter(d => d.batch_id === selected.reference_batch)}
                                                fill="var(--warning)" r={10} opacity={1}
                                            />
                                        </ScatterChart>
                                    </ResponsiveContainer>
                                    <div style={{ fontSize: 11, color: 'var(--text-muted)', textAlign: 'center', marginTop: 4 }}>
                                        🟡 Reference batch ({selected.reference_batch}) · 🟣 All other batches
                                    </div>
                                </div>
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    )
}
