import { useEffect, useState } from 'react'
import { api } from '../api'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { Sparkles, RefreshCw, Info } from 'lucide-react'

const OBJ_LABELS = { quality: '🏆 Quality', yield_score: '📈 Yield', energy: '⚡ Energy', performance: '🚀 Performance' }

const INPUT_PARAMS = [
    'Granulation_Time', 'Binder_Amount', 'Drying_Temp', 'Drying_Time',
    'Compression_Force', 'Machine_Speed', 'Lubricant_Conc'
]

const DAG_INPUTS = ['Granulation_Time', 'Binder_Amount', 'Drying_Temp', 'Drying_Time', 'Compression_Force', 'Machine_Speed', 'Lubricant_Conc']
const DAG_MID = ['Moisture_Content', 'Tablet_Weight', 'Power_Consumption']
const DAG_OUT = ['Hardness', 'Friability', 'Disintegration_Time', 'Dissolution_Rate', 'Content_Uniformity', 'Carbon_Emissions']

export default function CausalOptimizer() {
    const [objectives, setObjectives] = useState({ quality: 0.4, yield_score: 0.3, energy: 0.2, performance: 0.1 })
    const [result, setResult] = useState(null)
    const [cf, setCf] = useState(null)
    const [explanation, setExplanation] = useState(null)
    const [importance, setImportance] = useState(null)
    const [loading, setLoading] = useState(false)
    const [cfId, setCfId] = useState('T001')
    const [cfLoading, setCfLoading] = useState(false)
    const [explainLoading, setExplainLoading] = useState(false)

    useEffect(() => {
        api.getImportance().then(setImportance).catch(console.error)
    }, [])

    const handleOptimize = async () => {
        setLoading(true)
        try {
            const res = await api.optimize(objectives)
            setResult(res)
        } catch (e) { console.error(e) }
        setLoading(false)
    }

    const handleCounterfactual = async () => {
        setCfLoading(true)
        try {
            const res = await api.counterfactual(cfId)
            setCf(res)
            setExplanation(null)
        } catch (e) { console.error(e) }
        setCfLoading(false)
    }

    const handleExplain = async () => {
        setExplainLoading(true)
        try {
            const res = await api.explain(cfId)
            setExplanation(res.explanation)
        } catch (e) { console.error(e) }
        setExplainLoading(false)
    }

    const totalWeight = Object.values(objectives).reduce((s, v) => s + v, 0)

    // Feature importance for Quality_Score
    const importanceData = importance?.Quality_Score
        ? Object.entries(importance.Quality_Score).map(([k, v]) => ({ name: k.replace(/_/g, ' '), value: v }))
            .sort((a, b) => b.value - a.value)
        : []

    return (
        <div className="page fade-in">
            <div className="page-header">
                <h2>Causal Optimizer</h2>
                <p>Causal reasoning & counterfactual interventions — not just correlation</p>
            </div>

            <div className="grid-21" style={{ marginBottom: 20 }}>
                {/* Left: Controls */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                    {/* Objective Sliders */}
                    <div className="card">
                        <div className="section-title">Define Objectives</div>
                        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 16 }}>
                            Set relative importance of each target. BatchMind finds the causal intervention to achieve your goals.
                        </p>
                        {Object.entries(objectives).map(([key, val]) => (
                            <div className="slider-group" key={key} style={{ marginBottom: 16 }}>
                                <div className="slider-label">
                                    <span>{OBJ_LABELS[key]}</span>
                                    <strong>{(val * 100).toFixed(0)}%</strong>
                                </div>
                                <input type="range" min={0} max={1} step={0.05} value={val}
                                    onChange={e => setObjectives(o => ({ ...o, [key]: parseFloat(e.target.value) }))} />
                            </div>
                        ))}
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 14, padding: '8px 12px', background: 'var(--surface-2)', borderRadius: 8 }}>
                            ℹ️ Total weight: {(totalWeight * 100).toFixed(0)}% (auto-normalized)
                        </div>
                        <button className="btn btn-primary" style={{ width: '100%', justifyContent: 'center' }}
                            onClick={handleOptimize} disabled={loading}>
                            {loading ? <><RefreshCw size={14} className="pulse" /> Computing...</> : <><Sparkles size={14} /> Run Causal Optimization</>}
                        </button>
                    </div>

                    {/* Counterfactual Input */}
                    <div className="card">
                        <div className="section-title">Counterfactual Analysis</div>
                        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12 }}>
                            Select any batch to see what parameters would have saved energy while maintaining quality.
                        </p>
                        <select className="form-select" value={cfId} onChange={e => setCfId(e.target.value)}
                            style={{ marginBottom: 10 }}>
                            {Array.from({ length: 60 }, (_, i) => `T${String(i + 1).padStart(3, '0')}`).map(id =>
                                <option key={id} value={id}>{id}</option>
                            )}
                        </select>
                        <button className="btn btn-secondary" style={{ width: '100%', justifyContent: 'center' }}
                            onClick={handleCounterfactual} disabled={cfLoading}>
                            {cfLoading ? 'Analysing...' : '🔍 Analyse Counterfactual'}
                        </button>
                    </div>
                </div>

                {/* Right: Results */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                    {/* Optimization Result */}
                    {result && (
                        <div className="card">
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
                                <div className="section-title" style={{ margin: 0 }}>
                                    Causal Recommendation
                                    <span className={`badge ${result.confidence === 'HIGH' ? 'badge-success' : 'badge-warning'}`} style={{ marginLeft: 10 }}>
                                        {result.confidence}
                                    </span>
                                </div>
                                <span className="badge badge-muted">Based on: {result.recommended_batch}</span>
                            </div>

                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: 10, marginBottom: 20 }}>
                                {Object.entries(result.recommendations || {}).map(([param, val]) => (
                                    <div key={param} className="param-card">
                                        <div className="param-name">{param.replace(/_/g, ' ')}</div>
                                        <div className="param-value">{val.value?.toFixed(2)}</div>
                                        <div className="param-ci">CI: [{val.ci_low?.toFixed(2)}, {val.ci_high?.toFixed(2)}]</div>
                                    </div>
                                ))}
                            </div>

                            <div className="section-title">Predicted Outcomes</div>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 10 }}>
                                {Object.entries(result.predicted_outcomes || {}).slice(0, 8).map(([key, val]) => (
                                    <div key={key} style={{ padding: '10px 14px', background: 'var(--surface-2)', borderRadius: 8 }}>
                                        <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 3 }}>
                                            {key.replace(/_/g, ' ')}
                                        </div>
                                        <div style={{ fontSize: 18, fontWeight: 800, color: 'var(--teal)' }}>{typeof val === 'number' ? val.toFixed(2) : val}</div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Counterfactual Result */}
                    {cf && (
                        <div className="card">
                            <div className="section-title">⚗️ Counterfactual: Batch {cf.batch_id}</div>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 16 }}>
                                <div style={{ padding: '14px', background: 'var(--danger-bg)', borderRadius: 10, textAlign: 'center' }}>
                                    <div style={{ fontSize: 10, color: 'var(--danger)', fontWeight: 700, textTransform: 'uppercase', marginBottom: 4 }}>Actual Energy</div>
                                    <div style={{ fontSize: 22, fontWeight: 800, color: 'var(--danger)' }}>{cf.actual_energy_kwh} kWh</div>
                                </div>
                                <div style={{ padding: '14px', background: 'var(--success-bg)', borderRadius: 10, textAlign: 'center' }}>
                                    <div style={{ fontSize: 10, color: 'var(--success)', fontWeight: 700, textTransform: 'uppercase', marginBottom: 4 }}>Optimal Energy</div>
                                    <div style={{ fontSize: 22, fontWeight: 800, color: 'var(--success)' }}>{cf.counterfactual_energy_kwh} kWh</div>
                                </div>
                                <div style={{ padding: '14px', background: 'var(--teal-bg)', borderRadius: 10, textAlign: 'center' }}>
                                    <div style={{ fontSize: 10, color: 'var(--teal)', fontWeight: 700, textTransform: 'uppercase', marginBottom: 4 }}>Energy Saved</div>
                                    <div style={{ fontSize: 22, fontWeight: 800, color: 'var(--teal)' }}>{cf.pct_energy_saved}%</div>
                                </div>
                            </div>

                            <div style={{ padding: '12px 16px', background: 'var(--success-bg)', borderRadius: 8, marginBottom: 16, fontSize: 13 }}>
                                💚 Would have saved <strong>{cf.energy_saved_kwh} kWh</strong> and <strong>{cf.carbon_saved_kg} kg CO₂e</strong>
                                — using parameters from batch <strong>{cf.reference_batch}</strong>
                            </div>

                            {Object.keys(cf.parameter_changes || {}).length > 0 && (
                                <>
                                    <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 10 }}>
                                        Causal Interventions Required:
                                    </div>
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                                        {Object.entries(cf.parameter_changes).map(([param, change]) => (
                                            <div key={param} style={{
                                                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                                                padding: '10px 14px', background: 'var(--surface-2)', borderRadius: 8
                                            }}>
                                                <span style={{ fontSize: 13, color: 'var(--text-secondary)', fontWeight: 500 }}>
                                                    {param.replace(/_/g, ' ')}
                                                </span>
                                                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                                    <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>{change.actual}</span>
                                                    <span style={{ color: 'var(--text-muted)' }}>→</span>
                                                    <span style={{ fontSize: 13, fontWeight: 700, color: change.direction === 'increase' ? 'var(--success)' : 'var(--warning)' }}>
                                                        {change.counterfactual}
                                                    </span>
                                                    <span className={`badge ${change.direction === 'increase' ? 'badge-success' : 'badge-warning'}`}>
                                                        {change.direction === 'increase' ? '↑' : '↓'} {Math.abs(change.change).toFixed(2)}
                                                    </span>
                                                </div>
                                            </div>
                                        ))}
                                    </div>

                                    <button className="btn btn-primary btn-sm" style={{ marginTop: 14 }}
                                        onClick={handleExplain} disabled={explainLoading}>
                                        {explainLoading ? '⏳ Generating...' : '🤖 Get AI Explanation'}
                                    </button>
                                </>
                            )}

                            {explanation && (
                                <div className="insight-panel" style={{ marginTop: 14 }}>
                                    <div className="insight-label"><Sparkles size={12} /> AI CAUSAL EXPLANATION</div>
                                    <p>{explanation}</p>
                                </div>
                            )}
                        </div>
                    )}

                    {/* Feature Importance */}
                    {importanceData.length > 0 && (
                        <div className="card">
                            <div className="section-title">Feature Importance for Quality Score</div>
                            <div className="chart-wrapper">
                                <ResponsiveContainer width="100%" height={220}>
                                    <BarChart data={importanceData} layout="vertical" margin={{ top: 5, right: 20, left: 80, bottom: 5 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                                        <XAxis type="number" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                                        <YAxis dataKey="name" type="category" tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} width={80} />
                                        <Tooltip contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }} />
                                        <Bar dataKey="value" name="Importance %" radius={[0, 4, 4, 0]}>
                                            {importanceData.map((_, i) => (
                                                <Cell key={i} fill={`hsl(${245 + i * 15}, 70%, 65%)`} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    )}

                    {/* Causal DAG */}
                    <div className="card">
                        <div className="section-title">Causal Graph — Manufacturing DAG</div>
                        <div style={{ marginBottom: 10, fontSize: 12, color: 'var(--text-muted)' }}>
                            Directed edges represent causal (not correlational) relationships in pharmaceutical tablet manufacturing.
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                            <div>
                                <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase' }}>Inputs (Controllable)</div>
                                <div className="dag-nodes">{DAG_INPUTS.map(n => <div key={n} className="dag-node input">{n.replace(/_/g, ' ')}</div>)}</div>
                            </div>
                            <div style={{ borderLeft: '2px solid var(--accent-glow)', marginLeft: 12, paddingLeft: 12, color: 'var(--text-muted)', fontSize: 20 }}>↓</div>
                            <div>
                                <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase' }}>Intermediate Variables</div>
                                <div className="dag-nodes">{DAG_MID.map(n => <div key={n} className="dag-node intermediate">{n.replace(/_/g, ' ')}</div>)}</div>
                            </div>
                            <div style={{ borderLeft: '2px solid var(--teal-bg)', marginLeft: 12, paddingLeft: 12, color: 'var(--text-muted)', fontSize: 20 }}>↓</div>
                            <div>
                                <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase' }}>Outputs (Quality Targets)</div>
                                <div className="dag-nodes">{DAG_OUT.map(n => <div key={n} className="dag-node output">{n.replace(/_/g, ' ')}</div>)}</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
