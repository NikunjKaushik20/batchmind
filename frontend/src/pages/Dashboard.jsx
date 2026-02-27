import { useEffect, useState } from 'react'
import { api } from '../api'
import {
    ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis,
    CartesianGrid, Tooltip, BarChart, Bar, Cell, LineChart, Line, Legend
} from 'recharts'
import { Activity, Zap, Leaf, Award, FlaskConical, TrendingDown } from 'lucide-react'

const healthColor = (h) => h >= 75 ? 'high' : h >= 50 ? 'medium' : 'low'
const healthBadge = (h) => h >= 75 ? 'badge-success' : h >= 50 ? 'badge-warning' : 'badge-danger'

export default function Dashboard() {
    const [batches, setBatches] = useState([])
    const [health, setHealth] = useState([])
    const [carbon, setCarbon] = useState(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        Promise.all([api.getBatches(), api.getAllHealth(), api.getCarbonSummary()])
            .then(([b, h, c]) => { setBatches(b); setHealth(h); setCarbon(c) })
            .catch(console.error)
            .finally(() => setLoading(false))
    }, [])

    if (loading) return <div className="loading"><div className="spinner" /><p>Loading manufacturing data...</p></div>

    const avgQuality = (batches.reduce((s, b) => s + (b.quality_score || 0), 0) / batches.length).toFixed(1)
    const avgEnergy = (batches.reduce((s, b) => s + (b.total_kwh || 0), 0) / batches.length).toFixed(1)
    const avgHealth = (health.reduce((s, h) => s + h.overall_health, 0) / health.length).toFixed(1)
    const topBatches = [...batches].sort((a, b) => (b.quality_score || 0) - (a.quality_score || 0)).slice(0, 5)
    const scatterData = batches.map(b => ({ x: b.total_kwh || 0, y: b.quality_score || 0, id: b.Batch_ID }))
    const healthData = health.slice(0, 20).map(h => ({ name: h.batch_id, health: h.overall_health }))

    return (
        <div className="page fade-in">
            <div className="page-header">
                <h2>Manufacturing Intelligence Dashboard</h2>
                <p>Real-time overview of 60 production batches — energy, quality & asset health</p>
            </div>

            {/* KPI Grid */}
            <div className="kpi-grid">
                <div className="kpi-card accent">
                    <div className="kpi-label">Total Batches</div>
                    <div className="kpi-value">{batches.length}</div>
                    <div className="kpi-change">Pharmaceutical tablet production</div>
                    <div className="kpi-icon"><FlaskConical size={18} /></div>
                </div>
                <div className="kpi-card teal">
                    <div className="kpi-label">Avg Quality Score</div>
                    <div className="kpi-value">{avgQuality}</div>
                    <div className="kpi-change">Out of 100 composite score</div>
                    <div className="kpi-icon"><Award size={18} /></div>
                </div>
                <div className="kpi-card warning">
                    <div className="kpi-label">Avg Energy / Batch</div>
                    <div className="kpi-value">{avgEnergy} <span style={{ fontSize: 14, fontWeight: 400 }}>kWh</span></div>
                    <div className="kpi-change">Power consumption per batch</div>
                    <div className="kpi-icon"><Zap size={18} /></div>
                </div>
                <div className="kpi-card success">
                    <div className="kpi-label">Avg Asset Health</div>
                    <div className="kpi-value">{avgHealth}%</div>
                    <div className="kpi-change">Equipment reliability index</div>
                    <div className="kpi-icon"><Activity size={18} /></div>
                </div>
                <div className="kpi-card danger">
                    <div className="kpi-label">Total Carbon</div>
                    <div className="kpi-value">{carbon ? carbon.total_carbon_kg.toFixed(0) : '--'} <span style={{ fontSize: 14 }}>kg</span></div>
                    <div className="kpi-change">CO₂e across all batches</div>
                    <div className="kpi-icon"><Leaf size={18} /></div>
                </div>
                <div className="kpi-card purple">
                    <div className="kpi-label">Potential Savings</div>
                    <div className="kpi-value">{carbon ? carbon.potential_savings_kg.toFixed(0) : '--'} <span style={{ fontSize: 14 }}>kg</span></div>
                    <div className="kpi-change">CO₂e reduction opportunity</div>
                    <div className="kpi-icon"><TrendingDown size={18} /></div>
                </div>
            </div>

            {/* Charts Row 1 */}
            <div className="grid-2" style={{ marginBottom: 20 }}>
                <div className="card">
                    <div className="section-title">Energy vs Quality (All Batches) <small>Pareto frontier</small></div>
                    <div className="chart-wrapper">
                        <ResponsiveContainer width="100%" height={280}>
                            <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                                <XAxis dataKey="x" name="Energy (kWh)" tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                                    label={{ value: 'Energy (kWh)', position: 'insideBottom', offset: -10, fill: 'var(--text-muted)', fontSize: 11 }} />
                                <YAxis dataKey="y" name="Quality Score" tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                                    label={{ value: 'Quality', angle: -90, position: 'insideLeft', fill: 'var(--text-muted)', fontSize: 11 }} />
                                <Tooltip
                                    cursor={{ strokeDasharray: '3 3' }}
                                    content={({ payload }) => {
                                        if (!payload?.length) return null
                                        const d = payload[0].payload
                                        return (
                                            <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, padding: '10px 14px', fontSize: 12 }}>
                                                <div style={{ fontWeight: 700, color: 'var(--accent)' }}>{d.id}</div>
                                                <div>Energy: <strong>{d.x?.toFixed(1)} kWh</strong></div>
                                                <div>Quality: <strong>{d.y?.toFixed(1)}</strong></div>
                                            </div>
                                        )
                                    }}
                                />
                                <Scatter data={scatterData} fill="var(--accent)" opacity={0.75} r={5} />
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="card">
                    <div className="section-title">Asset Health — Recent Batches <small>Phase fingerprint score</small></div>
                    <div className="chart-wrapper">
                        <ResponsiveContainer width="100%" height={280}>
                            <BarChart data={healthData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                                <XAxis dataKey="name" tick={{ fill: 'var(--text-muted)', fontSize: 9 }} interval={2} />
                                <YAxis domain={[0, 100]} tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                                <Tooltip contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }} />
                                <Bar dataKey="health" radius={[4, 4, 0, 0]}>
                                    {healthData.map((entry, i) => (
                                        <Cell key={i}
                                            fill={entry.health >= 75 ? 'var(--success)' : entry.health >= 50 ? 'var(--warning)' : 'var(--danger)'}
                                        />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {/* Top Batches Table */}
            <div className="card">
                <div className="section-title">🏆 Top 5 Batches by Quality Score</div>
                <div className="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Batch ID</th>
                                <th>Quality Score</th>
                                <th>Energy (kWh)</th>
                                <th>Carbon (kg CO₂e)</th>
                                <th>Hardness</th>
                                <th>Dissolution Rate</th>
                                <th>Asset Health</th>
                            </tr>
                        </thead>
                        <tbody>
                            {topBatches.map(b => {
                                const h = health.find(x => x.batch_id === b.Batch_ID)
                                return (
                                    <tr key={b.Batch_ID}>
                                        <td><span className="badge badge-accent">{b.Batch_ID}</span></td>
                                        <td><strong>{(b.quality_score || 0).toFixed(1)}</strong></td>
                                        <td className="td-mono">{(b.total_kwh || 0).toFixed(1)}</td>
                                        <td className="td-mono">{(b.carbon_kg_co2e || 0).toFixed(2)}</td>
                                        <td>{b.Hardness ?? '--'}</td>
                                        <td>{b.Dissolution_Rate ?? '--'}%</td>
                                        <td>
                                            {h && (
                                                <span className={`badge ${healthBadge(h.overall_health)}`}>
                                                    {h.overall_health.toFixed(0)}%
                                                </span>
                                            )}
                                        </td>
                                    </tr>
                                )
                            })}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    )
}
