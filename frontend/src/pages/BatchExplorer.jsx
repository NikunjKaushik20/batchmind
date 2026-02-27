import { useEffect, useState } from 'react'
import { api } from '../api'
import {
    ResponsiveContainer, LineChart, Line, XAxis, YAxis,
    CartesianGrid, Tooltip, Legend, AreaChart, Area, ReferenceLine
} from 'recharts'
import { Search } from 'lucide-react'

const COLORS = {
    Power_Consumption_kW: 'var(--accent)',
    Temperature_C: 'var(--warning)',
    Vibration_mm_s: 'var(--danger)',
    Pressure_Bar: 'var(--teal)',
}

export default function BatchExplorer() {
    const [ids, setIds] = useState([])
    const [selected, setSelected] = useState('T001')
    const [batch, setBatch] = useState(null)
    const [anomaly, setAnomaly] = useState(null)
    const [loading, setLoading] = useState(false)

    useEffect(() => {
        api.getBatchIds().then(setIds).catch(console.error)
    }, [])

    useEffect(() => {
        if (!selected) return
        setLoading(true)
        Promise.all([api.getBatch(selected), api.getAnomaly(selected)])
            .then(([b, a]) => { setBatch(b); setAnomaly(a) })
            .catch(console.error)
            .finally(() => setLoading(false))
    }, [selected])

    const ts = batch?.time_series || []
    const summary = batch?.summary || {}

    // Downsample for chart perf
    const chartData = ts.filter((_, i) => i % 2 === 0).map(r => ({
        t: r.Time_Minutes,
        Power: parseFloat(r.Power_Consumption_kW?.toFixed(2)),
        Temp: parseFloat(r.Temperature_C?.toFixed(1)),
        Vibration: parseFloat((r.Vibration_mm_s * 10)?.toFixed(2)),
        Pressure: parseFloat(r.Pressure_Bar?.toFixed(2)),
        Phase: r.Phase,
    }))

    const phases = batch ? [...new Set(ts.map(r => r.Phase))] : []

    return (
        <div className="page fade-in">
            <div className="page-header">
                <h2>Batch Explorer</h2>
                <p>Minute-level sensor data, phase fingerprinting, and anomaly scoring</p>
            </div>

            {/* Selector */}
            <div className="card" style={{ marginBottom: 20, padding: '16px 20px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
                    <Search size={18} color="var(--text-muted)" />
                    <label className="form-label" style={{ margin: 0, whiteSpace: 'nowrap' }}>Select Batch:</label>
                    <select className="form-select" style={{ maxWidth: 200 }} value={selected} onChange={e => setSelected(e.target.value)}>
                        {ids.map(id => <option key={id} value={id}>{id}</option>)}
                    </select>
                    {anomaly && (
                        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                            <span className={`badge ${anomaly.overall_health >= 75 ? 'badge-success' : anomaly.overall_health >= 50 ? 'badge-warning' : 'badge-danger'}`}>
                                Asset Health: {anomaly.overall_health?.toFixed(0)}%
                            </span>
                            {phases.map(p => <span key={p} className="phase-chip">{p}</span>)}
                        </div>
                    )}
                </div>
            </div>

            {loading && <div className="loading"><div className="spinner" /><p>Loading batch data...</p></div>}

            {!loading && batch && (
                <>
                    {/* KPIs */}
                    <div className="kpi-grid" style={{ marginBottom: 20 }}>
                        <div className="kpi-card accent">
                            <div className="kpi-label">Duration</div>
                            <div className="kpi-value">{summary.duration_minutes} <span style={{ fontSize: 14 }}>min</span></div>
                            <div className="kpi-change">Production run time</div>
                        </div>
                        <div className="kpi-card warning">
                            <div className="kpi-label">Total Energy</div>
                            <div className="kpi-value">{summary.total_kwh} <span style={{ fontSize: 14 }}>kWh</span></div>
                            <div className="kpi-change">Avg {summary.avg_power_kw} kW</div>
                        </div>
                        <div className="kpi-card teal">
                            <div className="kpi-label">Quality Score</div>
                            <div className="kpi-value">{(summary.quality_score || 0).toFixed(1)}</div>
                            <div className="kpi-change">Composite quality index</div>
                        </div>
                        <div className="kpi-card danger">
                            <div className="kpi-label">Carbon</div>
                            <div className="kpi-value">{summary.carbon_kg_co2e} <span style={{ fontSize: 14 }}>kg</span></div>
                            <div className="kpi-change">CO₂e this batch</div>
                        </div>
                    </div>

                    {/* Time-Series Chart */}
                    <div className="card" style={{ marginBottom: 20 }}>
                        <div className="section-title">
                            Power Consumption & Process Parameters
                            <small>Minute-level sensor data</small>
                        </div>
                        <div className="chart-wrapper">
                            <ResponsiveContainer width="100%" height={320}>
                                <AreaChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 10 }}>
                                    <defs>
                                        <linearGradient id="powerGrad" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="var(--accent)" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="var(--accent)" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                                    <XAxis dataKey="t" tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                                        label={{ value: 'Time (min)', position: 'insideBottom', offset: -5, fill: 'var(--text-muted)', fontSize: 11 }} />
                                    <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                                    <Tooltip contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }} />
                                    <Legend wrapperStyle={{ fontSize: 12, color: 'var(--text-secondary)' }} />
                                    <Area type="monotone" dataKey="Power" stroke="var(--accent)" strokeWidth={2}
                                        fill="url(#powerGrad)" name="Power (kW)" dot={false} />
                                    <Line type="monotone" dataKey="Temp" stroke="var(--warning)" strokeWidth={1.5}
                                        name="Temp (°C)" dot={false} />
                                    <Line type="monotone" dataKey="Vibration" stroke="var(--danger)" strokeWidth={1.5}
                                        name="Vibration ×10" dot={false} />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Fingerprint & Anomaly */}
                    <div className="grid-2" style={{ marginBottom: 20 }}>
                        <div className="card">
                            <div className="section-title">Phase-Aware Energy Fingerprint</div>
                            {anomaly && Object.entries(anomaly.phases).map(([phase, data]) => (
                                <div key={phase} style={{ marginBottom: 16 }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                                        <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)' }}>{phase}</span>
                                        <span className={`badge ${data.asset_health >= 75 ? 'badge-success' : data.asset_health >= 50 ? 'badge-warning' : 'badge-danger'}`}>
                                            Health: {data.asset_health?.toFixed(0)}%
                                        </span>
                                    </div>
                                    <div className="health-bar">
                                        <div className={`health-bar-fill ${data.asset_health >= 75 ? 'high' : data.asset_health >= 50 ? 'medium' : 'low'}`}
                                            style={{ width: `${data.asset_health}%` }} />
                                    </div>
                                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                                        Anomaly score: {data.anomaly_score?.toFixed(1)} / 100
                                    </div>
                                </div>
                            ))}
                        </div>

                        <div className="card">
                            <div className="section-title">Production Quality Outcomes</div>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                                {[
                                    ['Hardness', summary.Hardness, 'kP'],
                                    ['Friability', summary.Friability, '%'],
                                    ['Dissolution Rate', summary.Dissolution_Rate, '%'],
                                    ['Disintegration', summary.Disintegration_Time, 'min'],
                                    ['Content Uniformity', summary.Content_Uniformity, '%'],
                                    ['Tablet Weight', summary.Tablet_Weight, 'mg'],
                                ].map(([label, val, unit]) => (
                                    <div key={label} className="param-card">
                                        <div className="param-name">{label}</div>
                                        <div className="param-value">{val ?? '--'} <span style={{ fontSize: 12, fontWeight: 400 }}>{unit}</span></div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Fingerprint visual for prep phase */}
                    {anomaly?.phases?.Preparation?.curve && (
                        <div className="card">
                            <div className="section-title">
                                Preparation Phase — Power Curve vs Golden Fingerprint
                                <small>blue = batch, shaded = normal range</small>
                            </div>
                            <div className="chart-wrapper">
                                <ResponsiveContainer width="100%" height={220}>
                                    <AreaChart
                                        data={anomaly.phases.Preparation.curve.map((v, i) => ({
                                            i, batch: v.toFixed(2),
                                            upper: anomaly.phases.Preparation.upper_band?.[i]?.toFixed(2),
                                            lower: anomaly.phases.Preparation.lower_band?.[i]?.toFixed(2),
                                            mean: anomaly.phases.Preparation.mean_curve?.[i]?.toFixed(2),
                                        }))}
                                        margin={{ top: 10, right: 20, left: -10, bottom: 10 }}
                                    >
                                        <defs>
                                            <linearGradient id="bandGrad" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="var(--teal)" stopOpacity={0.2} />
                                                <stop offset="95%" stopColor="var(--teal)" stopOpacity={0.05} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                                        <XAxis dataKey="i" tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
                                            label={{ value: 'Normalized Time', position: 'insideBottom', offset: -5, fill: 'var(--text-muted)', fontSize: 11 }} />
                                        <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                                        <Tooltip contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }} />
                                        <Area type="monotone" dataKey="upper" stroke="none" fill="url(#bandGrad)" name="Normal range" />
                                        <Line type="monotone" dataKey="mean" stroke="var(--teal)" strokeWidth={2}
                                            strokeDasharray="5 3" name="Mean fingerprint" dot={false} />
                                        <Line type="monotone" dataKey="batch" stroke="var(--accent)" strokeWidth={2.5}
                                            name={`Batch ${selected}`} dot={false} />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    )}
                </>
            )}
        </div>
    )
}
