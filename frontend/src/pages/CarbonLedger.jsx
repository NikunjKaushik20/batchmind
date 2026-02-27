import { useEffect, useState } from 'react'
import { api } from '../api'
import {
    ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid,
    Tooltip, LineChart, Line, ReferenceLine, Cell
} from 'recharts'
import { Leaf, TrendingUp, TrendingDown, Minus, AlertTriangle } from 'lucide-react'

export default function CarbonLedger() {
    const [carbon, setCarbon] = useState([])
    const [summary, setSummary] = useState(null)
    const [forecast, setForecast] = useState(null)
    const [budget, setBudget] = useState('')
    const [loading, setLoading] = useState(true)

    const load = () => {
        Promise.all([api.getCarbon(), api.getCarbonSummary(), api.getCarbonForecast()])
            .then(([c, s, f]) => { setCarbon(c); setSummary(s); setForecast(f) })
            .catch(console.error)
            .finally(() => setLoading(false))
    }
    useEffect(load, [])

    const handleBudget = async () => {
        if (!budget) return
        await api.setCarbonBudget({ budget_kg: parseFloat(budget) })
        load()
    }

    const statusColor = { on_track: 'var(--success)', at_risk: 'var(--warning)', exceeded: 'var(--danger)' }
    const statusIcon = { on_track: TrendingDown, at_risk: AlertTriangle, exceeded: TrendingUp }

    const cumulativeData = carbon.map((b, i) => ({
        name: b.Batch_ID,
        carbon: b.carbon_kg,
        cumulative: carbon.slice(0, i + 1).reduce((s, x) => s + x.carbon_kg, 0).toFixed(2),
    }))

    // Top opportunities
    const opportunities = [...carbon].sort((a, b) => b.carbon_kg - a.carbon_kg).slice(0, 5)

    if (loading) return <div className="loading"><div className="spinner" /></div>

    const StatusIcon = summary ? statusIcon[summary.status] || Minus : Minus

    return (
        <div className="page fade-in">
            <div className="page-header">
                <h2>Carbon Ledger</h2>
                <p>Per-batch carbon accounting, budget tracking & reduction opportunities</p>
            </div>

            {/* KPI Grid */}
            {summary && (
                <div className="kpi-grid" style={{ marginBottom: 20 }}>
                    <div className="kpi-card teal">
                        <div className="kpi-label">Total Carbon</div>
                        <div className="kpi-value">{summary.total_carbon_kg.toFixed(0)} <span style={{ fontSize: 14 }}>kg</span></div>
                        <div className="kpi-change">CO₂e across {summary.n_batches} batches</div>
                        <div className="kpi-icon"><Leaf size={18} /></div>
                    </div>
                    <div className="kpi-card warning">
                        <div className="kpi-label">Avg / Batch</div>
                        <div className="kpi-value">{summary.avg_carbon_per_batch_kg.toFixed(2)} <span style={{ fontSize: 14 }}>kg</span></div>
                        <div className="kpi-change">India grid: 0.82 kg CO₂e/kWh</div>
                    </div>
                    <div className="kpi-card" style={{ '--before-color': statusColor[summary.status] }}>
                        <div className="kpi-label">Budget Used</div>
                        <div className="kpi-value" style={{ color: statusColor[summary.status] }}>
                            {summary.budget_used_pct.toFixed(1)}%
                        </div>
                        <div className="kpi-change">of {summary.monthly_budget_kg} kg budget</div>
                    </div>
                    <div className="kpi-card success">
                        <div className="kpi-label">Potential Savings</div>
                        <div className="kpi-value">{summary.potential_savings_kg.toFixed(0)} <span style={{ fontSize: 14 }}>kg</span></div>
                        <div className="kpi-change">Through optimization</div>
                    </div>
                </div>
            )}

            <div className="grid-2" style={{ marginBottom: 20 }}>
                {/* Per Batch Carbon Bar Chart */}
                <div className="card">
                    <div className="section-title">Per-Batch Carbon Emissions</div>
                    <div className="chart-wrapper">
                        <ResponsiveContainer width="100%" height={280}>
                            <BarChart data={carbon.slice(0, 30)} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                                <XAxis dataKey="Batch_ID" tick={{ fill: 'var(--text-muted)', fontSize: 9 }} interval={4} />
                                <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                                <Tooltip contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }} />
                                <Bar dataKey="carbon_kg" name="Carbon (kg CO₂e)" radius={[3, 3, 0, 0]}>
                                    {carbon.slice(0, 30).map((_, i) => (
                                        <Cell key={i} fill={`hsl(${145 + i * 2}, 65%, 45%)`} />
                                    ))}
                                </Bar>
                                {summary && <ReferenceLine y={summary.avg_carbon_per_batch_kg} stroke="var(--warning)" strokeDasharray="5 3" label={{ value: 'avg', fill: 'var(--warning)', fontSize: 11 }} />}
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Cumulative vs Budget */}
                <div className="card">
                    <div className="section-title">Cumulative Carbon vs Budget</div>
                    <div className="chart-wrapper">
                        <ResponsiveContainer width="100%" height={280}>
                            <LineChart data={cumulativeData} margin={{ top: 10, right: 20, left: -10, bottom: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                                <XAxis dataKey="name" tick={{ fill: 'var(--text-muted)', fontSize: 9 }} interval={9} />
                                <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                                <Tooltip contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }} />
                                <Line type="monotone" dataKey="cumulative" stroke="var(--teal)" strokeWidth={2.5} dot={false} name="Cumulative CO₂e (kg)" />
                                {summary && <ReferenceLine y={summary.monthly_budget_kg} stroke="var(--danger)" strokeDasharray="6 3"
                                    label={{ value: `Budget: ${summary.monthly_budget_kg}kg`, fill: 'var(--danger)', fontSize: 11 }} />}
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            <div className="grid-2" style={{ marginBottom: 20 }}>
                {/* Forecast Card */}
                {forecast && (
                    <div className="card">
                        <div className="section-title">📈 Month-End Forecast</div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                                <StatusIcon size={32} color={statusColor[forecast.status]} />
                                <div>
                                    <div style={{ fontSize: 22, fontWeight: 800, color: statusColor[forecast.status] }}>
                                        {forecast.status === 'on_track' ? 'On Track' : 'At Risk'}
                                    </div>
                                    <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                                        Trend: {forecast.trend} · {forecast.batches_remaining_estimate} batches remaining
                                    </div>
                                </div>
                            </div>
                            {[
                                ['Current Carbon', `${forecast.current_total_kg.toFixed(1)} kg CO₂e`],
                                ['Projected Month-End', `${forecast.projected_month_end_kg.toFixed(1)} kg CO₂e`],
                                ['Surplus / Deficit', `${forecast.projected_surplus_deficit_kg > 0 ? '+' : ''}${forecast.projected_surplus_deficit_kg.toFixed(1)} kg`],
                                ['Avg Per Batch', `${forecast.avg_per_batch_kg.toFixed(2)} kg CO₂e`],
                            ].map(([k, v]) => (
                                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 14px', background: 'var(--surface-2)', borderRadius: 8 }}>
                                    <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{k}</span>
                                    <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text)' }}>{v}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Budget Setter + Opportunities */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                    <div className="card">
                        <div className="section-title">Set Monthly Budget</div>
                        <div style={{ display: 'flex', gap: 10 }}>
                            <input className="form-input" type="number" placeholder="Budget in kg CO₂e"
                                value={budget} onChange={e => setBudget(e.target.value)} />
                            <button className="btn btn-success" onClick={handleBudget}>Set</button>
                        </div>
                        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8 }}>
                            Current: {summary?.monthly_budget_kg} kg CO₂e/month
                        </p>
                    </div>

                    <div className="card">
                        <div className="section-title">🚨 Highest Emitting Batches</div>
                        {opportunities.map((b, i) => (
                            <div key={b.Batch_ID} style={{
                                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                                padding: '9px 0', borderBottom: i < 4 ? '1px solid var(--border)' : 'none'
                            }}>
                                <div style={{ display: 'flex', align: 'center', gap: 10 }}>
                                    <span className="badge badge-danger">{b.Batch_ID}</span>
                                </div>
                                <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text)' }}>{b.carbon_kg.toFixed(2)} kg CO₂e</div>
                                <span className="badge badge-muted">{b.total_kwh.toFixed(1)} kWh</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    )
}
