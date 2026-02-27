import { useState, useRef, useEffect } from 'react'
import { api } from '../api'
import { useLocation } from 'react-router-dom'
import { MessageSquare, X, Send, Cpu, Zap, BarChart2, Leaf, Award, ChevronRight, RefreshCw } from 'lucide-react'

const QUICK_ACTIONS = [
    { label: 'Energy anomalies?', msg: 'Which batches have energy anomalies right now?', icon: Zap },
    { label: 'Optimize for yield', msg: 'Prioritize yield and quality this week', icon: BarChart2 },
    { label: 'Carbon status', msg: "What's our carbon budget status and forecast?", icon: Leaf },
    { label: 'Best signature', msg: 'Show me the balanced golden signature parameters', icon: Award },
]

const PAGE_CONTEXT = {
    '/': 'dashboard',
    '/batch': 'batch-explorer',
    '/optimizer': 'causal-optimizer',
    '/golden': 'golden-signature',
    '/carbon': 'carbon-ledger',
}

// Render structured tool result cards inline in chat
function ToolCard({ name, data }) {
    if (!data || data.error) return null

    if (name === 'run_optimization') {
        const outcomes = data.predicted_outcomes || {}
        return (
            <div style={{
                marginTop: 10, background: 'var(--surface-3)', borderRadius: 10,
                padding: '12px 14px', border: '1px solid var(--border)'
            }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--accent)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
                    ⚡ NSGA-II Results · {data.n_pareto_solutions} Pareto solutions
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                    {[
                        ['Quality', data.best_objectives?.quality],
                        ['Energy', `${data.best_objectives?.energy_kwh?.toFixed(1)} kWh`],
                        ['Yield', outcomes?.Yield_Score?.toFixed(1)],
                        ['Dissolution', `${outcomes?.Dissolution_Rate?.toFixed(1)}%`],
                    ].map(([label, val]) => val != null && (
                        <div key={label} style={{ background: 'var(--surface-2)', borderRadius: 8, padding: '8px 10px' }}>
                            <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600 }}>{label}</div>
                            <div style={{ fontSize: 16, fontWeight: 800, color: 'var(--teal)', marginTop: 2 }}>{val}</div>
                        </div>
                    ))}
                </div>
                {data.top_recommendations && (
                    <div style={{ marginTop: 10 }}>
                        <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, marginBottom: 6 }}>KEY INTERVENTIONS</div>
                        {Object.entries(data.top_recommendations).slice(0, 3).map(([param, val]) => (
                            <div key={param} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 4, color: 'var(--text-secondary)' }}>
                                <span>{param.replace(/_/g, ' ')}</span>
                                <span style={{ fontWeight: 700, color: 'var(--accent)' }}>{val.value?.toFixed(2)}</span>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        )
    }

    if (name === 'analyze_counterfactual') {
        const saved = data.energy_saved_kwh
        const pct = data.pct_energy_saved
        return (
            <div style={{
                marginTop: 10, background: 'var(--surface-3)', borderRadius: 10,
                padding: '12px 14px', border: '1px solid var(--border)'
            }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--teal)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
                    ⚗️ Counterfactual: Batch {data.batch_id}
                </div>
                <div style={{ display: 'flex', gap: 8 }}>
                    <div style={{ flex: 1, background: 'var(--danger-bg)', borderRadius: 8, padding: '8px', textAlign: 'center' }}>
                        <div style={{ fontSize: 9, color: 'var(--danger)', fontWeight: 700 }}>ACTUAL</div>
                        <div style={{ fontSize: 15, fontWeight: 800, color: 'var(--danger)' }}>{data.actual_energy_kwh} kWh</div>
                    </div>
                    <div style={{ flex: 1, background: 'var(--success-bg)', borderRadius: 8, padding: '8px', textAlign: 'center' }}>
                        <div style={{ fontSize: 9, color: 'var(--success)', fontWeight: 700 }}>OPTIMAL</div>
                        <div style={{ fontSize: 15, fontWeight: 800, color: 'var(--success)' }}>{data.counterfactual_energy_kwh} kWh</div>
                    </div>
                    <div style={{ flex: 1, background: 'var(--teal-bg)', borderRadius: 8, padding: '8px', textAlign: 'center' }}>
                        <div style={{ fontSize: 9, color: 'var(--teal)', fontWeight: 700 }}>SAVED</div>
                        <div style={{ fontSize: 15, fontWeight: 800, color: 'var(--teal)' }}>{pct}%</div>
                    </div>
                </div>
                {data.top_interventions && (
                    <div style={{ marginTop: 10 }}>
                        <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, marginBottom: 6 }}>do-calculus INTERVENTIONS</div>
                        {Object.entries(data.top_interventions).slice(0, 3).map(([param, change]) => (
                            <div key={param} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 4, color: 'var(--text-secondary)' }}>
                                <span>{param.replace(/_/g, ' ')}</span>
                                <span>
                                    <span style={{ color: 'var(--text-muted)' }}>{change.actual}</span>
                                    <span style={{ margin: '0 4px', color: 'var(--text-muted)' }}>→</span>
                                    <span style={{ fontWeight: 700, color: change.direction === 'increase' ? 'var(--success)' : 'var(--warning)' }}>
                                        {change.counterfactual}
                                    </span>
                                </span>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        )
    }

    if (name === 'get_carbon_analysis') {
        const s = data.summary || {}
        const f = data.forecast || {}
        return (
            <div style={{
                marginTop: 10, background: 'var(--surface-3)', borderRadius: 10,
                padding: '12px 14px', border: '1px solid var(--border)'
            }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--teal)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
                    🌱 Carbon Status
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                    {[
                        ['Total CO₂e', `${s.total_carbon_kg?.toFixed(0)} kg`],
                        ['Budget Used', `${s.budget_used_pct?.toFixed(0)}%`],
                        ['Forecast', `${f.projected_month_end_kg?.toFixed(0)} kg`],
                        ['Savings Avail.', `${s.potential_savings_kg?.toFixed(0)} kg`],
                    ].map(([label, val]) => (
                        <div key={label} style={{ background: 'var(--surface-2)', borderRadius: 8, padding: '7px 10px' }}>
                            <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600 }}>{label}</div>
                            <div style={{ fontSize: 14, fontWeight: 800, color: 'var(--teal)', marginTop: 1 }}>{val}</div>
                        </div>
                    ))}
                </div>
            </div>
        )
    }

    if (name === 'get_anomaly_report') {
        const critical = data.critical_batches || []
        const warning = data.warning_batches || []
        return (
            <div style={{
                marginTop: 10, background: 'var(--surface-3)', borderRadius: 10,
                padding: '12px 14px', border: '1px solid var(--border)'
            }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--danger)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
                    ⚠️ Asset Health — DTW Anomaly Report
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>Fleet avg: {data.fleet_avg_health}%</div>
                {[...critical, ...warning].slice(0, 4).map(b => (
                    <div key={b.batch_id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)' }}>{b.batch_id}</span>
                        <span className={`badge ${b.overall_health < 70 ? 'badge-danger' : 'badge-warning'}`}>
                            {b.overall_health?.toFixed(0)}%
                        </span>
                    </div>
                ))}
            </div>
        )
    }

    if (name === 'get_golden_signature') {
        const params = data.top_parameters || {}
        return (
            <div style={{
                marginTop: 10, background: 'var(--surface-3)', borderRadius: 10,
                padding: '12px 14px', border: '1px solid var(--border)'
            }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--purple)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
                    🎯 Golden Signature — {data.label}
                </div>
                {Object.entries(params).map(([k, v]) => (
                    <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 5, color: 'var(--text-secondary)' }}>
                        <span>{k.replace(/_/g, ' ')}</span>
                        <span>
                            <span style={{ fontWeight: 700, color: 'var(--text)' }}>{v.value?.toFixed(2)}</span>
                            <span style={{ color: 'var(--text-muted)', fontSize: 10, marginLeft: 4 }}>
                                [{v.ci_low?.toFixed(1)}, {v.ci_high?.toFixed(1)}]
                            </span>
                        </span>
                    </div>
                ))}
            </div>
        )
    }

    return null
}

function ChatMessage({ msg }) {
    const isUser = msg.role === 'user'
    return (
        <div style={{
            display: 'flex',
            justifyContent: isUser ? 'flex-end' : 'flex-start',
            marginBottom: 12,
        }}>
            <div style={{ maxWidth: '88%' }}>
                {!isUser && (
                    <div style={{ fontSize: 10, color: 'var(--accent)', fontWeight: 700, marginBottom: 4, letterSpacing: 0.5 }}>
                        ⚡ BATCHMIND COPILOT
                    </div>
                )}
                <div style={{
                    padding: '10px 14px',
                    borderRadius: isUser ? '14px 14px 4px 14px' : '4px 14px 14px 14px',
                    background: isUser ? 'var(--accent)' : 'var(--surface-2)',
                    color: isUser ? 'white' : 'var(--text)',
                    fontSize: 13,
                    lineHeight: 1.55,
                    border: isUser ? 'none' : '1px solid var(--border)',
                    whiteSpace: 'pre-wrap',
                }}>
                    {msg.content}
                </div>

                {/* Tool result cards */}
                {msg.toolResults?.map((tr, i) => (
                    <ToolCard key={i} name={tr.name} data={tr.data} />
                ))}
            </div>
        </div>
    )
}

export default function CopilotChat() {
    const [open, setOpen] = useState(false)
    const [messages, setMessages] = useState([
        {
            role: 'assistant',
            content: "I'm BatchMind CoPilot — your industrial AI assistant.\n\nI can run causal optimization, analyze energy anomalies, explain counterfactuals, and monitor carbon budgets. What do you need?",
            toolResults: [],
        }
    ])
    const [input, setInput] = useState('')
    const [loading, setLoading] = useState(false)
    const [copilotMode, setCopilotMode] = useState('...')
    const bottomRef = useRef(null)
    const inputRef = useRef(null)
    const loc = useLocation()

    useEffect(() => {
        api.copilotHealth().then(h => setCopilotMode(h.mode)).catch(() => setCopilotMode('fallback'))
    }, [])

    useEffect(() => {
        if (open) {
            setTimeout(() => inputRef.current?.focus(), 300)
        }
    }, [open])

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages, loading])

    const send = async (messageText) => {
        const text = messageText || input.trim()
        if (!text || loading) return
        setInput('')

        const userMsg = { role: 'user', content: text }
        setMessages(prev => [...prev, userMsg])
        setLoading(true)

        try {
            const history = messages.map(m => ({ role: m.role, content: m.content }))
            const res = await api.copilotChat({
                message: text,
                history,
                context: { page: PAGE_CONTEXT[loc.pathname] || loc.pathname },
            })

            const assistantMsg = {
                role: 'assistant',
                content: res.reply || 'Analysis complete.',
                toolResults: res.tool_results || [],
            }
            setMessages(prev => [...prev, assistantMsg])
        } catch (e) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: '⚠️ Connection error. Make sure the backend is running on port 8000.',
                toolResults: [],
            }])
        }
        setLoading(false)
    }

    const hasNotification = messages.length > 1 && !open

    return (
        <>
            {/* ── FLOATING BUTTON ── */}
            <button
                onClick={() => setOpen(o => !o)}
                style={{
                    position: 'fixed',
                    bottom: 24,
                    right: 24,
                    width: 56,
                    height: 56,
                    borderRadius: '50%',
                    background: 'linear-gradient(135deg, var(--accent), var(--teal))',
                    border: 'none',
                    color: 'white',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    cursor: 'pointer',
                    boxShadow: '0 4px 20px rgba(99,102,241,0.5)',
                    zIndex: 1000,
                    transition: 'all 0.2s ease',
                    transform: open ? 'rotate(90deg)' : 'none',
                }}
                title="BatchMind CoPilot"
            >
                {open ? <X size={22} /> : <MessageSquare size={22} />}
                {hasNotification && !open && (
                    <span style={{
                        position: 'absolute', top: 2, right: 2,
                        width: 12, height: 12, borderRadius: '50%',
                        background: 'var(--warning)', border: '2px solid var(--bg)',
                    }} />
                )}
            </button>

            {/* ── CHAT PANEL ── */}
            <div style={{
                position: 'fixed',
                bottom: 90,
                right: 24,
                width: 380,
                height: 560,
                maxWidth: 'calc(100vw - 48px)',
                maxHeight: 'calc(100vh - 120px)',
                background: 'var(--surface)',
                border: '1px solid var(--border)',
                borderRadius: 20,
                boxShadow: '0 20px 60px rgba(0,0,0,0.25)',
                display: 'flex',
                flexDirection: 'column',
                zIndex: 999,
                overflow: 'hidden',
                transform: open ? 'scale(1) translateY(0)' : 'scale(0.9) translateY(20px)',
                opacity: open ? 1 : 0,
                pointerEvents: open ? 'all' : 'none',
                transition: 'all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1)',
                transformOrigin: 'bottom right',
            }}>

                {/* Header */}
                <div style={{
                    padding: '14px 18px',
                    borderBottom: '1px solid var(--border)',
                    background: 'linear-gradient(135deg, var(--accent-glow), var(--teal-bg))',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                }}>
                    <div style={{
                        width: 36, height: 36, borderRadius: 10,
                        background: 'linear-gradient(135deg, var(--accent), var(--teal))',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                    }}>
                        <Cpu size={18} color="white" />
                    </div>
                    <div style={{ flex: 1 }}>
                        <div style={{ fontWeight: 700, fontSize: 14, color: 'var(--text)' }}>BatchMind CoPilot</div>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                            {copilotMode === 'gpt-4o-mini'
                                ? '🟢 GPT-4o-mini · Function calling active'
                                : '🟡 Rule-based mode (add OpenAI key)'}
                        </div>
                    </div>
                    <button
                        onClick={() => setMessages([{
                            role: 'assistant',
                            content: "Conversation cleared. What can I help you with?",
                            toolResults: [],
                        }])}
                        style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: 4 }}
                        title="Clear conversation"
                    >
                        <RefreshCw size={14} />
                    </button>
                </div>

                {/* Messages */}
                <div style={{ flex: 1, overflowY: 'auto', padding: '14px 14px 4px' }}>
                    {messages.map((msg, i) => <ChatMessage key={i} msg={msg} />)}

                    {loading && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12, padding: '0 4px' }}>
                            <div style={{
                                width: 28, height: 28, borderRadius: 8,
                                background: 'linear-gradient(135deg, var(--accent), var(--teal))',
                                display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                            }}>
                                <Cpu size={13} color="white" />
                            </div>
                            <div style={{
                                padding: '8px 14px',
                                borderRadius: '4px 14px 14px 14px',
                                background: 'var(--surface-2)',
                                border: '1px solid var(--border)',
                                display: 'flex', gap: 5, alignItems: 'center',
                            }}>
                                {[0, 1, 2].map(i => (
                                    <span key={i} style={{
                                        width: 6, height: 6, borderRadius: '50%',
                                        background: 'var(--accent)',
                                        animation: `pulse 1.2s ease ${i * 0.2}s infinite`,
                                    }} />
                                ))}
                            </div>
                        </div>
                    )}
                    <div ref={bottomRef} />
                </div>

                {/* Quick actions */}
                {messages.length <= 2 && (
                    <div style={{ padding: '6px 14px', display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                        {QUICK_ACTIONS.map(({ label, msg, icon: Icon }) => (
                            <button
                                key={label}
                                onClick={() => send(msg)}
                                disabled={loading}
                                style={{
                                    padding: '5px 10px',
                                    borderRadius: 100,
                                    background: 'var(--surface-2)',
                                    border: '1px solid var(--border)',
                                    color: 'var(--text-secondary)',
                                    fontSize: 11,
                                    fontWeight: 600,
                                    cursor: 'pointer',
                                    display: 'flex', alignItems: 'center', gap: 4,
                                    transition: 'all 0.15s',
                                    fontFamily: 'var(--font)',
                                }}
                            >
                                <Icon size={11} />
                                {label}
                            </button>
                        ))}
                    </div>
                )}

                {/* Input */}
                <div style={{
                    padding: '10px 12px',
                    borderTop: '1px solid var(--border)',
                    display: 'flex', gap: 8, alignItems: 'flex-end',
                }}>
                    <textarea
                        ref={inputRef}
                        value={input}
                        onChange={e => setInput(e.target.value)}
                        onKeyDown={e => {
                            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
                        }}
                        placeholder="Ask about energy, quality, carbon…"
                        rows={1}
                        style={{
                            flex: 1,
                            padding: '9px 12px',
                            background: 'var(--surface-2)',
                            border: '1px solid var(--border)',
                            borderRadius: 12,
                            color: 'var(--text)',
                            fontSize: 13,
                            outline: 'none',
                            resize: 'none',
                            fontFamily: 'var(--font)',
                            lineHeight: 1.4,
                            maxHeight: 80,
                            overflowY: 'auto',
                        }}
                        disabled={loading}
                    />
                    <button
                        onClick={() => send()}
                        disabled={loading || !input.trim()}
                        style={{
                            width: 38, height: 38,
                            borderRadius: 12,
                            background: loading || !input.trim() ? 'var(--surface-3)' : 'var(--accent)',
                            border: 'none',
                            color: loading || !input.trim() ? 'var(--text-muted)' : 'white',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            cursor: loading || !input.trim() ? 'default' : 'pointer',
                            transition: 'all 0.15s',
                            flexShrink: 0,
                        }}
                    >
                        <Send size={16} />
                    </button>
                </div>
            </div>
        </>
    )
}
