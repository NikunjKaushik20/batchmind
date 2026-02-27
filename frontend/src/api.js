const BASE = 'http://localhost:8000'

async function req(path, options = {}) {
    const res = await fetch(BASE + path, {
        headers: { 'Content-Type': 'application/json' },
        ...options,
    })
    if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`)
    return res.json()
}

export const api = {
    // Batches
    getBatches: () => req('/api/batches'),
    getBatch: (id) => req(`/api/batches/${id}`),
    getBatchPhases: (id) => req(`/api/batches/${id}/phases`),
    getBatchIds: () => req('/api/batches/ids/all'),

    // Fingerprints
    getFingerprints: () => req('/api/fingerprints'),
    getAnomaly: (id) => req(`/api/fingerprints/anomaly/${id}`),
    getAllHealth: () => req('/api/fingerprints/health/all'),

    // Optimizer
    optimize: (body) => req('/api/optimize', { method: 'POST', body: JSON.stringify(body) }),
    counterfactual: (id) => req(`/api/optimize/counterfactual/${id}`),
    explain: (id) => req(`/api/optimize/explain/${id}`),
    getCausalGraph: () => req('/api/optimize/graph'),
    getImportance: () => req('/api/optimize/importance'),

    // Golden Signature
    getSignatures: () => req('/api/golden'),
    createSignature: (body) => req('/api/golden', { method: 'POST', body: JSON.stringify(body) }),
    getPareto: () => req('/api/golden/pareto'),
    updateSignature: (body) => req('/api/golden/update', { method: 'POST', body: JSON.stringify(body) }),

    // Carbon
    getCarbon: () => req('/api/carbon'),
    getCarbonSummary: () => req('/api/carbon/summary'),
    setCarbonBudget: (body) => req('/api/carbon/budget', { method: 'POST', body: JSON.stringify(body) }),
    getCarbonForecast: () => req('/api/carbon/forecast'),

    // CoPilot
    copilotChat: (body) => req('/api/copilot/chat', { method: 'POST', body: JSON.stringify(body) }),
    copilotHealth: () => req('/api/copilot/health'),
}
