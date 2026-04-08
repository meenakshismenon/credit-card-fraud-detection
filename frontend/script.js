const API_URL = 'http://localhost:8000';
let liveFraudCount = 0;

// Initialize Dashboard Data
async function initDashboard() {
    try {
        const response = await fetch(`${API_URL}/metrics`);
        if (!response.ok) throw new Error('API not available');
        const data = await response.json();
        
        if (data.error) {
            console.error(data.error);
            return;
        }

        const metricsList = data.metrics;
        const bestModelName = data.best_model;
        const bestModel = metricsList[bestModelName];

        // Update KPIs
        document.getElementById('kpi-model').textContent = bestModelName;
        document.getElementById('kpi-acc').textContent = `${(bestModel['Accuracy'] * 100).toFixed(1)}%`;
        document.getElementById('kpi-auc').textContent = `${(bestModel['AUC-ROC'] * 100).toFixed(1)}%`;

        // Render Chart
        renderChart(metricsList);

    } catch (e) {
        console.error('Failed to load metrics:', e);
    }
}

// Render Comparison Chart using Chart.js
function renderChart(metricsList) {
    const ctx = document.getElementById('aucChart').getContext('2d');
    
    // Sort models by AUC
    const sortedModels = Object.keys(metricsList).sort((a,b) => metricsList[b]['AUC-ROC'] - metricsList[a]['AUC-ROC']);
    const aucData = sortedModels.map(model => metricsList[model]['AUC-ROC']);
    const f1Data = sortedModels.map(model => metricsList[model]['F1-Score']);

    Chart.defaults.color = '#8b8b99';
    Chart.defaults.font.family = "'Rajdhani', sans-serif";

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sortedModels,
            datasets: [
                {
                    label: 'AUC-ROC',
                    data: aucData,
                    backgroundColor: 'rgba(255, 0, 255, 0.5)',
                    borderColor: '#f0f',
                    borderWidth: 1
                },
                {
                    label: 'F1 Score',
                    data: f1Data,
                    backgroundColor: 'rgba(0, 255, 255, 0.5)',
                    borderColor: '#0ff',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    grid: { color: 'rgba(255,255,255,0.1)' }
                },
                x: {
                    grid: { color: 'rgba(255,255,255,0.1)' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#e0e0ff' }
                }
            }
        }
    });
}

// Form Submission targetting /predict
document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const payload = {
        amount: parseFloat(document.getElementById('amount').value),
        transaction_hour: parseInt(document.getElementById('hour').value),
        merchant_type: document.getElementById('merchant').value,
        location: document.getElementById('location').value,
        account_age: parseInt(document.getElementById('age').value)
    };

    const submitBtn = e.target.querySelector('button');
    submitBtn.textContent = "ANALYZING...";
    submitBtn.style.pointerEvents = "none";

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        
        const resultBox = document.getElementById('result-box');
        resultBox.classList.remove('hidden', 'result-fraud', 'result-safe');
        
        document.getElementById('result-status').textContent = `STATUS: ${data.prediction.toUpperCase()}`;
        document.getElementById('result-score').textContent = `${data.fraud_probability}%`;
        document.getElementById('result-explanation').textContent = `// ${data.explanation}`;

        if (data.prediction.toLowerCase() === 'fraud') {
            resultBox.classList.add('result-fraud');
            liveFraudCount++;
            document.getElementById('kpi-live').textContent = liveFraudCount;
        } else {
            resultBox.classList.add('result-safe');
        }

    } catch (error) {
        console.error("Error connecting to API");
        alert("Cannot connect to the ML Backend API. Is it running?");
    } finally {
        submitBtn.textContent = "ANALYZE TRANSACTION";
        submitBtn.style.pointerEvents = "all";
    }
});

// Live Transaction Simulator
function simulateLiveStream() {
    const merchants = ['Groceries', 'Dining', 'Retail', 'Online Subscription', 'Travel', 'Electronics', 'Crypto'];
    const locations = ['Local', 'Domestic', 'International', 'DarkWeb'];
    
    setInterval(async () => {
        // Generate random realistic payload based on distributions
        const isFraudHeavy = Math.random() > 0.8; // 20% chance to generate a sketchy transaction
        
        const payload = {
            amount: parseFloat((Math.random() * (isFraudHeavy ? 5000 : 300)).toFixed(2)),
            transaction_hour: Math.floor(Math.random() * 24),
            merchant_type: merchants[Math.floor(Math.random() * merchants.length)],
            location: isFraudHeavy ? 'DarkWeb' : locations[Math.floor(Math.random() * 3)],
            account_age: Math.floor(Math.random() * 365)
        };

        try {
            // Predict silently with backend
            const response = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            
            appendLog(payload, data);
        } catch (e) {
            // API not ready, do nothing silently
        }
    }, 2500); // New tx every 2.5 seconds
}

function appendLog(payload, result) {
    const container = document.getElementById('stream-container');
    const isFraud = result.prediction.toLowerCase() === 'fraud';
    
    if (isFraud) {
        liveFraudCount++;
        document.getElementById('kpi-live').textContent = liveFraudCount;
    }

    const logEl = document.createElement('div');
    logEl.className = `tx-log ${isFraud ? 'fraud' : 'safe'}`;
    
    const formattedAmount = `$${payload.amount.toFixed(2)}`;
    
    logEl.innerHTML = `
        <span>[${new Date().toLocaleTimeString()}] ${payload.merchant_type} @ ${payload.location}</span>
        <span class="tx-amount">${formattedAmount}</span>
        <span>${isFraud ? '>> ALERT <<' : 'OK'}</span>
    `;
    
    container.insertBefore(logEl, container.firstChild);
    
    // Keep only last 20 elements
    if (container.children.length > 20) {
        container.lastChild.remove();
    }
}

// Start
initDashboard();
simulateLiveStream();
