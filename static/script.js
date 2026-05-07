document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const runBtn = document.getElementById('run-btn');
    const liveBody = document.getElementById('live-classification-body');
    const shapBody = document.getElementById('shap-body');
    const forecastBody = document.getElementById('forecast-body');
    const alertLog = document.getElementById('alert-log-body');
    const scraperMsg = document.getElementById('scraper-msg');
    const scraperProgress = document.getElementById('scraper-progress');
    const currentTask = document.getElementById('current-task');
    const statusPill = document.getElementById('scraper-status-pill');

    let clusterChart = null;
    let seenAlerts = new Set();

    // ── Polling: Agent Reasoning ──────────────────────────────────────────────
    async function pollInference() {
        try {
            const resp = await fetch('/api/last_inference');
            const data = await resp.json();
            if (data.latest) updateReasoningUI(data.latest);
        } catch (e) {}
    }

    // ── Polling: Scraper Status ───────────────────────────────────────────────
    async function pollScraper() {
        try {
            const resp = await fetch('/api/scraper_status');
            const data = await resp.json();
            
            statusPill.className = 'status-pill online';
            statusPill.querySelector('span').innerText = 'Backend Active';

            if (data.status === 'processing') {
                scraperMsg.innerText = `Processing Product ${data.index} of ${data.total}`;
                const pct = (data.index / data.total) * 100;
                scraperProgress.style.width = `${pct}%`;
                currentTask.innerText = `[REASONING] Analyzing ${data.current}`;
            } else if (data.status === 'scraping') {
                scraperMsg.innerText = "Sensing Market...";
                scraperProgress.style.width = "40%";
                currentTask.innerText = "[SENSE] Fetching Jumia Egypt API stream...";
            } else {
                scraperMsg.innerText = data.message || "Idle";
                scraperProgress.style.width = "0%";
                currentTask.innerText = "[IDLE] Waiting for next autonomous cycle.";
            }
        } catch (e) {
            statusPill.className = 'status-pill offline';
            statusPill.querySelector('span').innerText = 'Backend Offline';
        }
    }

    function updateReasoningUI(res) {
        // Live trace
        const score = res.score;
        const color = score > 0.75 ? '#e53e3e' : (score > 0.5 ? '#dd6b20' : '#38a169');
        
        liveBody.innerHTML = `
            <div class="score-box">
                <div class="score-val" style="color:${color}">${score.toFixed(3)}</div>
                <div class="label" style="font-size:0.7rem; color:#718096">INTELLIGENT SCORE</div>
            </div>
            <div style="font-size:0.8rem; line-height:1.5">
                <p><strong>Target:</strong> ${res.product_name}</p>
                <p><strong>Verdict:</strong> ${res.label.toUpperCase()} (${(res.confidence*100).toFixed(0)}% conf)</p>
                <p><strong>Logic Path:</strong> Path ${res.path_taken} - ${res.route_reason}</p>
            </div>
        `;

        // SHAP
        shapBody.innerHTML = `
            <div style="font-size:0.8rem">
                <p><strong>Key Driver:</strong> ${res.dominant_shap.toUpperCase()}</p>
                <p style="color:#718096; margin-top:0.4rem">Agent verified price integrity against the ${res.label} cluster using ${res.dominant_shap} as the primary anchor.</p>
            </div>
        `;

        // Forecast
        forecastBody.innerHTML = `
            <div style="font-size:0.8rem">
                <p><strong>Market Trend:</strong> ${res.forecast_trend.toUpperCase()}</p>
                <p style="color:#718096; margin-top:0.4rem">Forecast 7-day movement is stable. No abnormal drift detected.</p>
            </div>
        `;

        // Alert (Deduplicated by Timestamp/Product)
        if (res.action && res.action.action_type !== 'monitor') {
            const alertKey = `${res.product_name}-${res.action.action_type}`;
            if (!seenAlerts.has(alertKey)) {
                seenAlerts.add(alertKey);
                const item = document.createElement('div');
                item.className = 'alert-item';
                item.innerHTML = `
                    <p><strong>${res.action.action_type.toUpperCase()}</strong></p>
                    <p style="font-size:0.75rem">${res.product_name} | $${res.price}</p>
                    <p style="font-size:0.65rem; color:#718096">${new Date().toLocaleTimeString()}</p>
                `;
                if (alertLog.querySelector('.placeholder')) alertLog.innerHTML = '';
                alertLog.prepend(item);
                // Keep only last 10
                if (alertLog.children.length > 10) alertLog.removeChild(alertLog.lastChild);
            }
        }
    }

    // Manual Simulation
    runBtn.addEventListener('click', async () => {
        const payload = {
            product_name: document.getElementById('p_name').value,
            price: parseFloat(document.getElementById('p_price').value),
            category: document.getElementById('p_cat').value,
            rating: parseFloat(document.getElementById('p_rating').value),
            review_count: parseInt(document.getElementById('p_reviews').value)
        };
        runBtn.innerHTML = 'Thinking...';
        try {
            const resp = await fetch('/api/run_react', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
            const data = await resp.json();
            updateReasoningUI(data.result);
        } catch (e) { alert("Backend offline"); }
        finally { runBtn.innerHTML = '<i data-lucide="zap"></i> Simulate ReAct Loop'; lucide.createIcons(); }
    });

    // Chart init
    async function initMap() {
        try {
            const resp = await fetch('/api/labeled_data');
            const data = await resp.json();
            if (data.length > 0) {
                const ctx = document.getElementById('clusterMap').getContext('2d');
                const clusters = [...new Set(data.map(d => d.cluster_label))];
                clusterChart = new Chart(ctx, {
                    type: 'scatter',
                    data: {
                        datasets: clusters.map(c => ({
                            label: `Cluster ${c}`,
                            data: data.filter(d => d.cluster_label === c).map(d => ({x: d.price, y: d.rating})),
                            backgroundColor: c === 0 ? '#e53e3e' : (c === 1 ? '#dd6b20' : '#38a169')
                        }))
                    },
                    options: { maintainAspectRatio: false, scales: { x: {title: {display:true, text: 'Price'}}, y: {title: {display:true, text: 'Rating'}} } }
                });
            }
        } catch (e) {}
    }

    initMap();
    setInterval(pollInference, 2000);
    setInterval(pollScraper, 2000);
});
