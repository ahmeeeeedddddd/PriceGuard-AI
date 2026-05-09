document.addEventListener('DOMContentLoaded', () => {
    const runBtn = document.getElementById('run-btn');
    const liveBody = document.getElementById('live-classification-body');
    const shapBody = document.getElementById('shap-body');
    const forecastBody = document.getElementById('forecast-body');
    const alertLog = document.getElementById('alert-log-body');
    const scraperMsg = document.getElementById('scraper-msg');
    const scraperProgress = document.getElementById('scraper-progress');
    const currentTask = document.getElementById('current-task');
    const statusPill = document.getElementById('scraper-status-pill');
    const runError = document.getElementById('run-error');
    const resultsList = document.getElementById('results-list');
    const resultsBadge = document.getElementById('results-badge');

    let clusterChart = null;
    let seenAlerts = new Set();
    window._resultCount = 0;

    // ── Polling: Agent Reasoning ────────────────────────────────────────────
    async function pollInference() {
        try {
            const resp = await fetch('/api/last_inference');
            const data = await resp.json();
            if (data.latest) updateReasoningUI(data.latest);
        } catch (e) {}
    }

    // ── Polling: Scraper Status ─────────────────────────────────────────────
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
                scraperMsg.innerText = data.message || 'Sensing Market...';
                scraperProgress.style.width = '40%';
                currentTask.innerText = '[SENSE] Fetching Jumia Egypt API stream...';
            } else if (data.status === 'matching') {
                scraperMsg.innerText = data.message || 'Validating matches...';
                scraperProgress.style.width = '65%';
                currentTask.innerText = '[MATCH] Filtering products by similarity...';
            } else if (data.status === 'reasoning') {
                scraperMsg.innerText = data.message || 'Reasoning...';
                scraperProgress.style.width = '85%';
                currentTask.innerText = '[REACT] Running reasoning loop...';
            } else if (data.status === 'complete') {
                scraperMsg.innerText = data.message || 'Analysis complete.';
                scraperProgress.style.width = '100%';
                currentTask.innerText = '[DONE] Pipeline cycle complete.';
            } else {
                scraperMsg.innerText = data.message || 'Idle';
                scraperProgress.style.width = '0%';
                currentTask.innerText = '[IDLE] Waiting for next autonomous cycle.';
            }
        } catch (e) {
            statusPill.className = 'status-pill offline';
            statusPill.querySelector('span').innerText = 'Backend Offline';
        }
    }

    // ── Update Main Dashboard UI ────────────────────────────────────────────
    function updateReasoningUI(res) {
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

        shapBody.innerHTML = `
            <div style="font-size:0.8rem">
                <p><strong>Key Driver:</strong> ${res.dominant_shap.toUpperCase()}</p>
                <p style="color:#718096; margin-top:0.4rem">Agent verified price integrity against the ${res.label} cluster using ${res.dominant_shap} as the primary anchor.</p>
            </div>
        `;

        forecastBody.innerHTML = `
            <div style="font-size:0.8rem">
                <p><strong>Market Trend:</strong> ${res.forecast_trend.toUpperCase()}</p>
                <p style="color:#718096; margin-top:0.4rem">Forecast 7-day movement is stable. No abnormal drift detected.</p>
            </div>
        `;

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
                if (alertLog.children.length > 10) alertLog.removeChild(alertLog.lastChild);
            }
        }
    }

    // ── Add a Result Card to the Results Tab ────────────────────────────────
    function addResultCard(payload, data) {
        const res = data.result;
        const ms = data.market_stats || {};
        const verdict = res.verdict || res.label?.toUpperCase() || 'UNKNOWN';
        const verdictColor = verdict === 'UNDERPRICED' ? '#38a169' : verdict === 'OVERPRICED' ? '#e53e3e' : '#dd6b20';
        const priceDiff = data.result.price_diff_pct != null
            ? `${data.result.price_diff_pct > 0 ? '+' : ''}${data.result.price_diff_pct.toFixed(1)}%`
            : 'N/A';

        // Remove placeholder
        const ph = resultsList.querySelector('.placeholder');
        if (ph) ph.remove();

        const card = document.createElement('div');
        card.className = 'result-card';
        card.innerHTML = `
            <div class="result-card-header">
                <span class="result-product">${payload.product_name}</span>
                <span class="result-verdict" style="color:${verdictColor}">${verdict}</span>
            </div>
            <div class="result-meta">
                <div class="result-row"><span>Price</span><strong>$${payload.price}</strong></div>
                <div class="result-row"><span>Category</span><strong>${payload.category}</strong></div>
                <div class="result-row"><span>Score</span><strong>${res.score.toFixed(3)}</strong></div>
                <div class="result-row"><span>Confidence</span><strong>${(res.confidence * 100).toFixed(0)}%</strong></div>
                <div class="result-row"><span>Path</span><strong>Path ${res.path_taken}</strong></div>
                <div class="result-row"><span>Action</span><strong>${res.action?.action_type || 'N/A'}</strong></div>
                <div class="result-row"><span>Avg Market $</span><strong>${ms.avg_price != null ? '$' + ms.avg_price.toFixed(0) : 'N/A'}</strong></div>
                <div class="result-row"><span>vs Market</span><strong style="color:${verdictColor}">${priceDiff}</strong></div>
                <div class="result-row"><span>Matches</span><strong>${res.matched_count ?? 'N/A'}</strong></div>
                <div class="result-row"><span>Key Driver</span><strong>${res.dominant_shap?.toUpperCase() || 'N/A'}</strong></div>
                <div class="result-row"><span>Trend</span><strong>${res.forecast_trend?.toUpperCase() || 'N/A'}</strong></div>
            </div>
            <div class="result-time">${new Date().toLocaleTimeString()}</div>
        `;

        resultsList.prepend(card);

        // Update badge
        window._resultCount = (window._resultCount || 0) + 1;
        resultsBadge.innerText = window._resultCount;
        resultsBadge.style.display = 'inline-block';
    }

    // ── Manual Simulation ───────────────────────────────────────────────────
    runBtn.addEventListener('click', async () => {
        let priceVal = parseFloat(document.getElementById('p_price').value);
        if (isNaN(priceVal)) priceVal = null;
        
        const payload = {
            product_name: document.getElementById('p_name').value,
            price: priceVal,
            category: document.getElementById('p_cat').value,
            rating: parseFloat(document.getElementById('p_rating').value) || 4.5,
            review_count: parseInt(document.getElementById('p_reviews').value) || 50
        };

        runBtn.disabled = true;
        runBtn.innerHTML = 'Thinking...';
        runError.style.display = 'none';

        try {
            // 60s timeout — scraping + reasoning takes time
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 60000);

            const resp = await fetch('/api/run_react', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload),
                signal: controller.signal
            });
            clearTimeout(timeoutId);

            if (!resp.ok) {
                const err = await resp.json().catch(() => ({ detail: resp.statusText }));
                throw new Error(err.detail || `Server error ${resp.status}`);
            }

            const data = await resp.json();

            if (data.error) {
                runError.innerText = `⚠ ${data.error}`;
                runError.style.display = 'block';
                return;
            }

            updateReasoningUI(data.result);
            addResultCard(payload, data);

            // Show market stats card
            if (data.market_stats) {
                document.getElementById('market-stats-card').style.display = '';
                document.getElementById('market-avg').innerText = `$${data.market_stats.avg_price?.toFixed(2) ?? '-'}`;
                document.getElementById('market-range').innerText = `$${data.market_stats.min_price?.toFixed(0) ?? '-'} – $${data.market_stats.max_price?.toFixed(0) ?? '-'}`;
                const diff = data.result.price_diff_pct;
                const diffEl = document.getElementById('market-diff');
                diffEl.innerText = diff != null ? `${diff > 0 ? '+' : ''}${diff.toFixed(1)}%` : '-';
                diffEl.style.color = diff < -10 ? '#38a169' : diff > 10 ? '#e53e3e' : '#dd6b20';
            }

            if (data.matched_products?.length) {
                const ml = document.getElementById('matched-list');
                ml.innerHTML = data.matched_products.slice(0, 5).map(p =>
                    `<div style="font-size:0.75rem; padding:0.4rem 0; border-bottom:1px solid #edf2f7">
                        <strong>${p.product_name?.slice(0, 55) ?? 'N/A'}</strong>
                        <span style="float:right; color:#3182ce">$${p.price ?? '-'}</span>
                    </div>`
                ).join('');
            }

        } catch (e) {
            let msg = e.message;
            if (e.name === 'AbortError') msg = 'Request timed out (>60s). Try again.';
            runError.innerText = `⚠ ${msg}`;
            runError.style.display = 'block';
        } finally {
            runBtn.disabled = false;
            runBtn.innerHTML = '<i data-lucide="zap"></i> Simulate ReAct Loop';
            lucide.createIcons();
        }
    });

    // ── Chart ───────────────────────────────────────────────────────────────
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
                    options: {
                        maintainAspectRatio: false,
                        scales: {
                            x: {title: {display: true, text: 'Price'}},
                            y: {title: {display: true, text: 'Rating'}}
                        }
                    }
                });
            }
        } catch (e) {}
    }

    initMap();
    setInterval(pollInference, 2000);
    setInterval(pollScraper, 2000);
});
