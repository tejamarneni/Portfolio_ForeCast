let chartInstance = null;

function addRow(ticker = '', weight = '') {
  const container = document.getElementById('allocations-container');
  const row = document.createElement('div');
  row.className = 'row';
  row.innerHTML = `
    <input type="text" class="ticker-input" placeholder="Ticker" value="${ticker}">
    <input type="number" step="0.05" min="0" max="1" class="weight-input" placeholder="Weight" value="${weight}">
    <button type="button" class="btn-del" onclick="removeRow(this)">✕</button>
  `;
  container.appendChild(row);
  updateWeightFeedback();
}

function removeRow(button) {
  const container = document.getElementById('allocations-container');
  if (container.querySelectorAll('.row').length > 1) {
    button.closest('.row').remove();
    updateWeightFeedback();
  } else {
    showError("Portfolio must contain at least one asset.");
  }
}

function updateWeightFeedback() {
  const weights = Array.from(document.querySelectorAll('.weight-input'))
    .map(w => parseFloat(w.value) || 0.0);
  const total = weights.reduce((acc, curr) => acc + curr, 0);
  const statusElement = document.getElementById('weight-status');

  statusElement.textContent = `Total Allocation: ${(total * 100).toFixed(1)}%`;
  statusElement.style.color = (total >= 0.99 && total <= 1.01) ? '#16a34a' : '#dc2626';
}

function showError(msg) {
  const errBox = document.getElementById('error-message');
  if (msg) {
    errBox.textContent = msg;
    errBox.classList.remove('hidden');
  } else {
    errBox.textContent = '';
    errBox.classList.add('hidden');
  }
}

async function fetchForecast() {
  showError(null);

  const initialAmount = parseFloat(document.getElementById('initial-investment').value);
  const horizonDays = parseInt(document.getElementById('horizon-days').value, 10);
  const tickerInputs = document.querySelectorAll('.ticker-input');
  const weightInputs = document.querySelectorAll('.weight-input');

  const portfolio = [];
  tickerInputs.forEach((tInput, idx) => {
    const symbol = tInput.value.trim().toUpperCase();
    const weight = parseFloat(weightInputs[idx].value);
    if (symbol && !isNaN(weight)) {
      portfolio.push({ ticker: symbol, weight: weight });
    }
  });

  if (portfolio.length === 0) {
    showError("Please enter at least one valid ticker and allocation.");
    return;
  }

  const totalWeight = portfolio.reduce((sum, item) => sum + item.weight, 0.0);
  if (totalWeight < 0.99 || totalWeight > 1.01) {
    showError(`Total weight must equal 1.0 (Current: ${totalWeight.toFixed(2)})`);
    return;
  }

  const payload = {
    portfolio: portfolio,
    initial_investment: initialAmount,
    horizon_days: horizonDays,
    confidence_interval: 0.90,
    model_type: "geometric_brownian_motion"
  };

  const runBtn = document.getElementById('run-btn');
  runBtn.disabled = true;
  runBtn.textContent = 'Simulating...';

  try {
    const response = await fetch("/forecast/portfolio", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }));
      showError(err.detail || `Server returned status: ${response.status}`);
      return;
    }

    const data = await response.json();
    renderMetrics(data.metrics, data.final_expected_value);
    renderChart(data.trajectory);
  } catch (err) {
    console.error("Fetch failure:", err);
    showError(`Client error: ${err.message}`);
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = 'Run Simulation';
  }
}

function renderMetrics(metrics, finalValue) {
  document.getElementById('metrics-card').classList.remove('hidden');
  document.getElementById('metric-final-val').textContent = `$${finalValue.toLocaleString()}`;
  document.getElementById('metric-drift').textContent = `${(metrics.expected_annual_return * 100).toFixed(1)}%`;
  document.getElementById('metric-vol').textContent = `${(metrics.annualized_volatility * 100).toFixed(1)}%`;
  document.getElementById('metric-var').textContent = `$${metrics.var_95.toLocaleString()}`;
}

function renderChart(trajectory) {
  const labels = trajectory.map(p => `Day ${p.day}`);
  const medians = trajectory.map(p => p.median_value);
  const lowers = trajectory.map(p => p.lower_bound);
  const uppers = trajectory.map(p => p.upper_bound);

  if (chartInstance) chartInstance.destroy();

  const ctx = document.getElementById('forecastChart').getContext('2d');
  chartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Upper Bound (95%)',
          data: uppers,
          borderColor: 'rgba(59, 130, 246, 0.25)',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          fill: '+1',
          pointRadius: 0
        },
        {
          label: 'Lower Bound (5%)',
          data: lowers,
          borderColor: 'rgba(59, 130, 246, 0.25)',
          fill: false,
          pointRadius: 0
        },
        {
          label: 'Median Projection',
          data: medians,
          borderColor: '#2563eb',
          borderWidth: 2.5,
          fill: false,
          pointRadius: 2
        }
      ]
    },
    options: {
      responsive: true,
      scales: {
        y: { ticks: { callback: v => '$' + v.toLocaleString() } }
      }
    }
  });
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('allocations-container').addEventListener('input', (e) => {
    if (e.target.classList.contains('weight-input')) updateWeightFeedback();
  });
  updateWeightFeedback();
});