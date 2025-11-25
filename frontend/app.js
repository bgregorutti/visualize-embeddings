const API_BASE_URL = 'http://localhost:8000';

let chart = null;

const textInput = document.getElementById('textInput');
const submitBtn = document.getElementById('submitBtn');
const statusDiv = document.getElementById('status');
const embeddingDisplay = document.getElementById('embeddingDisplay');

async function submitText() {
    const text = textInput.value.trim();

    if (!text) {
        showStatus('Please enter some text', 'error');
        return;
    }

    submitBtn.disabled = true;
    showStatus('Computing embedding...', 'loading');

    try {
        const response = await fetch(`${API_BASE_URL}/embed`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to compute embedding');
        }

        const data = await response.json();

        displayEmbedding(data.embedding);
        showStatus(`Embedding computed successfully! (ID: ${data.id.substring(0, 8)}...)`, 'success');

        textInput.value = '';

        await updateVisualization();

    } catch (error) {
        showStatus(`Error: ${error.message}`, 'error');
        console.error('Error:', error);
    } finally {
        submitBtn.disabled = false;
    }
}

function displayEmbedding(embedding) {
    const displayText = `[${embedding.slice(0, 10).map(v => v.toFixed(4)).join(', ')}... ] (${embedding.length} dimensions)`;
    embeddingDisplay.textContent = displayText;
}

async function updateVisualization() {
    try {
        const response = await fetch(`${API_BASE_URL}/embeddings`);

        if (!response.ok) {
            throw new Error('Failed to fetch embeddings');
        }

        const data = await response.json();

        if (data.count === 0) {
            return;
        }

        updateChart(data.embeddings);

    } catch (error) {
        console.error('Error updating visualization:', error);
        showStatus(`Visualization error: ${error.message}`, 'error');
    }
}

function updateChart(embeddings) {
    const ctx = document.getElementById('embeddingChart').getContext('2d');

    const chartData = {
        datasets: [{
            label: 'Embeddings',
            data: embeddings.map(e => ({
                x: e.x,
                y: e.y,
                label: e.text
            })),
            backgroundColor: 'rgba(102, 126, 234, 0.6)',
            borderColor: 'rgba(102, 126, 234, 1)',
            borderWidth: 2,
            pointRadius: 8,
            pointHoverRadius: 12,
        }]
    };

    const config = {
        type: 'scatter',
        data: chartData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const point = context.raw;
                            return `${point.label} (${point.x.toFixed(2)}, ${point.y.toFixed(2)})`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: {
                        display: true,
                        text: 'UMAP Dimension 1'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'UMAP Dimension 2'
                    }
                }
            }
        },
        plugins: [{
            afterDatasetsDraw: function(chart) {
                const ctx = chart.ctx;
                chart.data.datasets.forEach((dataset, i) => {
                    const meta = chart.getDatasetMeta(i);
                    meta.data.forEach((point, index) => {
                        const data = dataset.data[index];
                        ctx.fillStyle = '#333';
                        ctx.font = '12px Arial';
                        ctx.textAlign = 'center';
                        ctx.fillText(data.label, point.x, point.y - 15);
                    });
                });
            }
        }]
    };

    if (chart) {
        chart.destroy();
    }

    chart = new Chart(ctx, config);
}

function showStatus(message, type = 'info') {
    statusDiv.textContent = message;
    statusDiv.className = 'status';

    if (type === 'error') {
        statusDiv.classList.add('error');
    } else if (type === 'loading') {
        statusDiv.classList.add('loading');
    }

    if (type === 'success') {
        setTimeout(() => {
            statusDiv.textContent = '';
        }, 3000);
    }
}

submitBtn.addEventListener('click', submitText);

textInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        submitText();
    }
});

window.addEventListener('load', () => {
    showStatus('Ready! Enter text to begin.', 'info');
});
