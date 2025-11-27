const API_BASE_URL = 'http://localhost:8000';

let chart = null;
let selectedPoints = [];
let allEmbeddings = [];

const textInput = document.getElementById('textInput');
const submitBtn = document.getElementById('submitBtn');
const resetBtn = document.getElementById('resetBtn');
const statusDiv = document.getElementById('status');
const embeddingDisplay = document.getElementById('embeddingDisplay');
const similarityDisplay = document.getElementById('similarityDisplay');
const word1Display = document.getElementById('word1Display');
const word2Display = document.getElementById('word2Display');
const similarityValue = document.getElementById('similarityValue');
const clearSelectionBtn = document.getElementById('clearSelectionBtn');

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
    const displayText = `[${embedding.slice(0, 30).map(v => v.toFixed(4)).join(', ')}... ] (${embedding.length} dimensions)`;
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

        allEmbeddings = data.embeddings;
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
                label: e.text,
                id: e.id
            })),
            backgroundColor: embeddings.map(e =>
                selectedPoints.includes(e.id)
                    ? 'rgba(255, 99, 71, 0.8)'
                    : 'rgba(102, 126, 234, 0.6)'
            ),
            borderColor: embeddings.map(e =>
                selectedPoints.includes(e.id)
                    ? 'rgba(255, 99, 71, 1)'
                    : 'rgba(102, 126, 234, 1)'
            ),
            borderWidth: 2,
            pointRadius: embeddings.map(e =>
                selectedPoints.includes(e.id) ? 12 : 8
            ),
            pointHoverRadius: 12,
        }]
    };

    const config = {
        type: 'scatter',
        data: chartData,
        options: {
            animation: false,
            responsive: true,
            maintainAspectRatio: false,
            onClick: handleChartClick,
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
                    min: -0.4,
                    max: 0.6,
                    title: {
                        display: true,
                        text: 'Dimension 1'
                    }
                },
                y: {
                    min: -0.6,
                    max: 0.8,
                    title: {
                        display: true,
                        text: 'Dimension 2'
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

function handleChartClick(event, activeElements) {
    if (activeElements.length === 0) return;

    const clickedIndex = activeElements[0].index;
    const clickedPoint = allEmbeddings[clickedIndex];

    if (!clickedPoint) return;

    // If point is already selected, deselect it
    if (selectedPoints.includes(clickedPoint.id)) {
        selectedPoints = selectedPoints.filter(id => id !== clickedPoint.id);
        updateChart(allEmbeddings);

        // Hide similarity if less than 2 points selected
        if (selectedPoints.length < 2) {
            similarityDisplay.classList.remove('active');
        }
        return;
    }

    // Add point to selection
    selectedPoints.push(clickedPoint.id);

    // Limit to 2 points
    if (selectedPoints.length > 2) {
        selectedPoints.shift(); // Remove first point
    }

    // Update chart with new selection
    updateChart(allEmbeddings);

    // If 2 points selected, fetch and display similarity
    if (selectedPoints.length === 2) {
        fetchAndDisplaySimilarity(selectedPoints[0], selectedPoints[1]);
    }
}

async function fetchAndDisplaySimilarity(id1, id2) {
    try {
        const response = await fetch(`${API_BASE_URL}/similarity?id1=${id1}&id2=${id2}`);

        if (!response.ok) {
            throw new Error('Failed to compute similarity');
        }

        const data = await response.json();

        word1Display.textContent = data.word1;
        word2Display.textContent = data.word2;
        similarityValue.textContent = data.cosine_similarity.toFixed(4);
        similarityDisplay.classList.add('active');

    } catch (error) {
        console.error('Error fetching similarity:', error);
        showStatus(`Similarity error: ${error.message}`, 'error');
    }
}

function clearSelection() {
    selectedPoints = [];
    similarityDisplay.classList.remove('active');
    updateChart(allEmbeddings);
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

async function resetData() {
    if (!confirm('Are you sure you want to clear all embeddings? This cannot be undone.')) {
        return;
    }

    resetBtn.disabled = true;
    showStatus('Clearing all data...', 'loading');

    try {
        const response = await fetch(`${API_BASE_URL}/embeddings`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            throw new Error('Failed to clear embeddings');
        }

        embeddingDisplay.textContent = '';
        clearSelection();

        if (chart) {
            chart.destroy();
            chart = null;
        }

        showStatus('All data cleared successfully!', 'success');

    } catch (error) {
        showStatus(`Error: ${error.message}`, 'error');
        console.error('Error:', error);
    } finally {
        resetBtn.disabled = false;
    }
}

submitBtn.addEventListener('click', submitText);
resetBtn.addEventListener('click', resetData);
clearSelectionBtn.addEventListener('click', clearSelection);

textInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        submitText();
    }
});

window.addEventListener('load', () => {
    showStatus('Ready! Enter text to begin.', 'info');
});
