// AuralyTica Tel AI Dashboard JavaScript

// Global variables
let dashboardData = null;
let refreshInterval = null;

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
});

/**
 * Initialize the dashboard
 */
function initializeDashboard() {
    loadDashboardData();
    setupEventListeners();
    startAutoRefresh();
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add loading states to forms
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
            }
        });
    });
}

/**
 * Load dashboard data from API
 */
function loadDashboardData() {
    showLoadingState();

    fetch('/api/dashboard_data')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            dashboardData = data;
            updateDashboard(data);
            hideLoadingState();
        })
        .catch(error => {
            console.error('Error loading dashboard data:', error);
            showErrorState(error.message);
        });
}

/**
 * Update dashboard with new data
 */
function updateDashboard(data) {
    try {
        updateKeyMetrics(data.key_metrics);
        updateTopThemes(data.top_themes);
        updateEmotionSummary(data.emotion_summary);
        updateProcessingStats(data.processing_stats);
        createCharts(data);
        updateAlerts(data.alerts_count);
    } catch (error) {
        console.error('Error updating dashboard:', error);
        showErrorState('Failed to update dashboard components');
    }
}

/**
 * Update key metrics cards
 */
function updateKeyMetrics(metrics) {
    const elements = {
        totalFeedback: document.getElementById('totalFeedback'),
        overallSentiment: document.getElementById('overallSentiment'),
        avgConfidence: document.getElementById('avgConfidence'),
        alertsCount: document.getElementById('alertsCount')
    };

    if (elements.totalFeedback) {
        elements.totalFeedback.textContent = formatNumber(metrics.total_feedback);
    }

    if (elements.overallSentiment) {
        const sentiment = metrics.overall_sentiment;
        elements.overallSentiment.textContent = sentiment.toUpperCase();
        elements.overallSentiment.className = `badge bg-${getSentimentColor(sentiment)}`;
    }

    if (elements.avgConfidence) {
        elements.avgConfidence.textContent = `${(metrics.average_confidence * 100).toFixed(0)}%`;
    }
}

/**
 * Update top themes section
 */
function updateTopThemes(themes) {
    const container = document.getElementById('topThemes');
    if (!container || !themes) return;

    let html = '';
    themes.slice(0, 5).forEach(theme => {
        html += `
            <div class="theme-item">
                <div class="d-flex justify-content-between align-items-center">
                    <span class="theme-name">${theme.name}</span>
                    <div>
                        <span class="badge bg-${getSentimentColor(theme.sentiment)} me-2">
                            ${theme.sentiment}
                        </span>
                        <span class="theme-frequency">${theme.frequency}</span>
                    </div>
                </div>
            </div>
        `;
    });

    container.innerHTML = html || '<p class="text-muted">No themes available</p>';
}

/**
 * Update emotion summary
 */
function updateEmotionSummary(emotions) {
    const container = document.getElementById('emotionSummary');
    if (!container || !emotions) return;

    let html = `
        <div class="emotion-summary">
            <div class="mb-3">
                <strong>Dominant Emotion:</strong> 
                <span class="badge bg-info">${emotions.dominant_emotion || 'None'}</span>
            </div>
            <div class="mb-3">
                <strong>Total Detected:</strong> ${emotions.total_emotions_detected || 0}
            </div>
    `;

    if (emotions.emotion_distribution) {
        html += '<div class="emotion-bars">';
        Object.entries(emotions.emotion_distribution).slice(0, 3).forEach(([emotion, data]) => {
            const percentage = (data.percentage * 100).toFixed(1);
            html += `
                <div class="emotion-bar mb-2">
                    <div class="d-flex justify-content-between">
                        <span>${capitalize(emotion)}</span>
                        <span>${percentage}%</span>
                    </div>
                    <div class="progress" style="height: 5px;">
                        <div class="progress-bar bg-info" style="width: ${percentage}%"></div>
                    </div>
                </div>
            `;
        });
        html += '</div>';
    }

    html += '</div>';
    container.innerHTML = html;
}

/**
 * Update processing statistics
 */
function updateProcessingStats(stats) {
    if (!stats) return;

    const elements = {
        lastUpdated: document.getElementById('lastUpdated'),
        processingTime: document.getElementById('processingTime'),
        successRate: document.getElementById('successRate')
    };

    if (elements.lastUpdated && stats.last_updated) {
        elements.lastUpdated.textContent = formatDateTime(stats.last_updated);
    }

    if (elements.processingTime && stats.processing_time) {
        elements.processingTime.textContent = `${stats.processing_time}s`;
    }

    if (elements.successRate && stats.success_rate) {
        elements.successRate.textContent = `${(stats.success_rate * 100).toFixed(1)}%`;
    }
}

/**
 * Create charts using Chart.js or Plotly
 */
function createCharts(data) {
    createSentimentChart(data.key_metrics.sentiment_distribution);
    createThemesChart(data.top_themes);
    createTrendChart(data.recent_feedback);
}

/**
 * Create sentiment distribution chart
 */
function createSentimentChart(sentimentData) {
    const chartElement = document.getElementById('sentimentChart');
    if (!chartElement || !sentimentData) return;

    // Check if Plotly is available
    if (typeof Plotly !== 'undefined') {
        const data = [{
            values: Object.values(sentimentData),
            labels: Object.keys(sentimentData).map(capitalize),
            type: 'pie',
            marker: {
                colors: ['#28a745', '#dc3545', '#6c757d']
            },
            textinfo: 'label+percent',
            hovertemplate: '%{label}<br>%{percent}<br>%{value} entries<extra></extra>'
        }];

        const layout = {
            title: '',
            showlegend: true,
            height: 300,
            margin: {t: 30, b: 30, l: 30, r: 30},
            font: {
                family: '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif',
                size: 12
            }
        };

        const config = {
            responsive: true,
            displayModeBar: false
        };

        Plotly.newPlot('sentimentChart', data, layout, config);
    }
}

/**
 * Create themes chart
 */
function createThemesChart(themesData) {
    const chartElement = document.getElementById('themesChart');
    if (!chartElement || !themesData || themesData.length === 0) {
        if (chartElement) {
            chartElement.innerHTML = '<p class="text-muted text-center py-4">No themes data available</p>';
        }
        return;
    }

    if (typeof Plotly !== 'undefined') {
        const data = [{
            x: themesData.map(t => t.name),
            y: themesData.map(t => t.frequency),
            type: 'bar',
            marker: {
                color: themesData.map(t => getSentimentColorHex(t.sentiment)),
                opacity: 0.8
            },
            hovertemplate: '%{x}<br>Frequency: %{y}<br>Sentiment: %{text}<extra></extra>',
            text: themesData.map(t => t.sentiment)
        }];

        const layout = {
            title: '',
            height: 300,
            margin: {t: 30, b: 80, l: 50, r: 30},
            xaxis: {
                tickangle: -45,
                tickfont: {size: 10}
            },
            yaxis: {
                title: 'Frequency'
            },
            font: {
                family: '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif',
                size: 12
            }
        };

        const config = {
            responsive: true,
            displayModeBar: false
        };

        Plotly.newPlot('themesChart', data, layout, config);
    }
}

/**
 * Update alerts section
 */
function updateAlerts(alertsCount) {
    const alertsCountElement = document.getElementById('alertsCount');
    const alertsListElement = document.getElementById('alertsList');

    if (alertsCountElement) {
        alertsCountElement.textContent = alertsCount || 0;
    }

    if (alertsListElement) {
        if (alertsCount && alertsCount > 0) {
            alertsListElement.innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    You have <strong>${alertsCount}</strong> active alert${alertsCount > 1 ? 's' : ''} that require attention.
                    <button class="btn btn-sm btn-outline-warning ms-2" onclick="viewDetailedAlerts()">
                        View Details
                    </button>
                </div>
            `;
        } else {
            alertsListElement.innerHTML = `
                <div class="alert alert-success">
                    <i class="fas fa-check-circle me-2"></i>
                    No active alerts. All systems are running smoothly.
                </div>
            `;
        }
    }
}

/**
 * Show loading state
 */
function showLoadingState() {
    const loadingElements = document.querySelectorAll('.loading-placeholder');
    loadingElements.forEach(element => {
        element.innerHTML = `
            <div class="text-center py-4">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-2">Loading...</p>
            </div>
        `;
    });
}

/**
 * Hide loading state
 */
function hideLoadingState() {
    // Loading states are replaced by actual content
    console.log('Dashboard loaded successfully');
}

/**
 * Show error state
 */
function showErrorState(message) {
    const errorHtml = `
        <div class="alert alert-danger">
            <i class="fas fa-exclamation-circle me-2"></i>
            <strong>Error:</strong> ${message}
            <br>
            <button class="btn btn-sm btn-outline-danger mt-2" onclick="loadDashboardData()">
                <i class="fas fa-retry me-1"></i>Retry
            </button>
        </div>
    `;

    const errorContainers = document.querySelectorAll('.loading-placeholder');
    errorContainers.forEach(container => {
        container.innerHTML = errorHtml;
    });
}

/**
 * Start auto-refresh
 */
function startAutoRefresh() {
    // Refresh every 5 minutes
    refreshInterval = setInterval(() => {
        loadDashboardData();
    }, 5 * 60 * 1000);
}

/**
 * Stop auto-refresh
 */
function stopAutoRefresh() {
    if (refreshInterval) {
        clearInterval(refreshInterval);
        refreshInterval = null;
    }
}

/**
 * Manual refresh function
 */
function refreshDashboard() {
    loadDashboardData();
}

/**
 * Utility functions
 */

function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    }
    if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

function formatDateTime(dateString) {
    try {
        const date = new Date(dateString);
        return date.toLocaleString();
    } catch (error) {
        return 'N/A';
    }
}

function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
}

function getSentimentColor(sentiment) {
    switch (sentiment.toLowerCase()) {
        case 'positive': return 'success';
        case 'negative': return 'danger';
        case 'neutral': return 'secondary';
        default: return 'secondary';
    }
}

function getSentimentColorHex(sentiment) {
    switch (sentiment.toLowerCase()) {
        case 'positive': return '#28a745';
        case 'negative': return '#dc3545';
        case 'neutral': return '#6c757d';
        default: return '#6c757d';
    }
}

function viewDetailedAlerts() {
    // This could open a modal or navigate to a detailed alerts page
    alert('Detailed alerts view would be implemented here');
}

// Export functions for global access
window.refreshDashboard = refreshDashboard;
window.viewDetailedAlerts = viewDetailedAlerts;
window.loadDashboardData = loadDashboardData;
