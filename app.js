// Sample data from the application data
const sampleFeedback = [
    {"text": "Great product! Really satisfied with the quality and service.", "sentiment": "positive", "confidence": 0.92, "themes": ["product quality", "service"], "emotions": ["joy", "satisfaction"]},
    {"text": "Terrible experience. Product didn't work as expected.", "sentiment": "negative", "confidence": 0.88, "themes": ["product quality", "expectations"], "emotions": ["anger", "disappointment"]},
    {"text": "The product is okay. Nothing special but does the job.", "sentiment": "neutral", "confidence": 0.65, "themes": ["product performance"], "emotions": ["neutrality"]},
    {"text": "Excellent customer support! They resolved my issue quickly.", "sentiment": "positive", "confidence": 0.95, "themes": ["customer service", "support"], "emotions": ["gratitude", "satisfaction"]},
    {"text": "Poor customer service. No one responded to my complaints.", "sentiment": "negative", "confidence": 0.91, "themes": ["customer service", "communication"], "emotions": ["frustration", "anger"]},
    {"text": "Amazing experience! Fast delivery and great packaging.", "sentiment": "positive", "confidence": 0.89, "themes": ["delivery", "packaging", "experience"], "emotions": ["joy", "excitement"]},
    {"text": "Website is confusing and difficult to navigate.", "sentiment": "negative", "confidence": 0.76, "themes": ["website usability", "navigation"], "emotions": ["frustration", "confusion"]},
    {"text": "Outstanding service. The team was very helpful and professional.", "sentiment": "positive", "confidence": 0.93, "themes": ["service quality", "team", "professionalism"], "emotions": ["appreciation", "trust"]}
];

const dashboardData = {
    "total_feedback": 1247,
    "sentiment_distribution": {
        "positive": 0.42,
        "negative": 0.23,
        "neutral": 0.35
    },
    "top_themes": [
        {"name": "Product Quality", "frequency": 156, "sentiment": "positive"},
        {"name": "Customer Service", "frequency": 134, "sentiment": "positive"}, 
        {"name": "Delivery Speed", "frequency": 98, "sentiment": "neutral"},
        {"name": "Website Usability", "frequency": 76, "sentiment": "negative"},
        {"name": "Pricing", "frequency": 65, "sentiment": "neutral"}
    ]
};

// Sentiment analysis keywords
const sentimentKeywords = {
    positive: [
        'great', 'excellent', 'amazing', 'fantastic', 'wonderful', 'outstanding', 'perfect', 'love', 'awesome', 'brilliant',
        'satisfied', 'happy', 'pleased', 'delighted', 'impressed', 'good', 'nice', 'best', 'superb', 'incredible',
        'helpful', 'professional', 'quick', 'fast', 'efficient', 'reliable', 'quality', 'recommend', 'thank', 'thanks'
    ],
    negative: [
        'terrible', 'awful', 'horrible', 'bad', 'worst', 'hate', 'disappointed', 'frustrating', 'annoying', 'useless',
        'poor', 'pathetic', 'disgusting', 'unacceptable', 'ridiculous', 'stupid', 'waste', 'problem', 'issue', 'broken',
        'slow', 'delayed', 'confusing', 'difficult', 'complicated', 'expensive', 'overpriced', 'rude', 'unhelpful'
    ],
    neutral: [
        'okay', 'average', 'normal', 'standard', 'typical', 'usual', 'fine', 'adequate', 'decent', 'reasonable',
        'acceptable', 'nothing', 'regular', 'basic', 'simple', 'plain', 'moderate', 'fair'
    ]
};

const themeKeywords = {
    'product quality': ['product', 'quality', 'item', 'goods', 'material', 'build', 'construction', 'durability'],
    'customer service': ['service', 'support', 'help', 'staff', 'team', 'representative', 'agent', 'customer', 'care'],
    'delivery': ['delivery', 'shipping', 'transport', 'arrived', 'package', 'mail', 'courier', 'dispatch'],
    'website': ['website', 'site', 'online', 'web', 'page', 'interface', 'navigation', 'browse'],
    'pricing': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'value', 'money', 'payment'],
    'packaging': ['packaging', 'box', 'wrapped', 'container', 'presentation'],
    'communication': ['communication', 'response', 'reply', 'contact', 'message', 'call', 'email'],
    'experience': ['experience', 'journey', 'process', 'overall', 'impression']
};

const emotionKeywords = {
    'joy': ['happy', 'joy', 'excited', 'thrilled', 'delighted', 'cheerful', 'elated'],
    'satisfaction': ['satisfied', 'content', 'pleased', 'fulfilled', 'gratified'],
    'anger': ['angry', 'mad', 'furious', 'outraged', 'livid', 'irritated'],
    'frustration': ['frustrated', 'annoyed', 'bothered', 'aggravated', 'exasperated'],
    'trust': ['trust', 'reliable', 'dependable', 'confident', 'secure'],
    'disappointment': ['disappointed', 'let down', 'unsatisfied', 'disheartened'],
    'gratitude': ['grateful', 'thankful', 'appreciate', 'thanks'],
    'excitement': ['excited', 'thrilled', 'enthusiastic', 'eager'],
    'confusion': ['confused', 'puzzled', 'unclear', 'lost', 'bewildered']
};

// Global function to scroll to demo (needed for onclick in HTML)
function scrollToDemo() {
    const demoSection = document.getElementById('demo');
    if (demoSection) {
        demoSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM Content Loaded - Initializing app');
    initializeApp();
});

function initializeApp() {
    console.log('Starting app initialization');
    setupEventListeners();
    setupNavigation();
    setupExampleButtons();
    
    // Initialize charts with a small delay to ensure DOM is ready
    setTimeout(() => {
        initializeDashboardCharts();
    }, 500);
}

function setupEventListeners() {
    console.log('Setting up event listeners');
    const analyzeBtn = document.getElementById('analyze-btn');
    const feedbackText = document.getElementById('feedback-text');
    
    if (analyzeBtn) {
        console.log('Found analyze button, adding click listener');
        analyzeBtn.addEventListener('click', function(e) {
            e.preventDefault();
            analyzeSentiment();
        });
    } else {
        console.error('Analyze button not found');
    }
    
    // Allow enter key to trigger analysis
    if (feedbackText) {
        feedbackText.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                e.preventDefault();
                analyzeSentiment();
            }
        });
    } else {
        console.error('Feedback text element not found');
    }
}

function setupNavigation() {
    console.log('Setting up navigation');
    
    // Setup navigation links
    const navLinks = document.querySelectorAll('.nav__link');
    console.log('Found nav links:', navLinks.length);
    
    navLinks.forEach((link, index) => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            console.log('Navigation clicked:', targetId);
            
            const targetSection = document.querySelector(targetId);
            if (targetSection) {
                console.log('Scrolling to section:', targetId);
                targetSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            } else {
                console.error('Target section not found:', targetId);
            }
        });
    });

    // Setup Try Live Demo button - find it more specifically
    const heroButtons = document.querySelectorAll('.hero .btn--primary');
    console.log('Found hero buttons:', heroButtons.length);
    
    heroButtons.forEach(button => {
        if (button.textContent.includes('Try Live Demo')) {
            console.log('Found Try Live Demo button');
            button.addEventListener('click', function(e) {
                e.preventDefault();
                console.log('Try Live Demo clicked');
                scrollToDemo();
            });
        }
    });
}

function setupExampleButtons() {
    console.log('Setting up example buttons');
    const exampleButtons = document.querySelectorAll('.example-btn');
    const feedbackText = document.getElementById('feedback-text');
    
    console.log('Found example buttons:', exampleButtons.length);
    
    if (feedbackText) {
        exampleButtons.forEach((button, index) => {
            button.addEventListener('click', function(e) {
                e.preventDefault();
                const exampleText = this.getAttribute('data-text');
                console.log('Example button clicked:', index, exampleText);
                feedbackText.value = exampleText;
                feedbackText.focus();
            });
        });
    }
}

async function analyzeSentiment() {
    console.log('Starting sentiment analysis');
    
    const feedbackText = document.getElementById('feedback-text');
    const sourceType = document.getElementById('source-type');
    
    if (!feedbackText || !feedbackText.value.trim()) {
        alert('Please enter some customer feedback to analyze.');
        return;
    }
    
    const text = feedbackText.value.trim();
    const source = sourceType ? sourceType.value : 'email';
    
    console.log('Analyzing text:', text.substring(0, 50) + '...');
    
    // Hide results and show processing pipeline
    const resultsSection = document.getElementById('demo-results');
    const pipelineSection = document.getElementById('processing-pipeline');
    
    console.log('Results section found:', !!resultsSection);
    console.log('Pipeline section found:', !!pipelineSection);
    
    if (resultsSection) {
        resultsSection.style.display = 'none';
        console.log('Hidden results section');
    }
    if (pipelineSection) {
        pipelineSection.style.display = 'block';
        console.log('Showed pipeline section');
    }
    
    // Reset and run pipeline
    resetPipelineSteps();
    
    // Disable the analyze button
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Analyzing...';
    }
    
    try {
        // Run the multi-agent pipeline
        await runMultiAgentPipeline(text, source);
        
        // Analyze the text and show results
        const analysisResult = performSentimentAnalysis(text);
        console.log('Analysis result:', analysisResult);
        showAnalysisResults(analysisResult);
    } catch (error) {
        console.error('Analysis error:', error);
        alert('An error occurred during analysis. Please try again.');
    } finally {
        // Re-enable the button
        if (analyzeBtn) {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze Sentiment';
        }
    }
}

function resetPipelineSteps() {
    console.log('Resetting pipeline steps');
    const steps = document.querySelectorAll('.pipeline-step');
    console.log('Found pipeline steps:', steps.length);
    
    steps.forEach((step, index) => {
        step.classList.remove('active', 'completed');
        const status = step.querySelector('.step-status');
        if (status) {
            status.textContent = '⏳ Waiting';
            status.style.color = '';
        }
        console.log('Reset step', index);
    });
}

async function runMultiAgentPipeline(text, sourceType) {
    console.log('Running multi-agent pipeline');
    const steps = ['preprocessing', 'sentiment', 'insight', 'visualization'];
    
    for (let i = 0; i < steps.length; i++) {
        const stepName = steps[i];
        console.log('Processing step:', stepName);
        
        const stepElement = document.querySelector(`[data-step="${stepName}"]`);
        
        if (!stepElement) {
            console.error('Step element not found:', stepName);
            continue;
        }
        
        const statusElement = stepElement.querySelector('.step-status');
        
        // Mark as active
        stepElement.classList.add('active');
        if (statusElement) {
            statusElement.textContent = '⚙️ Processing...';
        }
        console.log('Marked step as active:', stepName);
        
        // Simulate processing time
        const processingTime = 1200 + Math.random() * 800;
        await new Promise(resolve => setTimeout(resolve, processingTime));
        
        // Mark as completed
        stepElement.classList.remove('active');
        stepElement.classList.add('completed');
        if (statusElement) {
            statusElement.textContent = '✅ Complete';
        }
        console.log('Marked step as completed:', stepName);
        
        // Show intermediate results for some steps
        if (stepName === 'sentiment') {
            updateStepDescription(stepElement, 'Sentiment classification completed with high confidence');
        } else if (stepName === 'insight') {
            updateStepDescription(stepElement, 'Extracted themes and emotional patterns');
        }
    }
    console.log('Pipeline completed');
}

function updateStepDescription(stepElement, newDescription) {
    const descriptionElement = stepElement.querySelector('p');
    if (descriptionElement) {
        descriptionElement.textContent = newDescription;
    }
}

function performSentimentAnalysis(text) {
    console.log('Performing sentiment analysis');
    const lowercaseText = text.toLowerCase();
    const words = lowercaseText.split(/\s+/);
    
    // Calculate sentiment scores
    let positiveScore = 0;
    let negativeScore = 0;
    let neutralScore = 0;
    
    words.forEach(word => {
        if (sentimentKeywords.positive.includes(word)) {
            positiveScore += 1;
        } else if (sentimentKeywords.negative.includes(word)) {
            negativeScore += 1;
        } else if (sentimentKeywords.neutral.includes(word)) {
            neutralScore += 0.5;
        }
    });
    
    console.log('Sentiment scores:', { positive: positiveScore, negative: negativeScore, neutral: neutralScore });
    
    // Determine overall sentiment
    let sentiment, confidence;
    if (positiveScore > negativeScore && positiveScore > neutralScore) {
        sentiment = 'positive';
        confidence = Math.min(0.95, 0.7 + (positiveScore / words.length) * 1.5);
    } else if (negativeScore > positiveScore && negativeScore > neutralScore) {
        sentiment = 'negative';
        confidence = Math.min(0.95, 0.7 + (negativeScore / words.length) * 1.5);
    } else {
        sentiment = 'neutral';
        confidence = Math.min(0.8, 0.55 + Math.random() * 0.2);
    }
    
    // Extract themes
    const themes = extractThemes(lowercaseText);
    
    // Detect emotions
    const emotions = detectEmotions(lowercaseText);
    
    // Calculate processing time
    const processingTime = (1.5 + Math.random() * 1.0).toFixed(1);
    
    const result = {
        sentiment,
        confidence: Math.round(confidence * 100),
        themes,
        emotions,
        processingTime
    };
    
    console.log('Analysis complete:', result);
    return result;
}

function extractThemes(text) {
    const detectedThemes = [];
    
    Object.keys(themeKeywords).forEach(theme => {
        const keywords = themeKeywords[theme];
        const matches = keywords.filter(keyword => text.includes(keyword));
        if (matches.length > 0) {
            detectedThemes.push(theme);
        }
    });
    
    // If no themes detected, add generic ones based on common words
    if (detectedThemes.length === 0) {
        if (text.includes('product') || text.includes('item')) {
            detectedThemes.push('product quality');
        }
        if (text.includes('service') || text.includes('support')) {
            detectedThemes.push('customer service');
        }
        if (detectedThemes.length === 0) {
            detectedThemes.push('general feedback');
        }
    }
    
    return detectedThemes.slice(0, 4); // Limit to 4 themes
}

function detectEmotions(text) {
    const detectedEmotions = [];
    
    Object.keys(emotionKeywords).forEach(emotion => {
        const keywords = emotionKeywords[emotion];
        const matches = keywords.filter(keyword => text.includes(keyword));
        if (matches.length > 0) {
            detectedEmotions.push(emotion);
        }
    });
    
    // If no emotions detected, infer from sentiment
    if (detectedEmotions.length === 0) {
        if (text.includes('good') || text.includes('great') || text.includes('excellent')) {
            detectedEmotions.push('satisfaction');
        } else if (text.includes('bad') || text.includes('terrible') || text.includes('awful')) {
            detectedEmotions.push('frustration');
        } else {
            detectedEmotions.push('neutrality');
        }
    }
    
    return detectedEmotions.slice(0, 3); // Limit to 3 emotions
}

function showAnalysisResults(result) {
    console.log('Showing analysis results');
    
    // Hide processing pipeline and show results
    const pipelineSection = document.getElementById('processing-pipeline');
    const resultsSection = document.getElementById('demo-results');
    
    if (pipelineSection) {
        pipelineSection.style.display = 'none';
        console.log('Hidden pipeline section');
    }
    if (resultsSection) {
        resultsSection.style.display = 'block';
        console.log('Showed results section');
    }
    
    // Update sentiment result
    const sentimentLabel = document.getElementById('sentiment-label');
    const confidenceScore = document.getElementById('confidence-score');
    
    if (sentimentLabel) {
        sentimentLabel.textContent = result.sentiment.charAt(0).toUpperCase() + result.sentiment.slice(1);
        sentimentLabel.className = `sentiment-label ${result.sentiment}`;
        console.log('Updated sentiment label:', result.sentiment);
    }
    
    if (confidenceScore) {
        confidenceScore.textContent = `${result.confidence}%`;
        console.log('Updated confidence score:', result.confidence);
    }
    
    // Update themes
    const themesList = document.getElementById('themes-list');
    if (themesList) {
        themesList.innerHTML = '';
        result.themes.forEach(theme => {
            const themeTag = document.createElement('span');
            themeTag.className = 'theme-tag';
            themeTag.textContent = theme;
            themesList.appendChild(themeTag);
        });
        console.log('Updated themes:', result.themes);
    }
    
    // Update emotions
    const emotionsList = document.getElementById('emotions-list');
    if (emotionsList) {
        emotionsList.innerHTML = '';
        result.emotions.forEach(emotion => {
            const emotionTag = document.createElement('span');
            emotionTag.className = 'emotion-tag';
            emotionTag.textContent = emotion;
            emotionsList.appendChild(emotionTag);
        });
        console.log('Updated emotions:', result.emotions);
    }
    
    // Update processing time
    const processingTimeElement = document.getElementById('processing-time');
    if (processingTimeElement) {
        processingTimeElement.textContent = `${result.processingTime}s`;
        console.log('Updated processing time:', result.processingTime);
    }
    
    // Scroll to results
    setTimeout(() => {
        if (resultsSection) {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
            console.log('Scrolled to results');
        }
    }, 300);
}

function initializeDashboardCharts() {
    console.log('Initializing dashboard charts');
    
    // Wait a bit more to ensure DOM is ready and Chart.js is loaded
    setTimeout(() => {
        if (typeof Chart !== 'undefined') {
            initializeSentimentChart();
            initializeThemesChart();
        } else {
            console.error('Chart.js not loaded');
        }
    }, 1000);
}

function initializeSentimentChart() {
    const ctx = document.getElementById('sentiment-chart');
    if (!ctx) {
        console.error('Sentiment chart canvas not found');
        return;
    }
    
    console.log('Creating sentiment chart');
    
    new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Positive', 'Neutral', 'Negative'],
            datasets: [{
                data: [
                    dashboardData.sentiment_distribution.positive * 100,
                    dashboardData.sentiment_distribution.neutral * 100,
                    dashboardData.sentiment_distribution.negative * 100
                ],
                backgroundColor: [
                    '#1FB8CD',
                    '#FFC185', 
                    '#B4413C'
                ],
                borderWidth: 2,
                borderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true,
                        font: {
                            size: 14
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.label + ': ' + context.parsed.toFixed(1) + '%';
                        }
                    }
                }
            }
        }
    });
}

function initializeThemesChart() {
    const ctx = document.getElementById('themes-chart');
    if (!ctx) {
        console.error('Themes chart canvas not found');
        return;
    }
    
    console.log('Creating themes chart');
    
    const themes = dashboardData.top_themes;
    const labels = themes.map(theme => theme.name);
    const data = themes.map(theme => theme.frequency);
    const colors = ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F'];
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Frequency',
                data: data,
                backgroundColor: colors,
                borderColor: colors,
                borderWidth: 1
            }]
        },
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
                            return 'Frequency: ' + context.parsed.y;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    },
                    ticks: {
                        font: {
                            size: 12
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            size: 12
                        },
                        maxRotation: 45
                    }
                }
            }
        }
    });
}

// Export functions for potential external use
window.AuralyTicaDemo = {
    analyzeSentiment,
    performSentimentAnalysis,
    scrollToDemo
};