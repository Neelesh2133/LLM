/**
 * GPT-2 Studio — Frontend Application
 * Handles mode switching, API calls, chat rendering, and classification display.
 */

// ============================================================
// DOM Elements
// ============================================================
const navItems = document.querySelectorAll('.nav-item[data-mode]');
const panels = {
    generate: document.getElementById('panel-generate'),
    classify: document.getElementById('panel-classify'),
};

// Chat elements
const chatMessages = document.getElementById('chat-messages');
const chatWelcome = document.getElementById('chat-welcome');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const sendIcon = document.getElementById('send-icon');

// Sliders
const sliderTokens = document.getElementById('slider-tokens');
const sliderTemp = document.getElementById('slider-temp');
const sliderTopk = document.getElementById('slider-topk');
const valTokens = document.getElementById('val-tokens');
const valTemp = document.getElementById('val-temp');
const valTopk = document.getElementById('val-topk');

// Classification elements
const classifyInput = document.getElementById('classify-input');
const classifyBtn = document.getElementById('classify-btn');
const classifyBtnText = document.getElementById('classify-btn-text');
const charCount = document.getElementById('char-count');
const classifyResult = document.getElementById('classify-result');
const resultIcon = document.getElementById('result-icon');
const resultLabel = document.getElementById('result-label');
const resultSublabel = document.getElementById('result-sublabel');
const confidenceBar = document.getElementById('confidence-bar');
const confidenceValue = document.getElementById('confidence-value');
const logitHam = document.getElementById('logit-ham');
const logitSpam = document.getElementById('logit-spam');

// Status
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');

// ============================================================
// State
// ============================================================
let currentMode = 'generate';
let isGenerating = false;
let isClassifying = false;
let messageHistory = [];

// ============================================================
// Initialization
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    setupSliders();
    setupNavigation();
    setupChatInput();
    setupClassification();
    setupWelcomeChips();
    setupExampleChips();
});

// ============================================================
// Health Check
// ============================================================
async function checkHealth() {
    try {
        const res = await fetch('/api/health');
        const data = await res.json();

        if (data.status === 'ok') {
            const genLoaded = data.models.generation;
            const clsLoaded = data.models.classification;

            if (genLoaded && clsLoaded) {
                statusDot.className = 'status-dot';
                statusText.textContent = `Models ready · ${data.device.toUpperCase()}`;
            } else {
                statusDot.className = 'status-dot loading';
                const missing = [];
                if (!genLoaded) missing.push('gen');
                if (!clsLoaded) missing.push('cls');
                statusText.textContent = `Missing: ${missing.join(', ')}`;
            }
        }
    } catch (err) {
        statusDot.className = 'status-dot error';
        statusText.textContent = 'Connection failed';
    }
}

// ============================================================
// Navigation
// ============================================================
function setupNavigation() {
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const mode = item.dataset.mode;
            if (!mode || mode === currentMode) return;
            switchMode(mode);
        });
    });
}

function switchMode(mode) {
    currentMode = mode;

    // Update nav
    navItems.forEach(item => {
        item.classList.toggle('active', item.dataset.mode === mode);
    });

    // Update panels
    Object.entries(panels).forEach(([key, panel]) => {
        panel.classList.toggle('active', key === mode);
    });

    // Focus appropriate input
    if (mode === 'generate') {
        chatInput.focus();
    } else {
        classifyInput.focus();
    }
}

// ============================================================
// Sliders
// ============================================================
function setupSliders() {
    sliderTokens.addEventListener('input', () => {
        valTokens.textContent = sliderTokens.value;
    });

    sliderTemp.addEventListener('input', () => {
        valTemp.textContent = (parseInt(sliderTemp.value) / 100).toFixed(1);
    });

    sliderTopk.addEventListener('input', () => {
        valTopk.textContent = sliderTopk.value;
    });
}

// ============================================================
// Chat — Text Generation
// ============================================================
function setupChatInput() {
    // Auto-resize textarea
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
    });

    // Enter to send, Shift+Enter for newline
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.addEventListener('click', sendMessage);
}

function setupWelcomeChips() {
    document.querySelectorAll('.welcome-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            chatInput.value = chip.dataset.prompt;
            chatInput.dispatchEvent(new Event('input'));
            sendMessage();
        });
    });
}

async function sendMessage() {
    const prompt = chatInput.value.trim();
    if (!prompt || isGenerating) return;

    // Hide welcome
    chatWelcome.style.display = 'none';

    // Add user message
    addMessage('user', prompt);
    chatInput.value = '';
    chatInput.style.height = 'auto';

    // Show loading
    isGenerating = true;
    sendBtn.disabled = true;
    sendIcon.innerHTML = '<div class="spinner"></div>';

    // Add assistant placeholder with typing cursor
    const assistantMsgEl = addMessage('assistant', '', true);

    try {
        const res = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: prompt,
                max_tokens: parseInt(sliderTokens.value),
                temperature: parseInt(sliderTemp.value) / 100,
                top_k: parseInt(sliderTopk.value),
            }),
        });

        const data = await res.json();

        if (data.generated) {
            // Type out the response
            await typeText(assistantMsgEl, data.generated);
        } else if (data.detail) {
            assistantMsgEl.querySelector('.message-text').textContent = `Error: ${JSON.stringify(data.detail)}`;
        }
    } catch (err) {
        assistantMsgEl.querySelector('.message-text').textContent = `Error: ${err.message}`;
    }

    isGenerating = false;
    sendBtn.disabled = false;
    sendIcon.textContent = '➤';
}

function addMessage(role, text, showCursor = false) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    avatarDiv.textContent = role === 'user' ? '👤' : '⚡';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const textSpan = document.createElement('span');
    textSpan.className = 'message-text';
    textSpan.textContent = text;
    contentDiv.appendChild(textSpan);

    if (showCursor) {
        const cursor = document.createElement('span');
        cursor.className = 'typing-cursor';
        contentDiv.appendChild(cursor);
    }

    msgDiv.appendChild(avatarDiv);
    msgDiv.appendChild(contentDiv);
    chatMessages.appendChild(msgDiv);

    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return msgDiv;
}

async function typeText(msgEl, text) {
    const textSpan = msgEl.querySelector('.message-text');
    const cursor = msgEl.querySelector('.typing-cursor');
    const chars = text.split('');

    for (let i = 0; i < chars.length; i++) {
        textSpan.textContent += chars[i];
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Variable speed for natural feel
        const delay = chars[i] === ' ' ? 15 : chars[i] === '\n' ? 40 : 20;
        await sleep(delay);
    }

    // Remove cursor after typing
    if (cursor) cursor.remove();
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================================
// Classification
// ============================================================
function setupClassification() {
    classifyInput.addEventListener('input', () => {
        charCount.textContent = `${classifyInput.value.length} chars`;
    });

    classifyInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            e.preventDefault();
            classifyText();
        }
    });

    classifyBtn.addEventListener('click', classifyText);
}

function setupExampleChips() {
    document.querySelectorAll('.example-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            classifyInput.value = chip.dataset.text;
            charCount.textContent = `${chip.dataset.text.length} chars`;
            classifyText();
        });
    });
}

async function classifyText() {
    const text = classifyInput.value.trim();
    if (!text || isClassifying) return;

    isClassifying = true;
    classifyBtn.disabled = true;
    classifyBtnText.innerHTML = '<div class="spinner"></div> Classifying...';

    // Hide previous result
    classifyResult.classList.remove('visible', 'spam', 'ham');

    try {
        const res = await fetch('/api/classify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });

        const data = await res.json();
        showClassificationResult(data);
    } catch (err) {
        resultLabel.textContent = 'Error';
        resultSublabel.textContent = err.message;
        classifyResult.classList.add('visible', 'spam');
    }

    isClassifying = false;
    classifyBtn.disabled = false;
    classifyBtnText.textContent = '🔍 Classify';
}

function showClassificationResult(data) {
    const isSpam = data.label === 'spam';
    const confidencePct = (data.confidence * 100).toFixed(1);

    // Set result card class
    classifyResult.className = `classify-result visible ${data.label}`;

    // Icon
    resultIcon.textContent = isSpam ? '🚨' : '✅';

    // Label
    resultLabel.textContent = isSpam ? 'SPAM' : 'HAM';
    resultSublabel.textContent = isSpam
        ? 'This message appears to be spam'
        : 'This message appears legitimate';

    // Confidence bar (animate)
    confidenceBar.style.width = '0%';
    confidenceValue.textContent = '0%';

    // Trigger animation after a brief delay
    requestAnimationFrame(() => {
        setTimeout(() => {
            confidenceBar.style.width = `${confidencePct}%`;
            animateCounter(confidenceValue, 0, parseFloat(confidencePct), 600);
        }, 100);
    });

    // Logits
    logitHam.textContent = data.logits[0].toFixed(4);
    logitSpam.textContent = data.logits[1].toFixed(4);
}

function animateCounter(element, start, end, duration) {
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = start + (end - start) * eased;

        element.textContent = `${current.toFixed(1)}%`;

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}
