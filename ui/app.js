/**
 * GPT-2 Engineering Studio — Frontend JS
 * Handles panel navigation, API communications, real-time typing animation, and status monitoring.
 */

// DOM Elements
const navItems = document.querySelectorAll('.nav-item[data-mode]');
const panels = {
    generate: document.getElementById('panel-generate'),
    classify: document.getElementById('panel-classify'),
    architecture: document.getElementById('panel-architecture'),
};

// Chat controls
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

// Classifier elements
const classifyInput = document.getElementById('classify-input');
const classifyBtn = document.getElementById('classify-btn');
const classifyBtnText = document.getElementById('classify-btn-text');
const charCount = document.getElementById('char-count');
const classifyResult = document.getElementById('classify-result');
const resultLabel = document.getElementById('result-label');
const resultSublabel = document.getElementById('result-sublabel');
const confidenceBar = document.getElementById('confidence-bar');
const confidenceValue = document.getElementById('confidence-value');
const logitHam = document.getElementById('logit-ham');
const logitSpam = document.getElementById('logit-spam');

// Status
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');

// State
let currentMode = 'generate';
let isGenerating = false;
let isClassifying = false;

document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    setupSliders();
    setupNavigation();
    setupChatInput();
    setupClassification();
    setupPromptChips();
    setupExampleChips();
});

// Health check
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
                statusText.textContent = 'Models loading...';
            }
        }
    } catch (err) {
        statusDot.className = 'status-dot error';
        statusText.textContent = 'Connection error';
    }
}

// Navigation switching
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

    navItems.forEach(item => {
        item.classList.toggle('active', item.dataset.mode === mode);
    });

    Object.entries(panels).forEach(([key, panel]) => {
        if (panel) panel.classList.toggle('active', key === mode);
    });

    if (mode === 'generate') {
        chatInput.focus();
    } else if (mode === 'classify') {
        classifyInput.focus();
    }
}

// Slider controls
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

// Chat input handling
function setupChatInput() {
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
    });

    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.addEventListener('click', sendMessage);
}

function setupPromptChips() {
    document.querySelectorAll('.prompt-chip').forEach(chip => {
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

    if (chatWelcome) chatWelcome.style.display = 'none';

    addMessage('user', prompt);
    chatInput.value = '';
    chatInput.style.height = 'auto';

    isGenerating = true;
    sendBtn.disabled = true;
    sendIcon.innerHTML = '<div class="spinner"></div>';

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
            await typeText(assistantMsgEl, data.generated);
        } else if (data.detail) {
            assistantMsgEl.querySelector('.msg-text').textContent = `Error: ${JSON.stringify(data.detail)}`;
        }
    } catch (err) {
        assistantMsgEl.querySelector('.msg-text').textContent = `Connection error: ${err.message}`;
    }

    isGenerating = false;
    sendBtn.disabled = false;
    sendIcon.textContent = '➤';
}

function addMessage(role, text, showCursor = false) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `msg-bubble ${role}`;

    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'msg-avatar';
    avatarDiv.textContent = role === 'user' ? 'U' : 'G2';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'msg-body';

    const textSpan = document.createElement('span');
    textSpan.className = 'msg-text';
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

    chatMessages.scrollTop = chatMessages.scrollHeight;

    return msgDiv;
}

async function typeText(msgEl, text) {
    const textSpan = msgEl.querySelector('.msg-text');
    const cursor = msgEl.querySelector('.typing-cursor');
    const chars = text.split('');

    for (let i = 0; i < chars.length; i++) {
        textSpan.textContent += chars[i];
        chatMessages.scrollTop = chatMessages.scrollHeight;
        const delay = chars[i] === ' ' ? 12 : chars[i] === '\n' ? 35 : 18;
        await sleep(delay);
    }

    if (cursor) cursor.remove();
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Classifier setup
function setupClassification() {
    classifyInput.addEventListener('input', () => {
        charCount.textContent = `${classifyInput.value.length} chars`;
    });

    classifyBtn.addEventListener('click', classifyText);
}

function setupExampleChips() {
    document.querySelectorAll('.ex-chip').forEach(chip => {
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

    classifyResult.classList.remove('visible', 'is-spam', 'is-ham');

    try {
        const res = await fetch('/api/classify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });

        const data = await res.json();
        showClassificationResult(data);
    } catch (err) {
        resultLabel.textContent = 'ERROR';
        resultSublabel.textContent = err.message;
        classifyResult.classList.add('visible', 'is-spam');
    }

    isClassifying = false;
    classifyBtn.disabled = false;
    classifyBtnText.textContent = '⚡ Classify Message';
}

function showClassificationResult(data) {
    const isSpam = data.label === 'spam';
    const confidencePct = (data.confidence * 100).toFixed(1);

    classifyResult.className = `result-card visible ${isSpam ? 'is-spam' : 'is-ham'}`;

    resultLabel.textContent = isSpam ? 'SPAM MESSAGE' : 'HAM MESSAGE';
    resultSublabel.textContent = isSpam
        ? 'Flagged as unwanted/spam content by classifier'
        : 'Identified as legitimate user message';

    confidenceBar.style.width = '0%';
    confidenceValue.textContent = '0%';

    setTimeout(() => {
        confidenceBar.style.width = `${confidencePct}%`;
        confidenceValue.textContent = `${confidencePct}%`;
    }, 50);

    logitHam.textContent = (data.logits[0] >= 0 ? '+' : '') + data.logits[0].toFixed(3);
    logitSpam.textContent = (data.logits[1] >= 0 ? '+' : '') + data.logits[1].toFixed(3);
}
