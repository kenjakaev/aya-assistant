const chatForm = document.getElementById('chat-form');
const messageInput = document.getElementById('message-input');
const chatMessages = document.getElementById('chat-messages');

const md = window.markdownit({
    html: true,
    linkify: true,
    breaks: true
}).use(texmath, {
    engine: katex,
    delimiters: 'dollars',
    katOptions: { throwOnError: false }
});

chatForm.addEventListener('submit', async (event) => {
    event.preventDefault();

    const messageText = messageInput.value.trim();
    if (!messageText) return;

    appendMessage(messageText, 'user-message');
    messageInput.value = '';

    try {
        showTypingIndicator()
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: messageText })
        });

        if (response.ok) {
            const data = await response.json();
            removeTypingIndicator()
            appendMessage(data.response, 'bot-message');
        } else {
            appendMessage('Error: failed to receive a response from Aya.',
                'bot-message');
        }
    } catch (error) {
        console.error(error);
        appendMessage('Server connection error', 'bot-message');
    }
});

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

async function appendMessage(text, className) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', className);
    chatMessages.appendChild(messageDiv);

    if (className === 'user-message') {
        messageDiv.innerHTML = md.render(text);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return;
    }

    let currentMarkdownText = '';

    for (let i = 0; i < text.length; i++) {
        currentMarkdownText += text[i];

        messageDiv.innerHTML = md.render(currentMarkdownText);

        chatMessages.scrollTop = chatMessages.scrollHeight;

        await sleep(15);
    }
}

function showTypingIndicator() {
    const indicatorDiv = document.createElement('div');
    indicatorDiv.id = 'typing-indicator';
    indicatorDiv.classList.add('typing-indicator');

    indicatorDiv.innerHTML = '<span></span><span></span><span></span>';

    chatMessages.appendChild(indicatorDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) {
        indicator.remove();
    }
}