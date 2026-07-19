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

    appendUserMessage(messageText);
    messageInput.value = '';

    try {
        showTypingIndicator();

        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: messageText })
        });

        removeTypingIndicator();

        if (response.ok) {
            const botMessageDiv = document.createElement('div');
            botMessageDiv.classList.add('message', 'bot-message');
            chatMessages.appendChild(botMessageDiv);

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let currentMarkdownText = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                currentMarkdownText += chunk;

                botMessageDiv.innerHTML = md.render(currentMarkdownText);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        } else {
            appendErrorMessage('Error: failed to receive a response from Aya.');
        }
    } catch (error) {
        console.error(error);
        removeTypingIndicator();
        appendErrorMessage('Server connection error');
    }
});

function appendUserMessage(text) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', 'user-message');
    messageDiv.innerHTML = md.render(text);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendErrorMessage(text) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', 'bot-message');
    messageDiv.textContent = text;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
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