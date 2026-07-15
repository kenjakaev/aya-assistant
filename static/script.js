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
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: messageText })
        });

        if (response.ok) {
            const data = await response.json();
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

function appendMessage(text, className) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', className);

    messageDiv.innerHTML = md.render(text);

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}