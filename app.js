document.getElementById("send-btn").addEventListener("click", function() {
    sendMessage();
});

document.getElementById("user-input").addEventListener("keydown", function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

function sendMessage() {
    let inputBox = document.getElementById("user-input");
    let userMessage = inputBox.value.trim();

    if (userMessage === "") return;

    appendMessage("user", userMessage);
    appendMessage("assistant", "Thinking...");  

    inputBox.value = "";

    fetch('http://127.0.0.1:5000/chatbot', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userMessage }),
    })
    .then(response => response.json())
    .then(data => {
        removeLastAssistantMessage();
        appendMessage("assistant", data.response);
    })
    .catch((error) => {
        console.error('Error:', error);
        appendMessage("assistant", "Sorry, something went wrong.");
    });
}

function appendMessage(role, message) {
    let chatBox = document.getElementById("chat-box");
    let messageElement = document.createElement("div");
    messageElement.classList.add("message", role);
    messageElement.textContent = message;

    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
}


function removeLastAssistantMessage() {
    let chatBox = document.getElementById("chat-box");
    let messages = chatBox.getElementsByClassName("message assistant");

    if (messages.length > 0) {
        chatBox.removeChild(messages[messages.length - 1]);
    }
}
