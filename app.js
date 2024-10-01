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

    if (userMessage === "") return;  // Don't send empty messages

    appendMessage("user", userMessage);  // Display user's message in chat
    inputBox.value = "";  // Clear the input box

    // Make a POST request to the Flask backend
    fetch('http://127.0.0.1:5000/chatbot', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',  // Tell the backend you're sending JSON
        },
        body: JSON.stringify({ message: userMessage }),  // Send the user's message
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("Network response was not ok");
        }
        return response.json();  // Parse JSON response
    })
    .then(data => {
        appendMessage("assistant", data.response);  // Display assistant's response
    })
    .catch((error) => {
        console.error('Error:', error);
        appendMessage("assistant", "Sorry, something went wrong.");  // Handle errors
    });
}

function appendMessage(role, message) {
    let chatBox = document.getElementById("chat-box");  // Find chat box element
    let messageElement = document.createElement("div");  // Create a new message div
    messageElement.classList.add("message", role);  // Add the appropriate classes
    messageElement.textContent = message;  // Set the message text

    chatBox.appendChild(messageElement);  // Add the message to the chat box
    chatBox.scrollTop = chatBox.scrollHeight;  // Auto-scroll to the bottom
}
