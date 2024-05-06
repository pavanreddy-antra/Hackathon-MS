css = '''
<style>
.chat-message {
    padding: 1rem; 
    border-radius: 0.5rem; 
    margin-bottom: 1rem; 
    display: flex; 
    word-wrap:break-word;
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 50px;
  height: 50px;
}
.chat-message .avatar img {
  max-width: 50px;
  max-height: 50px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message-content {
  display: flex;
  flex-direction: column;
  padding-left: 1rem; 
}
.chat-message .message {
  color: #fff;
}
.chat-message .references {
  width: 100%;
}
.chat-message .references ul {
  list-style: none;
  padding: 0;
}
.chat-message .references li a {
  color: #fff;
}

.button {
        border: none;
        padding: 10px 20px;
        cursor: pointer;
        margin-right: 10px; /* Add some spacing between buttons */
        font-size: 16px; /* Adjust the font size */
        background-color: #ccc; /* Initial background color */
        color: white;
        border-radius: 5px;
        transition: background-color 0.3s; /* Smooth color transition */
    }

.button:hover {
        background-color: #aaa; /* Hover background color */
    }
footer {
	visibility: hidden;
}
</style>


'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://img.icons8.com/carbon-copy/100/bot.png">
    </div>
    <div class="message-content">
        <div class="message">{message}</div>
        <BR>
        <div class="references">
          <div class="message">References:</div>
              <ul style=" width = 150px; overflow-wrap: break-word;">
              {references}
              </ul>
        </div>
    </div>
</div>
'''

idea_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="robot.png">
    </div>
    <div class="message-content">
        <div class="message">{message}</div>
        <br>
        <div class="button-row">
            <label class="button" onclick="setPrompt(0)">E</label>
            <label class="button" onclick="setPrompt(1)">S</label>
            <label class="button" onclick="setPrompt(2)">G</label>
        </div>
    </div>
</div>


'''




user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://img.icons8.com/laces/64/user.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''