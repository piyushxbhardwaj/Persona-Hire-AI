# Vapi Voice Representative Integration Guide

This guide walks you through connecting **PersonaHire AI** to a live telephone / voice agent channel using Vapi, ElevenLabs, and Deepgram.

---

## 1. Prerequisites
- A **Vapi** account (with free credits for testing).
- An **OpenAI API Key** (configured on Vapi or passed from our backend).
- Optional: An **ElevenLabs API Key** if you wish to use premium speech synthesis (though ElevenLabs default voices can be selected out-of-the-box on Vapi).

---

## 2. Fast Setup (Dashboard Import)
1. Copy the contents of [`voice_agent/vapi_config.json`](vapi_config.json).
2. Go to the [Vapi Dashboard Assistants page](https://dashboard.vapi.ai/assistants).
3. Click **Create Assistant** -> select **Import JSON**.
4. Paste the JSON config.
5. In the assistant's configuration panel under **Model**:
   - Replace the `url` value (`https://your-backend-api-url.render.com/api/voice`) with your public FastAPI backend URL (e.g., your Render domain or Ngrok tunnel ending in `/api/voice`).
6. Click **Save**.

---

## 3. Webhook Tool Call Setup (Function Calling)
For the voice representative to query slot availabilities and schedule calendar interviews:
1. Go to the **Tools** tab in the Vapi dashboard.
2. Import the tool functions declared in `vapi_config.json` (`get_available_slots`, `book_interview`, `cancel_interview`).
3. Set the **Server URL** of the tools to your backend webhook endpoint:
   ```text
   https://your-backend-api-url.render.com/api/voice/vapi/webhook
   ```
4. Bind these tools to your assistant under the **Tools** section of your assistant's profile page.

---

## 4. Local Testing via Ngrok Tunnel
To test the voice agent locally before deploying the backend:
1. Expose your local FastAPI port (8000) using Ngrok:
   ```bash
   ngrok http 8000
   ```
2. Copy the resulting forwarding HTTPS URL (e.g., `https://1234-abcd.ngrok-free.app`).
3. Update the Vapi custom LLM URL and tool Webhook server URL:
   - Custom LLM: `https://1234-abcd.ngrok-free.app/api/voice`
   - Tool Webhook: `https://1234-abcd.ngrok-free.app/api/voice/vapi/webhook`
4. Use the **Talk to Assistant** button on the Vapi dashboard to speak to the representative directly in your browser.
