# Free-API-Server n8n Integration Guide

A complete guide for using the Free-API-Server (Gemini via cookies) with n8n workflows.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Available Models](#available-models)
3. [Image Generation](#image-generation)
4. [Text/Chat Completions](#textchat-completions)
5. [n8n Workflow Examples](#n8n-workflow-examples)
6. [Cookie Management](#cookie-management)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Base URL
```
http://free-api-server:3897/v1
```

### Authentication
Use any dummy API key:
```
Authorization: Bearer dummy
```

---

## Available Models

### Text Models

| OpenAI Model Name | Gemini Model | Best For |
|------------------|--------------|----------|
| `gpt-3.5-turbo` | Gemini 2.5 Flash | Fast, everyday tasks |
| `gpt-4` | Gemini 3.0 Pro | Complex reasoning |
| `gpt-4-turbo` | Gemini 3.0 Pro | Advanced tasks |
| `gpt-4o` | Gemini 3.0 Pro | Multimodal |
| `gemini-3-pro` | Gemini 3.0 Pro | Latest Pro |
| `gemini-3.1-pro` | Gemini 3.0 Pro | Latest Pro |

### Image Generation Models

| OpenAI Model Name | Description |
|------------------|-------------|
| `gemini-3-pro-image` | Image generation (auto watermark removal) |
| `gemini-3-flash-image` | Faster image generation |
| `gemini-3-pro` | Default text model |

---

## Image Generation

### Basic Request

```json
{
  "model": "gemini-3-pro-image",
  "prompt": "A beautiful sunset over mountains",
  "n": 1,
  "size": "1024x1024"
}
```

### Supported Sizes

| Size | Aspect Ratio | Notes |
|------|--------------|-------|
| `1024x1024` | 1:1 | Square (default) |
| `1024x576` | 16:9 | Landscape |
| `576x1024` | 9:16 | Portrait |
| `1536x1024` | 3:2 | Wide |
| `1024x1536` | 2:3 | Tall |

### Image Sizes for n8n

**16:9 Landscape (YouTube, social media):**
```json
{
  "model": "gemini-3-pro-image",
  "prompt": "Your prompt here",
  "size": "1024x576"
}
```

**9:16 Portrait (Stories, reels):**
```json
{
  "model": "gemini-3-pro-image",
  "prompt": "Your prompt here",
  "size": "576x1024"
}
```

### Complete n8n Image Workflow

```
┌─────────────────┐
│ Trigger (Cron/  │
│ Webhook)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│ HTTP Request    │────▶│ HTTP Request     │
│ (Generate)      │     │ (Download Image) │
│                 │     │                  │
│ POST /v1/images │     │ GET {{json.data}} │
│ /generations    │     │ [0].url          │
└─────────────────┘     └──────────────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Your Action      │
                         │ (Email/Drive/    │
                         │  Slack/S3)       │
                         └──────────────────┘
```

### Get Image as Binary in n8n

1. **First HTTP Request** - Generate image
2. **Second HTTP Request** - Download:
   ```
   URL: {{ $json.data[0].url }}
   Response Format: File (Buffer)
   ```
3. **Use Binary Output** in subsequent nodes

---

## Text/Chat Completions

### Basic Chat

```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}
```

### Multi-turn Conversation

```json
{
  "model": "gpt-4",
  "messages": [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."},
    {"role": "user", "content": "How do I install it?"}
  ]
}
```

### With System Prompt

```json
{
  "model": "gpt-4",
  "messages": [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a hello world in Python"}
  ]
}
```

### Streaming Response

```json
{
  "model": "gpt-3.5-turbo",
  "messages": [{"role": "user", "content": "Write a story"}],
  "stream": true
}
```

> **Note:** For streaming in n8n, use "Response is Streaming" option in HTTP Request node.

### Available Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | gpt-3.5-turbo | Model to use |
| `messages` | array | required | Message history |
| `temperature` | float | 0.7 | Creativity (0-2) |
| `top_p` | float | 1.0 | Nucleus sampling |
| `max_tokens` | integer | auto | Max response length |
| `stream` | boolean | false | Enable streaming |

---

## n8n Workflow Examples

### Example 1: Image to Email Attachment

```json
{
  "nodes": [
    {
      "name": "Generate Image",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "POST",
        "url": "http://free-api-server:3897/v1/images/generations",
        "body": {
          "model": "gemini-3-pro-image",
          "prompt": "{{ $json.prompt }}",
          "n": 1,
          "size": "1024x1024"
        },
        "options": {
          "headers": {
            "Authorization": "Bearer dummy",
            "Content-Type": "application/json"
          }
        }
      }
    },
    {
      "name": "Download Image",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "GET",
        "url": "{{ $json.data[0].url }}",
        "responseFormat": "file"
      }
    },
    {
      "name": "Send Email",
      "type": "n8n-nodes-base.gmail",
      "parameters": {
        "to": "email@example.com",
        "subject": "Your Generated Image",
        "attachmentsBinary": "data"
      }
    }
  ]
}
```

### Example 2: Chat with Context

```json
{
  "nodes": [
    {
      "name": "Chat to Gemini",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "POST",
        "url": "http://free-api-server:3897/v1/chat/completions",
        "body": {
          "model": "gpt-4",
          "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "{{ $json.user_message }}"}
          ],
          "temperature": 0.7
        },
        "options": {
          "headers": {
            "Authorization": "Bearer dummy"
          }
        }
      }
    }
  ]
}
```

### Example 3: Image with Custom Size

```json
{
  "model": "gemini-3-pro-image",
  "prompt": "A YouTube thumbnail for a tech video",
  "n": 1,
  "size": "1280x720"
}
```

---

## Cookie Management

### Getting Cookies

1. Open https://gemini.google.com (login with your Google account)
2. Press F12 → Developer Tools
3. Go to Application → Cookies
4. Copy:
   - `__Secure-1PSID`
   - `__Secure-1PSIDTS`

### Updating Cookies in Docker

**Method 1: Via Portainer**
1. Go to Container → Environment
2. Edit `SECURE_1PSID` and `SECURE_1PSIDTS`
3. Recreate/Restart container

**Method 2: Via Volume**
1. Mount a volume to `/app/.env`
2. Edit .env file directly
3. Restart container

### Cookie Expiration

- **Pro accounts:** 7-14 days (sometimes longer)
- **Regular accounts:** 1-7 days
- Update cookies when you see authentication errors

---

## Troubleshooting

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `401 Unauthorized` | Invalid/expired cookies | Update cookies |
| `500 Internal Server Error` | Server initialization failed | Check logs, restart container |
| `403 Forbidden` | Cookies not passed | Use proxy URL, not direct Google URL |
| `Failed to initialize` | Cookie Service unreachable | Normal on first run, should auto-retry |

### Check Server Status

```bash
curl http://free-api-server:3897/v1/models
```

### View Logs

In Portainer: Container → Logs

---

## Advanced Usage

### Thinking Mode (Gemini Pro)

For complex reasoning tasks:
```json
{
  "model": "gpt-4-turbo",
  "messages": [{"role": "user", "content": "Solve this complex problem: ..."}]
}
```

### Multimodal Input (Images + Text)

```json
{
  "model": "gpt-4",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
      ]
    }
  ]
}
```

### Image URL Formats Supported

- Direct URLs: `https://example.com/image.jpg`
- Base64: `data:image/jpeg;base64,/9j/...`
- Local file path (not in n8n): `/path/to/image.jpg`

---

## API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Text chat |
| `/v1/images/generations` | POST | Generate images |
| `/v1/images/proxy` | GET | Proxy/download images |

---

## Support

For issues, check:
1. Container logs in Portainer
2. Cookie validity
3. Network connectivity between n8n and free-api-server
