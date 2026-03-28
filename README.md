# Gemini-Free-API

OpenAI-compatible API server for Google Gemini Web - free, no API key required.

## Features

- **Free Gemini Access** - Use Gemini 3.1 Pro and Gemini 3.0 Flash via web interface
- **OpenAI Compatible** - Works with OpenAI SDK, Claude SDK, and any OpenAI-compatible client
- **Image Generation** - Generate images with automatic watermark removal
- **Streaming Support** - Real-time streaming responses
- **Docker Ready** - One-command deployment with Docker

## Quick Start

### Docker (Recommended)

```bash
# Pull and run
docker run -d -p 3897:3897 \
  -e SECURE_1PSID="your_cookie_here" \
  -e SECURE_1PSIDTS="your_cookie_ts" \
  --name gemini-api \
  ghcr.io/sialabs/free-api-server:latest
```

Or use docker-compose:

```yaml
services:
  gemini-api:
    image: ghcr.io/sialabs/free-api-server:latest
    ports:
      - "3897:3897"
    environment:
      - SECURE_1PSID=${SECURE_1PSID}
      - SECURE_1PSIDTS=${SECURE_1PSIDTS}
```

### Get Cookies

1. Visit https://gemini.google.com and login
2. Press F12 → Application → Cookies
3. Copy `__Secure-1PSID` and `__Secure-1PSIDTS` values

### Usage

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy",
    base_url="http://localhost:3897/v1"
)

# Chat
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)

# Generate Image
response = client.images.generate(
    model="gemini-3-pro-image",
    prompt="A cute cat"
)
print(response.data[0].url)
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /v1/models` | List available models |
| `POST /v1/chat/completions` | Chat completion |
| `POST /v1/images/generations` | Image generation |
| `GET /docs` | API documentation (Swagger) |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SECURE_1PSID` | Yes | Gemini cookie |
| `SECURE_1PSIDTS` | Yes | Gemini cookie TS |
| `PROXY` | No | HTTP proxy URL |

## Model Mapping

| OpenAI Model | Gemini Model |
|--------------|--------------|
| gpt-3.5-turbo | gemini-3.0-flash |
| gpt-4 | gemini-3.1-pro |
| gemini-3-pro-image | gemini-3.0-flash (auto watermark removal) |

## License

See LICENSE file.

## Credits

- [gemini-webapi](https://github.com/HanaokaYuzu/Gemini-API) - Core library
