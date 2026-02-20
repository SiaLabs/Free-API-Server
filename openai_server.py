import os
import sys
import time
import uuid
import asyncio
import logging
import tempfile
import urllib.parse
import json
from typing import List, Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager
from io import BytesIO
import base64

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import dotenv
import httpx
from PIL import Image as PILImage
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

from gemini_webapi.client import GeminiClient, ChatSession
from gemini_webapi.constants import Model

# Setup logging
import sys
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s:%(name)s:%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger("openai_server")
logger.setLevel(logging.INFO)

# Load environment variables
load_dotenv(override=True)

# 启动时先杀掉占用端口的进程
if __name__ == "__main__":
    import subprocess
    port = int(os.getenv("PORT", 3897))
    
    # 循环直到端口完全空闲（包括TIME_WAIT）
    max_attempts = 15
    for attempt in range(max_attempts):
        try:
            # 检查LISTENING状态
            result = subprocess.run(f'netstat -ano | findstr :{port} | findstr LISTENING', 
                                  shell=True, capture_output=True, text=True)
            
            if result.stdout.strip():
                # 有LISTENING，杀掉
                pids = set()
                for line in result.stdout.splitlines():
                    parts = line.split()
                    if len(parts) >= 5 and parts[-1].isdigit() and parts[-1] != '0':
                        pids.add(parts[-1])
                
                if pids:
                    logger.info(f"Attempt {attempt+1}: Killing PIDs {pids} on port {port}")
                    for pid in pids:
                        subprocess.run(f'taskkill /F /PID {pid}', shell=True, 
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(5)  # 等待进程完全退出并释放端口
                    continue
            
            # 检查是否还有TIME_WAIT（仅记录，不阻塞）
            result_all = subprocess.run(f'netstat -ano | findstr :{port}', 
                                       shell=True, capture_output=True, text=True)
            if not result_all.stdout.strip():
                logger.info(f"Port {port} is completely free")
            else:
                logger.info(f"Port {port} has TIME_WAIT connections (ignored)")
            break
        except Exception as e:
            logger.error(f"Error checking port: {e}")
            break
    
    # 最后检查
    result = subprocess.run(f'netstat -ano | findstr :{port}', 
                          shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        logger.warning(f"Port {port} still has connections, but proceeding...")

# Session storage with timestamp
chat_sessions: Dict[str, tuple[ChatSession, float]] = {}  # key -> (session, last_access_time)
SESSION_TIMEOUT = 86400 * 30  # 30天后过期（仅清理本地缓存，Gemini服务端历史依然保留）
MAX_SESSIONS = 500  # 最多500个会话缓存

# --- OpenAI API Models ---

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = int(time.time())
    owned_by: str = "google"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON 字符串

class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall

class Message(BaseModel):
    role: str
    content: Optional[Any] = None  # 可以是 str 或 List[dict]（多模态格式）
    tool_calls: Optional[List[ToolCall]] = None  # 函数调用

class Tool(BaseModel):
    type: str = "function"
    function: Optional[Dict[str, Any]] = None
    # 也可能直接是 JSON 字符串格式
    name: Optional[str] = None
    description: Optional[str] = None
    inputSchema: Optional[Dict[str, Any]] = None

class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None

class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage

class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str]

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]

# --- OpenAI Image Models ---

class ImageRequest(BaseModel):
    prompt: str
    model: Optional[str] = "gemini-3-pro"
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    quality: Optional[str] = "standard"
    response_format: Optional[str] = "url"
    user: Optional[str] = None

class ImageData(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None

class ImageResponse(BaseModel):
    created: int
    data: List[ImageData]

# --- OpenAI TTS Models ---

class TTSRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "alloy"
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0

# --- Global Gemini Client ---
gemini_client: GeminiClient = None

# --- Vertex AI TTS Config (Optional) ---
# Uncomment and configure to enable TTS functionality
# Requirements: pip install pyjwt cryptography
# 
# Setup:
# 1. Create GCP service account with Vertex AI User role
# 2. Download JSON key file
# 3. Set environment variables in .env:
#    VERTEX_PROJECT_ID=your_gcp_project_id
#    VERTEX_LOCATION=us-central1
#    VERTEX_SERVICE_ACCOUNT_KEY=path/to/service-account-key.json
# 4. Uncomment the code below

# VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", "")
# VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
# VERTEX_SERVICE_ACCOUNT_KEY = os.getenv("VERTEX_SERVICE_ACCOUNT_KEY", "")
# 
# _vertex_token_cache = {"token": None, "expires_at": 0}
# 
# VOICE_MAP = {
#     "alloy": "Kore",
#     "echo": "Puck",
#     "fable": "Zephyr",
#     "onyx": "Charon",
#     "nova": "Aoede",
#     "shimmer": "Puck"
# }
# 
# def get_vertex_access_token():
#     import time
#     import jwt
#     
#     if _vertex_token_cache["token"] and time.time() < _vertex_token_cache["expires_at"] - 300:
#         return _vertex_token_cache["token"]
#     
#     with open(VERTEX_SERVICE_ACCOUNT_KEY) as f:
#         key_data = json.load(f)
#     
#     now = int(time.time())
#     payload = {
#         'iss': key_data['client_email'],
#         'sub': key_data['client_email'],
#         'aud': 'https://oauth2.googleapis.com/token',
#         'iat': now,
#         'exp': now + 3600,
#         'scope': 'https://www.googleapis.com/auth/cloud-platform'
#     }
#     
#     assertion = jwt.encode(payload, key_data['private_key'], algorithm='RS256')
#     
#     proxy = os.getenv("PROXY")
#     import httpx
#     response = httpx.post(
#         'https://oauth2.googleapis.com/token',
#         headers={'Content-Type': 'application/json'},
#         json={
#             'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer',
#             'assertion': assertion
#         },
#         timeout=30.0,
#         proxy=proxy
#     )
#     
#     token = response.json()['access_token']
#     _vertex_token_cache["token"] = token
#     _vertex_token_cache["expires_at"] = now + 3600
#     
#     return token

async def subscribe_to_cookie_service(port: int, max_retries: int = 10, retry_delay: int = 10):
    """订阅 Cookie 服务，支持重试"""
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:3898/subscribe",
                    json={
                        "domain": ".google.com",
                        "webhook_url": f"http://localhost:{port}/webhook/cookies",
                        "cookie_names": ["__Secure-1PSID", "__Secure-1PSIDTS"],
                        "app_name": "Gemini OpenAI Server"
                    },
                    timeout=5.0
                )
                logger.info(f"Subscribed to Cookie Service: {response.json()}")
                return True
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Cookie Service not ready (attempt {attempt+1}/{max_retries}): {e}")
                logger.info(f"Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
            else:
                logger.warning(f"Failed to subscribe after {max_retries} attempts: {e}")
                logger.warning("Will use cookies from .env file")
                return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global gemini_client
    
    secure_1psid = os.getenv("SECURE_1PSID")
    secure_1psidts = os.getenv("SECURE_1PSIDTS")
    proxy = os.getenv("PROXY")
    port = int(os.getenv("PORT", 3897))
    
    # 订阅 Cookie 服务（带重试）
    await subscribe_to_cookie_service(port)
    
    if not secure_1psid or not secure_1psidts:
        logger.error("SECURE_1PSID or SECURE_1PSIDTS not found in .env")
        logger.error("Please ensure browser extension is running and has updated cookies")
    
    gemini_client = GeminiClient(
        secure_1psid=secure_1psid,
        secure_1psidts=secure_1psidts,
        proxy=proxy
    )
    try:
        logger.info(f"Initializing Gemini client...")
        await gemini_client.init(auto_refresh=False)
        logger.info("Gemini client initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize Gemini Client: {e}")
        logger.warning("Server will start anyway. Waiting for extension to update cookies...")
    yield
    if gemini_client:
        await gemini_client.close()

app = FastAPI(lifespan=lifespan)

# 添加CORS支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/get_cookies")
async def get_cookies():
    """返回当前.env中的cookies"""
    return {
        "SECURE_1PSID": os.getenv("SECURE_1PSID", ""),
        "SECURE_1PSIDTS": os.getenv("SECURE_1PSIDTS", "")
    }

@app.post("/update_cookies")
async def update_cookies_legacy(request: Request):
    """接收扩展发送的cookies并更新内存和.env（兼容旧版）"""
    global gemini_client
    try:
        data = await request.json()
        
        new_psid = data.get('SECURE_1PSID')
        new_psidts = data.get('SECURE_1PSIDTS')
        
        if not new_psid or not new_psidts:
            return {"success": False, "message": "Missing required cookies"}
        
        # 更新内存中的 Cookie
        if gemini_client:
            gemini_client.cookies['__Secure-1PSID'] = new_psid
            gemini_client.cookies['__Secure-1PSIDTS'] = new_psidts
            if gemini_client.client:
                gemini_client.client.cookies.set('__Secure-1PSID', new_psid)
                gemini_client.client.cookies.set('__Secure-1PSIDTS', new_psidts)
            logger.info("Cookies updated in memory")
        
        # 异步更新 .env
        import dotenv
        dotenv.set_key('.env', 'SECURE_1PSID', new_psid)
        dotenv.set_key('.env', 'SECURE_1PSIDTS', new_psidts)
        
        return {"success": True, "message": "Cookies已更新"}
    except Exception as e:
        logger.error(f"Failed to update cookies: {e}")
        return {"success": False, "message": str(e)}


@app.post("/webhook/cookies")
async def webhook_cookies(request: Request):
    """接收 Cookie 服务的 webhook 推送"""
    global gemini_client
    try:
        data = await request.json()
        domain = data.get('domain')
        cookies = data.get('cookies', {})
        
        if domain != '.google.com':
            return {"status": "ignored", "reason": "not google domain"}
        
        new_psid = cookies.get('__Secure-1PSID')
        new_psidts = cookies.get('__Secure-1PSIDTS')
        
        if not new_psid or not new_psidts:
            return {"status": "ignored", "reason": "missing required cookies"}
        
        # 更新内存中的 Cookie
        if gemini_client:
            gemini_client.cookies['__Secure-1PSID'] = new_psid
            gemini_client.cookies['__Secure-1PSIDTS'] = new_psidts
            if gemini_client.client:
                gemini_client.client.cookies.set('__Secure-1PSID', new_psid)
                gemini_client.client.cookies.set('__Secure-1PSIDTS', new_psidts)
            logger.info("Cookies updated from webhook")
        
        # 异步更新 .env
        import dotenv
        dotenv.set_key('.env', 'SECURE_1PSID', new_psid)
        dotenv.set_key('.env', 'SECURE_1PSIDTS', new_psidts)
        
        return {"status": "ok", "message": "Cookies updated"}
    except Exception as e:
        logger.error(f"Failed to process webhook: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/v1/models")
async def list_models():
    """List available models."""
    models = [
        ModelInfo(id="gemini-3-flash", owned_by="google"),
        ModelInfo(id="gemini-3-pro", owned_by="google"),
        ModelInfo(id="gemini-3-pro-high", owned_by="google"),
        ModelInfo(id="gemini-3-flash-image", owned_by="google"),
        ModelInfo(id="gemini-3-pro-image", owned_by="google"),
    ]
    return ModelList(data=models)

def map_model(openai_model: str) -> Model:
    model_str = openai_model.lower()
    if "ultra" in model_str or "pro" in model_str:
        return Model.G_3_0_PRO
    return Model.G_2_5_FLASH

def estimate_tokens(text: str) -> int:
    return len(text) // 4

def remove_gemini_watermark(image_bytes: bytes, debug: bool = False) -> bytes:
    """使用固定位置 + 反向 Alpha 混合去除水印"""
    ALPHA_THRESHOLD = 0.002
    
    img = PILImage.open(BytesIO(image_bytes))
    
    if img.mode == 'RGBA':
        has_alpha = True
        alpha_channel = img.split()[3]
        img = img.convert('RGB')
    else:
        has_alpha = False
        img = img.convert('RGB')
    
    width, height = img.size
    
    # 选择水印尺寸和位置：必须两个维度都>1024才用大水印
    if width > 1024 and height > 1024:
        wm_size = 96
        margin = 64
        alpha_map_path = 'assets/bg_96.png'
    else:
        wm_size = 48
        margin = 32
        alpha_map_path = 'assets/bg_48.png'
    
    # 计算水印位置（固定在右下角）
    wm_x = width - margin - wm_size
    wm_y = height - margin - wm_size
    
    # 边界检查
    if wm_x < 0 or wm_y < 0 or wm_x + wm_size > width or wm_y + wm_size > height:
        logger.warning(f"Image too small for watermark removal: {width}x{height}")
        output = BytesIO()
        if has_alpha:
            img.putalpha(alpha_channel)
        img.save(output, format='PNG')
        return output.getvalue()
    
    # 加载 alpha map
    try:
        alpha_img = PILImage.open(alpha_map_path).convert('RGB')
        alpha_arr = np.array(alpha_img, dtype=np.float32)
        alpha_map = np.max(alpha_arr, axis=2) / 255.0
    except Exception as e:
        logger.warning(f"Failed to load alpha map: {e}")
        output = BytesIO()
        if has_alpha:
            img.putalpha(alpha_channel)
        img.save(output, format='PNG')
        return output.getvalue()
    
    img_array = np.array(img, dtype=np.float32)
    
    logger.info(f"Watermark at ({wm_x},{wm_y}) size {wm_size}x{wm_size}")
    
    # Debug: 保存标记图
    if debug:
        from PIL import ImageDraw
        debug_img = img.copy()
        draw = ImageDraw.Draw(debug_img)
        draw.rectangle([wm_x, wm_y, wm_x+wm_size-1, wm_y+wm_size-1], outline='red', width=2)
        debug_img.save('test/output/debug_detected.png')
    
    # 应用反向混合
    wm_region = img_array[wm_y:wm_y+wm_size, wm_x:wm_x+wm_size].copy()
    
    for row in range(wm_size):
        for col in range(wm_size):
            alpha = alpha_map[row, col]
            
            if alpha > ALPHA_THRESHOLD:
                alpha = min(alpha, 0.99)  # 避免除零
                for c in range(3):
                    watermarked = wm_region[row, col, c]
                    original = (watermarked - alpha * 255.0) / (1.0 - alpha)
                    wm_region[row, col, c] = np.clip(original, 0, 255)
    
    img_array[wm_y:wm_y+wm_size, wm_x:wm_x+wm_size] = wm_region
    result_img = PILImage.fromarray(img_array.astype(np.uint8), 'RGB')
    
    if has_alpha:
        result_img.putalpha(alpha_channel)
    
    output = BytesIO()
    result_img.save(output, format='PNG')
    return output.getvalue()

def get_proxy_url(request: Request, image_url: str, model: str = "") -> str:
    """Generate a proxy URL for the image."""
    base_url = str(request.base_url).rstrip("/")
    encoded_url = urllib.parse.quote(image_url)
    proxy_url = f"{base_url}/v1/images/proxy?url={encoded_url}"
    if model:
        proxy_url += f"&model={urllib.parse.quote(model)}"
    # 默认使用原图以便去水印
    proxy_url += "&force_original=true"
    return proxy_url

@app.get("/v1/images/proxy")
async def proxy_image(url: str, model: str = "", remove_watermark: Optional[bool] = None, force_original: bool = False):
    """Proxy image requests to Google with cookies and optionally remove watermark.
    
    Args:
        url: Image URL
        model: Model name (if contains 'image', remove watermark)
        remove_watermark: Whether to remove watermark (overrides model-based logic)
        force_original: Force fetch original resolution with =s0 (default: False)
    """
    if not gemini_client:
        raise HTTPException(status_code=503, detail="Gemini client not initialized")
    
    # 根据模型名称决定是否去水印
    if remove_watermark is None:
        remove_watermark = "image" in model.lower() if model else False
    
    logger.info(f"Proxy request: model={model}, remove_watermark={remove_watermark}, force_original={force_original}")
    
    try:
        async with httpx.AsyncClient(
            follow_redirects=True, 
            cookies=gemini_client.cookies, 
            proxy=gemini_client.proxy
        ) as client:
            # 处理图片尺寸参数
            if "googleusercontent.com" in url:
                if force_original:
                    # 强制原始分辨率
                    if "=s" in url:
                        url = url.split("=s")[0]
                    url += "=s0"
                elif "=s" not in url:
                    # 默认缩略图（512px）
                    url += "=s512"
                
            response = await client.get(url, timeout=30.0)
            if response.status_code == 200:
                content = response.content
                
                # 去水印处理
                if remove_watermark and "googleusercontent.com" in url:
                    try:
                        content = remove_gemini_watermark(content)
                        logger.info(f"Watermark removed from image")
                    except Exception as e:
                        logger.warning(f"Failed to remove watermark: {e}")
                
                return Response(
                    content=content,
                    media_type=response.headers.get("content-type", "image/png")
                )
            else:
                raise HTTPException(status_code=response.status_code, detail="Failed to fetch image from Google")
    except Exception as e:
        logger.error(f"Image proxy error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def parse_tool_calls_from_content(content: str) -> tuple[str, List[ToolCall], List[Dict]]:
    """解析 <tool_use> XML 标签，返回 (remaining_text, tool_calls, content_items)"""
    import re
    import json
    import html
    
    tool_calls = []
    content_items = []
    
    pattern = r'(?:&lt;|<)tool_use(?:&gt;|>)\s*(?:&lt;|<)name(?:&gt;|>)\s*(.*?)\s*(?:&lt;|<)/name(?:&gt;|>)\s*(?:&lt;|<)arguments(?:&gt;|>)\s*(.*?)\s*(?:&lt;|<)/arguments(?:&gt;|>)\s*(?:&lt;|<)/tool_use(?:&gt;|>)'
    
    matches = []
    for match in re.finditer(pattern, content, re.DOTALL):
        matches.append({
            'name': match.group(1).strip(),
            'arguments': match.group(2).strip(),
            'start': match.start(),
            'end': match.end()
        })
    
    for match_info in matches:
        name = match_info['name']
        arguments = html.unescape(match_info['arguments'])
        arguments = re.sub(r'\[([^\]]*)\]\(([^)]+)\)', r'\2', arguments)
        
        try:
            args_dict = json.loads(arguments)
        except:
            continue
        
        tool_call_id = f"call_{uuid.uuid4().hex[:12]}"
        
        tool_calls.append(ToolCall(
            id=tool_call_id,
            type="function",
            function=FunctionCall(name=name, arguments=arguments)
        ))
        
        content_items.append({
            "type": "tool-call",
            "toolCallId": tool_call_id,
            "toolName": name,
            "input": args_dict
        })
    
    remaining = content
    for match_info in reversed(matches):
        remaining = remaining[:match_info['start']] + remaining[match_info['end']:]
    
    return re.sub(r'\n{3,}', '\n\n', remaining).strip(), tool_calls, content_items

def extract_text_from_content(content: Any) -> str:
    """从 OpenAI 格式的 content 中提取文本。
    
    content 可以是:
    - 字符串: 直接返回
    - 数组: 提取所有 type=text 的项的 text 字段
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        return "\n".join(texts)
    return str(content) if content else ""

def extract_files_from_content(content: Any) -> list[str]:
    """从 OpenAI 格式的 content 中提取文件。
    
    支持:
    - image_url: {"type": "image_url", "image_url": {"url": "..."}}
    - 返回本地临时文件路径列表
    """
    import base64
    import tempfile
    from pathlib import Path
    
    if not isinstance(content, list):
        return []
    
    files = []
    for item in content:
        if not isinstance(item, dict):
            continue
            
        if item.get("type") == "image_url":
            image_url_data = item.get("image_url", {})
            url = image_url_data.get("url", "")
            
            if url.startswith("data:"):
                # Base64 编码的图片
                try:
                    header, data = url.split(",", 1)
                    img_data = base64.b64decode(data)
                    
                    # 从 header 推断文件扩展名
                    ext = ".png"
                    if "jpeg" in header or "jpg" in header:
                        ext = ".jpg"
                    elif "gif" in header:
                        ext = ".gif"
                    elif "webp" in header:
                        ext = ".webp"
                    
                    # 保存到临时文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
                        f.write(img_data)
                        files.append(f.name)
                except Exception as e:
                    logger.warning(f"Failed to decode base64 image: {e}")
            elif url.startswith("http://") or url.startswith("https://"):
                # HTTP URL - 下载到临时文件
                try:
                    import httpx
                    response = httpx.get(url, timeout=30.0, follow_redirects=True)
                    if response.status_code == 200:
                        # 从 URL 或 Content-Type 推断扩展名
                        ext = ".png"
                        content_type = response.headers.get("content-type", "")
                        if "jpeg" in content_type or "jpg" in content_type:
                            ext = ".jpg"
                        elif "gif" in content_type:
                            ext = ".gif"
                        elif "webp" in content_type:
                            ext = ".webp"
                        elif url.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
                            ext = Path(url).suffix
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
                            f.write(response.content)
                            files.append(f.name)
                except Exception as e:
                    logger.warning(f"Failed to download image from {url}: {e}")
            else:
                # 本地文件路径
                if Path(url).exists():
                    files.append(url)
    
    return files


def build_tools_description(tools: Optional[List[Any]]) -> str:
    """生成完整的工具使用系统提示词"""
    if not tools:
        return ""
    
    import json
    
    tool_xmls = []
    for tool in tools:
        
        # 处理JSON字符串格式的工具
        if isinstance(tool, str):
            try:
                tool = json.loads(tool)
            except Exception:
                continue
        
        if isinstance(tool, dict):
            # 支持两种格式：
            # 1. {"type": "function", "function": {...}}
            # 2. {"type": "function", "name": "...", "inputSchema": {...}}
            if "function" in tool:
                name = tool["function"].get("name", "")
                desc = tool["function"].get("description", "")
                schema = tool["function"].get("parameters", {})
            else:
                name = tool.get("name", "")
                desc = tool.get("description", "")
                schema = tool.get("inputSchema", {})
            
            tool_xml = f"""<tool>
  <name>{name}</name>
  <description>{desc}</description>
  <arguments>
    {{"jsonSchema":{json.dumps(schema, ensure_ascii=False)}}}
  </arguments>
</tool>"""
            tool_xmls.append(tool_xml)
    
    if not tool_xmls:
        return ""
    
    return f"""In this environment you have access to a set of tools you can use to answer the user's question.

## Tool Use Formatting

Tool use is formatted using XML-style tags:

<tool_use>
  <name>{{tool_name}}</name>
  <arguments>{{json_arguments}}</arguments>
</tool_use>

## Available Tools

<tools>

{chr(10).join(tool_xmls)}

</tools>

## Rules

1. Always use the right arguments for the tools
2. Call a tool only when needed
3. Use XML tag format as shown above
4. Never re-do a tool call with the exact same parameters"""


def build_prompt_from_messages(messages: List[Message], tools: Optional[List[Any]] = None) -> str:
    """将 OpenAI 格式的消息列表转换为单一 prompt 字符串。
    
    使用 XML 风格标签，这是大模型普遍能很好理解的结构化格式。
    """
    parts = []
    system_parts = []
    conversation_parts = []
    
    for msg in messages:
        role = msg.role
        content = extract_text_from_content(msg.content)
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            conversation_parts.append(f"<|user|>\n{content}\n</|user|>")
        elif role == "assistant":
            conversation_parts.append(f"<|assistant|>\n{content}\n</|assistant|>")
    
    # 系统提示词放在最前面
    if system_parts:
        combined_system = "\n\n".join(system_parts)
        parts.append(f"<|system|>\n{combined_system}\n</|system|>")
    
    # 添加工具描述
    tools_desc = build_tools_description(tools)
    if tools_desc:
        parts.append(f"<|tools|>\n{tools_desc}\n</|tools|>")
    
    # 添加对话历史
    if conversation_parts:
        parts.append("<|conversation|>")
        parts.extend(conversation_parts)
        parts.append("</|conversation|>")
    
    # 提示模型回复
    parts.append("<|assistant|>")
    
    return "\n\n".join(parts)


def get_or_create_session(session_key: Optional[str], model: Model) -> ChatSession:
    """获取或创建会话"""
    global chat_sessions
    import time
    
    current_time = time.time()
    
    # 清理过期会话
    expired_keys = [k for k, (_, last_time) in chat_sessions.items() 
                    if current_time - last_time > SESSION_TIMEOUT]
    for k in expired_keys:
        del chat_sessions[k]
    
    # 查找现有会话
    if session_key and session_key in chat_sessions:
        session, _ = chat_sessions[session_key]
        chat_sessions[session_key] = (session, current_time)  # 更新访问时间
        return session
    
    # 创建新会话
    session = ChatSession(geminiclient=gemini_client, model=model)
    
    # 限制会话数量（LRU淘汰）
    if len(chat_sessions) >= MAX_SESSIONS:
        # 按访问时间排序，删除最旧的
        sorted_keys = sorted(chat_sessions.items(), key=lambda x: x[1][1])
        for k, _ in sorted_keys[:MAX_SESSIONS // 3]:  # 删除1/3最旧的
            del chat_sessions[k]
    
    return session


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, fast_request: Request):
    global gemini_client
    
    # 验证 API Key
    auth_header = fast_request.headers.get("Authorization", "")
    provided_key = auth_header.replace("Bearer ", "").strip() if auth_header.startswith("Bearer ") else ""
    expected_key = os.getenv("GEMINI_API_KEY", "")
    
    if expected_key and provided_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not gemini_client:
        raise HTTPException(status_code=503, detail="Gemini client not initialized. Please wait for extension to update cookies.")
    
    model = map_model(request.model)
    
    # 尝试从最后一条assistant消息的tool_calls或特殊标记中提取session_key
    # 或者从消息历史中推断
    has_history = any(msg.role == "assistant" for msg in request.messages)
    
    # 生成session key：使用第一条user消息的hash
    session_key = None
    if has_history:
        import hashlib
        for msg in request.messages:
            if msg.role == "user":
                first_user = extract_text_from_content(msg.content)
                if first_user:
                    # 使用原始内容的hash，不做过多处理避免信息丢失
                    # 只做基本的空白符标准化
                    normalized = ' '.join(first_user.split())
                    session_key = hashlib.md5(normalized.encode('utf-8', errors='ignore')).hexdigest()[:16]
                    logger.info(f"Session key: {session_key}")
                    break
    
    logger.info(f"Has history: {has_history}, Cached: {len(chat_sessions)}")
    
    session = get_or_create_session(session_key, model)
    
    # 提取最后一条用户消息或工具结果
    last_user_content = ""
    last_user_files = []
    has_tool_result = False
    
    for msg in reversed(request.messages):
        if msg.role == "user":
            last_user_content = extract_text_from_content(msg.content)
            last_user_files = extract_files_from_content(msg.content)
            break
        elif msg.role == "tool":
            has_tool_result = True
            # 工具结果：转换为 <tool_result> 格式
            tool_result_parts = []
            
            # 处理content可能是字符串或列表
            content_to_parse = msg.content
            
            # 如果是JSON字符串，解析为对象
            if isinstance(content_to_parse, str):
                import json
                try:
                    content_to_parse = json.loads(content_to_parse)
                except:
                    # 如果不是JSON，直接作为文本
                    tool_result_parts.append(f"<tool_result>\n{content_to_parse}\n</tool_result>")
                    last_user_content = "\n\n".join(tool_result_parts)
                    break
            
            # 处理解析后的内容
            if isinstance(content_to_parse, dict):
                # 如果是 {"content": [...]} 格式
                if "content" in content_to_parse:
                    content_to_parse = content_to_parse["content"]
            
            if isinstance(content_to_parse, list):
                for item in content_to_parse:
                    if isinstance(item, dict) and item.get("type") == "text":
                        # 提取文本内容
                        result_text = item.get("text", "")
                        tool_result_parts.append(f"<tool_result>\n{result_text}\n</tool_result>")
            
            if tool_result_parts:
                last_user_content = "\n\n".join(tool_result_parts)
            break
    
    # 构建prompt
    parts = []
    
    # 只有新会话才需要添加系统提示和工具说明
    if not has_history and not has_tool_result:
        system_parts = [extract_text_from_content(m.content) for m in request.messages if m.role == "system"]
        if system_parts:
            parts.append(f"<|system|>\n{chr(10).join(system_parts)}\n</|system|>")
        
        if request.tools:
            tools_desc = build_tools_description(request.tools)
            if tools_desc:
                parts.append(f"<|tools|>\n{tools_desc}\n</|tools|>")
    
    # 用户消息或工具结果
    if has_tool_result:
        if last_user_content:
            parts.append(last_user_content)
        else:
            parts.append("<tool_result>\nTool execution completed.\n</tool_result>")
    else:
        # 有历史记录时，直接发送纯文本，不需要XML标签
        if has_history:
            parts.append(last_user_content)
        else:
            parts.append(f"<|user|>\n{last_user_content}\n</|user|>")
    
    prompt = "\n\n".join(parts)
    
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(request, session, prompt, model, fast_request, not has_history, last_user_files),
            media_type="text/event-stream",
        )
    
    MAX_RETRY = 2
    RETRY_DELAY = 3
    
    content = ""
    response = None
    for attempt in range(MAX_RETRY):
        try:
            response = await session.send_message(prompt, files=last_user_files if last_user_files else None)
            content = response.text or ""
            break
        except Exception as e:
            error_msg = str(e).lower()
            is_auth_error = "expired" in error_msg or "invalid" in error_msg or "failed initialization" in error_msg or "401" in error_msg
            
            if is_auth_error and attempt < MAX_RETRY - 1:
                logger.warning(f"Auth error (attempt {attempt + 1}/{MAX_RETRY}): {e}")
                logger.info(f"Waiting {RETRY_DELAY}s for Cookie Service to update...")
                await asyncio.sleep(RETRY_DELAY)
                # Cookie 服务会自动推送更新，直接重试即可
            else:
                # 清理临时文件后再抛出异常
                for file_path in last_user_files:
                    try:
                        if file_path.startswith(tempfile.gettempdir()):
                            os.unlink(file_path)
                    except Exception:
                        pass
                raise
    
    # 清理临时文件
    for file_path in last_user_files:
        try:
            if file_path.startswith(tempfile.gettempdir()):
                os.unlink(file_path)
        except Exception:
            pass
    
    # 修复 Gemini 返回的转义 XML 标签
    content = content.replace("\\<", "<").replace("\\>", ">")
    content = content.replace("\\_", "_")
    
    # 修复 Gemini 将 URL 转为 Markdown 链接的问题
    import re
    def fix_markdown_urls_in_json(text):
        def fix_tool_call_block(match):
            block_content = match.group(1)
            fixed = re.sub(r'\[([^\]]*)\]\(([^)]+)\)', r'\2', block_content)
            return f'<tool_call>{fixed}</tool_call>'
        
        def fix_args_block(match):
            args_content = match.group(1)
            fixed = re.sub(r'\[([^\]]*)\]\(([^)]+)\)', r'\2', args_content)
            return f'<arguments>{fixed}</arguments>'
        
        text = re.sub(r'<tool_call>(.*?)</tool_call>', fix_tool_call_block, text, flags=re.DOTALL)
        text = re.sub(r'<arguments>(.*?)</arguments>', fix_args_block, text, flags=re.DOTALL)
        return text
    
    content = fix_markdown_urls_in_json(content)
    
    if response.images:
        for img in response.images:
            if hasattr(img, 'url') and img.url:
                proxy_url = get_proxy_url(fast_request, img.url, request.model)
                content += "\n\n" + f"![Generated Image]({proxy_url})"
    
    # 如果是新会话，用第一条user消息生成key并缓存
    if not has_history and content:
        import hashlib
        import time
        
        for msg in request.messages:
            if msg.role == "user":
                first_user = extract_text_from_content(msg.content)
                if first_user:
                    normalized = ' '.join(first_user.split())
                    new_key = hashlib.md5(normalized.encode('utf-8', errors='ignore')).hexdigest()[:16]
                    chat_sessions[new_key] = (session, time.time())
                    logger.info(f"Cached session: {new_key}")
                    break
    
    # 解析 tool_calls（如果有工具定义）
    tool_calls = None
    final_content = content
    finish_reason = "stop"
    
    if request.tools:
        remaining_text, parsed_tool_calls, parsed_content_items = parse_tool_calls_from_content(content)
        if parsed_tool_calls:
            tool_calls = parsed_tool_calls
            finish_reason = "tool_calls"
            
            # 构建 content 数组
            content_array = []
            if remaining_text:
                content_array.append({"type": "text", "text": remaining_text})
            content_array.extend(parsed_content_items)
            
            final_content = content_array
    
    return ChatCompletionResponse(
        id=f"chatcmpl-{session.cid or 'new'}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=final_content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=estimate_tokens(str(final_content)),
            total_tokens=estimate_tokens(prompt) + estimate_tokens(str(final_content)),
        ),
    )

@app.post("/v1/images/generations")
async def image_generations(request: ImageRequest, fast_request: Request):
    """OpenAI-compatible image generation endpoint."""
    # 验证 API Key
    auth_header = fast_request.headers.get("Authorization", "")
    provided_key = auth_header.replace("Bearer ", "").strip() if auth_header.startswith("Bearer ") else ""
    expected_key = os.getenv("GEMINI_API_KEY", "")
    
    if expected_key and provided_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not gemini_client:
        raise HTTPException(status_code=503, detail="Gemini client not initialized")
    
    # 判断是否需要去水印（模型名包含 'image' 则去水印）
    should_remove_watermark = "image" in request.model.lower()
    
    try:
        gen_prompt = f"Generate an image: {request.prompt}"
        response = await gemini_client.generate_content(gen_prompt, model=Model.G_2_5_FLASH)
        
        if not response.images:
            raise HTTPException(status_code=400, detail="Gemini failed to generate images.")
        
        image_data = []
        for img in response.images:
            if hasattr(img, 'url') and img.url:
                # 直接下载图片
                try:
                    async with httpx.AsyncClient(
                        follow_redirects=True,
                        cookies=gemini_client.cookies,
                        proxy=gemini_client.proxy
                    ) as client:
                        img_url = img.url
                        # 强制获取原始分辨率图片（=s0 表示原始尺寸）
                        if "googleusercontent.com" in img_url:
                            if "=s" in img_url:
                                img_url = img_url.split("=s")[0]
                            img_url += "=s0"
                        
                        logger.info(f"Fetching original image: {img_url}")
                        
                        img_response = await client.get(img_url, timeout=30.0)
                        if img_response.status_code == 200:
                            content = img_response.content
                            
                            # 根据模型决定是否去水印
                            if should_remove_watermark:
                                try:
                                    content = remove_gemini_watermark(content)
                                    logger.info(f"Watermark removed (model: {request.model})")
                                except Exception as e:
                                    logger.warning(f"Failed to remove watermark: {e}")
                            
                            # 转换为 base64
                            import base64
                            b64_data = base64.b64encode(content).decode('utf-8')
                            
                            if request.response_format == "b64_json":
                                image_data.append(ImageData(b64_json=b64_data, revised_prompt=response.text))
                            else:
                                # 返回代理 URL，带模型参数和 force_original
                                proxy_url = get_proxy_url(fast_request, img.url, request.model) + "&force_original=true"
                                image_data.append(ImageData(url=proxy_url, revised_prompt=response.text))
                except Exception as e:
                    logger.error(f"Failed to process image: {e}")
                    # 降级：返回原始代理 URL
                    proxy_url = get_proxy_url(fast_request, img.url, request.model)
                    image_data.append(ImageData(url=proxy_url, revised_prompt=response.text))
        
        return ImageResponse(created=int(time.time()), data=image_data)
    except Exception as e:
        logger.error(f"Error in image_generations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def stream_chat_completion(
    request: ChatCompletionRequest, session: ChatSession, prompt: str, model: Model, fast_request: Request, is_new_session: bool = True, files: list[str] = None
) -> AsyncGenerator[str, None]:
    import time
    import json
    
    completion_id = f"chatcmpl-{session.cid or int(time.time())}"
    created = int(time.time())
    
    initial_chunk = ChatCompletionStreamResponse(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[ChatCompletionStreamChoice(index=0, delta={'role': 'assistant', 'content': ''}, finish_reason=None)]
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"
    
    try:
        response = await session.send_message(prompt, files=files if files else None)
        content = response.text or ""
        content = content.replace("\\<", "<").replace("\\>", ">").replace("\\_", "_")
        
        if is_new_session and content:
            import hashlib
            for msg in request.messages:
                if msg.role == "user":
                    first_user = extract_text_from_content(msg.content)
                    if first_user:
                        normalized = ' '.join(first_user.split())
                        new_key = hashlib.md5(normalized.encode('utf-8', errors='ignore')).hexdigest()[:16]
                        chat_sessions[new_key] = (session, time.time())
                        logger.info(f"Cached session: {new_key}")
                        break
        
        if request.tools:
            remaining_text, parsed_tool_calls, parsed_content_items = parse_tool_calls_from_content(content)
            
            if parsed_tool_calls:
                if remaining_text:
                    chunk_size = 20
                    for i in range(0, len(remaining_text), chunk_size):
                        delta_text = remaining_text[i:i+chunk_size]
                        text_chunk = ChatCompletionStreamResponse(
                            id=completion_id,
                            created=created,
                            model=request.model,
                            choices=[ChatCompletionStreamChoice(index=0, delta={'content': delta_text}, finish_reason=None)]
                        )
                        yield f"data: {text_chunk.model_dump_json()}\n\n"
                
                tool_chunk = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[ChatCompletionStreamChoice(
                        index=0,
                        delta={'tool_calls': [{
                            'index': i,
                            'id': tc.id,
                            'type': 'function',
                            'function': {'name': tc.function.name, 'arguments': tc.function.arguments}
                        } for i, tc in enumerate(parsed_tool_calls)]},
                        finish_reason=None
                    )]
                )
                yield f"data: {tool_chunk.model_dump_json()}\n\n"
                
                final_chunk = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[ChatCompletionStreamChoice(index=0, delta={}, finish_reason='tool_calls')]
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return
        
        chunk_size = 20
        for i in range(0, len(content), chunk_size):
            delta_text = content[i:i+chunk_size]
            text_chunk = ChatCompletionStreamResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[ChatCompletionStreamChoice(index=0, delta={'content': delta_text}, finish_reason=None)]
            )
            yield f"data: {text_chunk.model_dump_json()}\n\n"
        
        for img in response.images:
            if hasattr(img, 'url') and img.url:
                proxy_url = get_proxy_url(fast_request, img.url, request.model)
                image_chunk = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[ChatCompletionStreamChoice(index=0, delta={'content': f"\n\n![Image]({proxy_url})"}, finish_reason=None)]
                )
                yield f"data: {image_chunk.model_dump_json()}\n\n"
        
        final_chunk = ChatCompletionStreamResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[ChatCompletionStreamChoice(index=0, delta={}, finish_reason='stop')]
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Stream error: {e}")
        error_data = {"error": {"message": str(e)}}
        yield f"data: {json.dumps(error_data)}\n\n"
    finally:
        if files:
            for f in files:
                try:
                    if f.startswith(tempfile.gettempdir()): os.unlink(f)
                except: pass

# TTS endpoint (requires Vertex AI configuration - see comments above)
# @app.post("/v1/audio/speech")
# async def audio_speech(request: TTSRequest, fast_request: Request):
#     auth_header = fast_request.headers.get("Authorization", "")
#     provided_key = auth_header.replace("Bearer ", "").strip() if auth_header.startswith("Bearer ") else ""
#     expected_key = os.getenv("GEMINI_API_KEY", "")
#     
#     if expected_key and provided_key != expected_key:
#         raise HTTPException(status_code=401, detail="Invalid API key")
#     
#     try:
#         token = get_vertex_access_token()
#         vertex_voice = VOICE_MAP.get(request.voice, "Kore")
#         vertex_model = "gemini-2.5-pro-preview-tts" if "hd" in request.model else "gemini-2.5-flash-preview-tts"
#         
#         url = f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1/projects/{VERTEX_PROJECT_ID}/locations/{VERTEX_LOCATION}/publishers/google/models/{vertex_model}:generateContent"
#         
#         headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
#         data = {
#             "contents": [{"role": "user", "parts": [{"text": request.input}]}],
#             "generationConfig": {
#                 "responseModalities": ["AUDIO"],
#                 "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": vertex_voice}}}
#             }
#         }
#         
#         proxy = os.getenv("PROXY")
#         async with httpx.AsyncClient(proxy=proxy) as client:
#             response = await client.post(url, headers=headers, json=data, timeout=60.0)
#             response.raise_for_status()
#             
#             audio_base64 = response.json()['candidates'][0]['content']['parts'][0]['inlineData']['data']
#             pcm_data = base64.b64decode(audio_base64)
#             
#             import struct
#             sample_rate, num_channels, bits_per_sample = 24000, 1, 16
#             byte_rate = sample_rate * num_channels * bits_per_sample // 8
#             block_align = num_channels * bits_per_sample // 8
#             data_size = len(pcm_data)
#             
#             wav_header = b'RIFF' + struct.pack('<I', 36 + data_size) + b'WAVE' + b'fmt ' + struct.pack('<I', 16)
#             wav_header += struct.pack('<H', 1) + struct.pack('<H', num_channels) + struct.pack('<I', sample_rate)
#             wav_header += struct.pack('<I', byte_rate) + struct.pack('<H', block_align) + struct.pack('<H', bits_per_sample)
#             wav_header += b'data' + struct.pack('<I', data_size)
#             
#             return Response(content=wav_header + pcm_data, media_type="audio/wav")
#     except Exception as e:
#         logger.error(f"TTS error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 3897))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run("openai_server:app", host=host, port=port, reload=False)