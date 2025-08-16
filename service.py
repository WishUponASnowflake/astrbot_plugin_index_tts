import sys
import os
import re
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request, status, Depends
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logging.getLogger("pydub").setLevel(logging.WARNING)

#端口设置在这里
port = 5210



class TTSService:
    def __init__(self):
        self.if_remove_think_tag = False
        self.if_preload = False
        self.if_loaded = False
        self.model = None
        self.CORRECT_API_KEY = ""
        self.model_ver = "1.5"
        self.prompt_wav = ""
        self.max_text_tokens_per_sentence = 100
        self.thread_pool: ThreadPoolExecutor
        
        # 初始化路径
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        self.output_path = os.path.join(self.model_path, "outputs", "output.wav")
        
        # 添加必要的路径到sys.path
        sys.path.insert(0, os.path.join(self.model_path, "index-tts"))

    async def DownloadModel(self, model_ver=None):
        """下载并加载TTS模型"""
        model_ver = model_ver or self.model_ver
        if model_ver not in ["1", "1.5"]:
            raise ValueError("Unsupported model version. Supported versions are '1' and '1.5'.")

        from modelscope import snapshot_download

        actual_model_path = os.path.join(self.model_path, f"IndexTTS{'-1.5' if model_ver == '1.5' else ''}")

        if not os.path.exists(actual_model_path):
            logging.info(f"Downloading model version {model_ver} to {actual_model_path}...")

            # 使用线程池处理模型加载和推理
            loop = asyncio.get_event_loop()

            await loop.run_in_executor(
                self.thread_pool, 
                lambda: snapshot_download(f"IndexTeam/Index-TTS{'-1.5' if model_ver == '1.5' else ''}", local_dir=actual_model_path)
            )
        else:
            logging.warning(f"Model path {actual_model_path} already exists, skipping download.")

    async def load_model(self, model_ver=None):
        """加载TTS模型"""
        model_ver = model_ver or self.model_ver

        if model_ver == "1":
            actual_model_path = os.path.join(self.model_path, "IndexTTS")
        elif model_ver == "1.5":
            actual_model_path = os.path.join(self.model_path, "IndexTTS-1.5")

        cfg_path = os.path.join(actual_model_path, "config.yaml")
        
        if not os.path.exists(actual_model_path):
            try:
                await self.DownloadModel(model_ver)
            except:
                raise

        logging.info(f"Using model path: {actual_model_path}")
        logging.info(f"Using config path: {cfg_path}")

        from indextts.infer import IndexTTS

        self.model = IndexTTS(model_dir=actual_model_path, cfg_path=cfg_path)
        if self.model is None:
            logging.error("Failed to load model.")
            raise Exception("Failed to load model.")
        
        self.if_loaded = True
        return self.model

    @staticmethod
    def remove_thinktag(text): 
        """去除<think>标签"""
        if text:
            return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return ''

    async def verify_api_key(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """验证API密钥"""
        if credentials.scheme != "Bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme"
            )
        if credentials.credentials != self.CORRECT_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        return credentials.credentials

# 初始化FastAPI应用和服务
service = TTSService()
app = FastAPI()

def run_service():
    uvicorn.run(app, host="0.0.0.0", port=port)

@app.on_event("startup")
async def start_up():
    # 在服务类或模块级别创建线程池
    service.thread_pool = ThreadPoolExecutor(max_workers=4)  # 根据你的服务器配置调整worker数量

@app.on_event("shutdown")
async def shut_down():
    service.thread_pool.shutdown(wait=True)

# 定义请求模型
class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str

class Config(BaseModel):
    if_remove_think_tag: bool
    if_preload: bool
    prompt_wav: str
    model_ver: str
    max_text_tokens_per_sentence: int
    CORRECT_API_KEY: str

@app.post("/audio/speech")
async def generate_speech(
    request: Request, 
    speech_request: SpeechRequest, 
    apikey: str = Depends(service.verify_api_key)
):
    try:
        input_text = (service.remove_thinktag(speech_request.input) 
                    if service.if_remove_think_tag 
                    else speech_request.input)
        
        if not input_text:
            raise HTTPException(status_code=400, detail="Input text cannot be empty")

        # 使用线程池处理模型加载和推理
        loop = asyncio.get_event_loop()
        
        # 如果模型未加载，不使用线程池加载，以防爆显存
        if not service.if_preload and not service.if_loaded:
            logging.info("Loading model...")
            service.model = await service.load_model(service.model_ver)

        if service.model is not None:
            # 使用线程池执行模型推理
            await loop.run_in_executor(
                service.thread_pool, 
                lambda: service.model.infer(service.prompt_wav, input_text, service.output_path, service.max_text_tokens_per_sentence)
            )
            logging.info(f"Speech generating at {service.output_path}")
        else:
            raise HTTPException(status_code=500, detail="Model not loaded")

        if not os.path.exists(service.output_path):
            raise HTTPException(status_code=500, detail="Failed to generate speech")

        return FileResponse(
            path=service.output_path, 
            media_type='audio/wav', 
            filename="output.wav"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/config")
async def update_config(config: Config):
    if config.if_remove_think_tag is not None:
        service.if_remove_think_tag = config.if_remove_think_tag
        logging.info(f"已设置去除思考标签功能: {service.if_remove_think_tag}")
    
    if config.prompt_wav:
        service.prompt_wav = config.prompt_wav
        logging.info(f"已设置音频输入文件: {service.prompt_wav}")
    
    if config.model_ver:
        service.model_ver = config.model_ver
        logging.info(f"已设置模型版本: {service.model_ver}")
    
    if config.max_text_tokens_per_sentence:
        service.max_text_tokens_per_sentence = config.max_text_tokens_per_sentence
        logging.info(f"已设置单句最大Token数: {service.max_text_tokens_per_sentence}")

    if config.CORRECT_API_KEY:
        service.CORRECT_API_KEY = config.CORRECT_API_KEY                # type: ignore
        logging.info(f"已设置API密钥: {service.CORRECT_API_KEY}")
    
    if config.if_preload is not None:
        service.if_preload = config.if_preload
        logging.info(f"已设置预加载: {service.if_preload}")

        if service.if_preload and not service.if_loaded:
            service.model = await service.load_model()
            service.if_loaded = True
            logging.info("模型已预加载")

    return {"message": "配置已更新"}


if __name__ == "__main__":
    logging.warning("This is a model service, you can't run this separately.")