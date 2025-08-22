import sys
import os
import re
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import FastAPI, HTTPException, Request, status, Depends
from fastapi.responses import FileResponse
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
import subprocess
import uvicorn
import logging
import asyncio

logging.getLogger("pydub").setLevel(logging.WARNING)

#端口设置在这里
port = 5210

"""
todo:

class TTSServiceVLLM:
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
"""

class Misc:
    @staticmethod
    async def download_repo() -> None:
        """异步安全地下载 Index TTS 仓库"""
        repo_dir = Path(__file__).parent / "index-tts"
        repo_url = "https://github.com/index-tts/index-tts.git"
        
        try:
            if repo_dir.exists():
                logging.info("Index TTS Github Repo found, skipping download.")
                return
                
            repo_dir.mkdir(parents=True, exist_ok=True)
            
            if not await Misc._is_git_available():
                raise RuntimeError("Git is not installed or not in PATH")
            
            logging.info("Downloading Index TTS Github Repo...")
            
            await Misc._run_command(
                f"git clone --recursive {repo_url} {repo_dir}",
                timeout=600
            )
            
            logging.info("Successfully downloaded Index TTS Github Repo")
            
        except asyncio.TimeoutError:
            logging.error("Git clone operation timed out")
            raise
        except Exception as e:
            logging.error(f"Failed to download repository: {str(e)}")
            if repo_dir.exists():
                try:
                    await Misc._run_command(f"rm -rf {repo_dir}")
                except:
                    pass
            raise

    @staticmethod
    async def _is_git_available() -> bool:
        """检查系统是否安装了git"""
        try:
            await Misc._run_command("git --version", timeout=5)
            return True
        except:
            return False

    @staticmethod
    async def _run_command(command: str, timeout: Optional[float] = None) -> str:
        """异步执行命令并返回输出"""
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            start_new_session=True
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            if process.returncode != 0 and process.returncode is not None:
                error_msg = stderr.decode().strip()
                logging.error(f"Command failed with error: {error_msg}")
                raise subprocess.CalledProcessError(
                    process.returncode, 
                    command, 
                    stdout, 
                    stderr
                )
                
            return stdout.decode().strip()
            
        except asyncio.TimeoutError:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
            raise
        except Exception as e:
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise
    
    @staticmethod
    def remove_thinktag(text): 
        """去除<think>标签"""
        if text:
            return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return ''
    
    @staticmethod
    def remove_emoji(text):
        emoji_pattern = re.compile(
            "["
            # 表情符号
            "\U0001F600-\U0001F64F"  # Emoticons
            # 杂项符号和象形文字
            "\U0001F300-\U0001F5FF"  # Miscellaneous Symbols and Pictographs
            # 交通和地图符号
            "\U0001F680-\U0001F6FF"  # Transport and Map Symbols
            # 补充符号和象形文字
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            # 符号和象形文字扩展-A
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            # 杂项符号
            "\U00002600-\U000026FF"  # Miscellaneous Symbols
            # 丁巴特文符号（补充）
            "\U0001F000-\U0001F02F"  # Mahjong Tiles etc.
            "]+", 
            flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)
    
    def merge_audio_files(self, audio_files: list, output_path):
        """合并音频文件"""
        from pydub import AudioSegment # 如果调用了 merge_audio_files() 函数再import
        
        combined = AudioSegment.empty()
        for file in audio_files:
            if os.path.exists(file):
                audio_segment = AudioSegment.from_file(file)
                combined += audio_segment
            else:
                logging.warning(f"Audio file {file} does not exist, skipping.")
        
        combined.export(output_path, format="wav")
        return output_path

class TTSService:
    def __init__(self):
        self.if_remove_think_tag = False
        self.if_preload = False
        self.if_split_text = False
        self.if_remove_emoji = False
        self.if_loaded = False
        self.model = None
        self.CORRECT_API_KEY:str = ""
        self.model_ver = "1.5"
        self.prompt_wav = ""
        self.max_text_tokens_per_sentence = 100
        self.thread_pool: ThreadPoolExecutor
        
        # 初始化路径
        self.dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        self.output_path_final = os.path.join(self.dir_path, "outputs", "output.wav")
        
        # 添加必要的路径到sys.path
        sys.path.insert(0, os.path.join(self.dir_path, "index-tts"))

    async def DownloadModel(self, model_ver=None):
        """下载并加载TTS模型"""
        model_ver = model_ver or self.model_ver
        if model_ver not in ["1", "1.5"]:
            raise ValueError("Unsupported model version. Supported versions are '1' and '1.5'.")

        from modelscope import snapshot_download

        actual_model_path = os.path.join(self.dir_path, f"IndexTTS{'-1.5' if model_ver == '1.5' else ''}")

        if not os.path.exists(actual_model_path):
            logging.info(f"Downloading model version {model_ver} to {actual_model_path}...")

            # 使用线程池处理模型下载
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
            actual_model_path = os.path.join(self.dir_path, "IndexTTS")
        elif model_ver == "1.5":
            actual_model_path = os.path.join(self.dir_path, "IndexTTS-1.5")

        cfg_path = os.path.join(actual_model_path, "config.yaml")
        
        if not os.path.exists(actual_model_path):
            try:
                await self.DownloadModel(model_ver)
            except:
                raise

        logging.info(f"Using model path: {actual_model_path}")
        logging.info(f"Using config path: {cfg_path}")

        try:
            from indextts.infer import IndexTTS # type: ignore

            loop = asyncio.get_event_loop()

            self.model =  await loop.run_in_executor(
                    self.thread_pool, 
                    lambda: IndexTTS(model_dir=actual_model_path, cfg_path=cfg_path)
                )
            logging.info("Model loaded successfully.")
            
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail="Model loading failed")

        if self.model is None:
            logging.error("Failed to load model.")
            raise Exception("Failed to load model.")
        
        self.if_loaded = True
        return self.model


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
misc = Misc()
service = TTSService()
app = FastAPI()

def run_service():
    uvicorn.run(app, host="0.0.0.0", port=port)

@app.on_event("startup")
async def start_up():
    # 在服务类或模块级别创建线程池
    try:
        await misc.download_repo()  # 确保在服务启动前下载仓库
    except Exception as e:
        logging.error(f"Failed to download repository: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download repository")
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
    # 非必须参数
    if_remove_think_tag: bool = Field(False, description="是否移除思考标签")
    if_preload: bool = Field(False, description="是否预加载")
    if_split_text:bool = Field(False, description="是否分割文本")
    if_remove_emoji:bool = Field(False, description="是否移除表情(emoji)")

    # 必须参数
    prompt_wav: str = Field(..., description="输入音源")
    model_ver: str = Field(..., description="模型版本")
    max_text_tokens_per_sentence: int = Field(..., description="单句最大Token数")
    CORRECT_API_KEY: str = Field(..., description="API KEY")

@app.post("/audio/speech")
async def generate_speech(
    request: Request, 
    speech_request: SpeechRequest, 
    apikey: str = Depends(service.verify_api_key)
):
    try:
        if not speech_request.input:
            raise HTTPException(status_code=400, detail="Input text cannot be empty")
        
        input_text = (misc.remove_thinktag(speech_request.input) 
                    if service.if_remove_think_tag 
                    else speech_request.input) # 去除思考标签
        
        input_text = misc.remove_emoji(input_text) if service.if_remove_emoji else input_text

        
        # 如果模型未加载，使用线程池加载
        if not service.if_preload and not service.if_loaded:
            logging.info("Loading model...")
            service.model = await service.load_model(service.model_ver)

        if (len(input_text) > service.max_text_tokens_per_sentence) or service.if_split_text: 
            # 如果需要分割文本或超过最大Token数，进行分割处理

            pattern = re.compile(r'^[\u4e00-\u9fa5]+$')
            if bool(pattern.match(input_text)):
                # 如果是中文文本，按句子分割
                text_list = re.split(r'(?<=[。！？])', input_text)
            else:
                pattern = r'(?<=[.!?])(?!(?:\d|[A-Za-z]|\.)) +'
                text_list = [s.strip() for s in re.split(pattern, input_text) if s.strip()]
        else:
            text_list = [input_text]

        output_wav_list = []  # 用于存储每段生成的音频文件路径

        # 使用线程池处理模型推理
        loop = asyncio.get_event_loop()

        for idx, text in enumerate(text_list):          # 遍历分割后的文本列表,分次合成

            output_path = os.path.join(service.dir_path, "outputs", f"opt_{idx+1}.wav") # 每次重新赋值 -> 避免覆盖

            logging.info(f"Generating speech for segment {idx+1}/{len(text_list)}...")
            if service.model is not None:
                # 使用线程池执行模型推理
                await loop.run_in_executor(
                    service.thread_pool, 
                    lambda: service.model.infer(service.prompt_wav, text, output_path, service.max_text_tokens_per_sentence) # type: ignore
                )
                logging.info(f"Speech generating at {output_path}")
            else:
                raise HTTPException(status_code=500, detail="Model not loaded")

            if not os.path.exists(output_path):
                raise HTTPException(status_code=500, detail=f"Failed to generate speech segment {idx+1}/{len(text_list)}")
            else:
                output_wav_list.append(output_path)

        if len(output_wav_list) == 1:
            output_path = output_wav_list[0]

        elif len(output_wav_list) > 1:
            output_path = await loop.run_in_executor(
                    service.thread_pool, 
                    lambda: misc.merge_audio_files(output_wav_list, service.output_path_final)
                )

        else:
            raise HTTPException(status_code=500, detail="Speech failed to be generated.Output wav list is void")
        
        output_wav_list = [] # 清空列表，准备下次使用？

        return FileResponse(
            path=output_path, 
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

    if config.if_preload is not None:
        service.if_preload = config.if_preload
        logging.info(f"已设置预加载: {service.if_preload}")

        if service.if_preload and not service.if_loaded:
            service.model = await service.load_model()
            service.if_loaded = True
            logging.info("模型已预加载")
    
    if config.if_remove_emoji:
        service.if_remove_emoji = config.if_remove_emoji
        logging.info(f"已设置是否去除emoji: {service.if_remove_emoji}")

    if config.if_split_text:
        service.if_split_text = config.if_split_text
        logging.info(f"已设置是否分割文本: {service.if_split_text}")
    
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
        service.CORRECT_API_KEY = config.CORRECT_API_KEY                
        logging.info(f"已设置API密钥: {service.CORRECT_API_KEY}")
    

    return {"message": "配置已更新"}

if __name__ == "__main__":
    logging.warning("This is a model service, you can't run this separately.")
    logging.warning("But you can run this model using `service.run_service()` to start the service after import in cli.")