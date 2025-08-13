import sys
import os
import re
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request, status, Depends
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import logging
#不使用astrbot的日志系统，为分布式部署留空间
from pydub import AudioSegment

logging.getLogger("pydub").setLevel(logging.WARNING) #忽略pydb的info输出

global if_remove_think_tag, CORRECT_API_KEY, model_ver, prompt_wav, model, if_preload, if_loaded
if_remove_think_tag = False
if_preload = False
if_loaded = False
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"outputs","output.wav")

sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),"index-tts"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"index-tts"))

from indextts.infer import IndexTTS

def LoadModel(model_path=model_path,model_ver = "1.5"):

    if not os.path.exists(model_path):
        logging.error(f"Model path {model_path} does not exist.")
        logging.info(f"Downloading model to {model_path}...")
        #下载模型
        from modelscope import snapshot_download 
        if model_ver == "1":
            model_path = os.path.join(model_path,"IndexTTS")
            snapshot_download("IndexTeam/Index-TTS", local_dir=model_path)
        elif model_ver == "1.5":
            model_path = os.path.join(model_path,"IndexTTS-1.5")
            snapshot_download("IndexTeam/IndexTTS-1.5", local_dir=model_path)
    else:
        logging.warning(f"Model path {model_path} already exists, skipping download.")
        if model_ver == "1":
            model_path = os.path.join(model_path,"IndexTTS")
        elif model_ver == "1.5":
            model_path = os.path.join(model_path,"IndexTTS-1.5")
        #如果模型已经存在，就不再下载
        pass

    
    cfg_path = os.path.join(model_path,"config.yaml")

    logging.warning(f"Using model path: {model_path}")
    logging.warning(f"Using config path: {cfg_path}")
    #打印模型路径和配置路径
    #is_fp16,device,use_cuda_kernal保留，日后要用再说

    global model
    model = IndexTTS(model_dir=model_path, cfg_path=cfg_path)
    if model is None:
        logging.error("Failed to load model.")
        raise Exception("Failed to load model.")

app = FastAPI()
security = HTTPBearer()

#去除思考标签这一块
#<think></think>标签是用来标记思考的文本
def remove_thinktag(text):
    if text:
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return cleaned_text
    else:
        return ''
    
#wav转mp3这一块
def wav2mp3(wav_path,script_path):
    audio = AudioSegment.from_wav(wav_path)
    audio.export(os.path.join(script_path, "output.mp3"), format="mp3", parameters=["-loglevel", "quiet"])
    os.remove(wav_path)
    mp3_path = os.path.join(script_path, "output.mp3")
    return mp3_path

#验证api key这一块
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    global CORRECT_API_KEY
    if credentials.scheme != "Bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme"
        )
    if credentials.credentials != CORRECT_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return credentials.credentials

# Define the request model for speech generation
class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str

class Config(BaseModel):
    if_remove_think_tag: bool
    if_preload: bool
    prompt_wav: str
    model_ver: str
    CORRECT_API_KEY: str

def run_service():
    uvicorn.run(app, host="0.0.0.0", port=5210)
    
@app.post("/audio/speech")
async def generate_speech(request: Request, speech_request: SpeechRequest, apikey: str = Depends(verify_api_key)):
    
    script_path = os.path.dirname(os.path.abspath(__file__))
    
    try:
        global if_remove_think_tag, CORRECT_API_KEY, model_ver, prompt_wav, if_preload, if_loaded, output_path
        if if_remove_think_tag == True:
            input_text = remove_thinktag(speech_request.input)
        else:
            input_text = speech_request.input
        
        if input_text != "":

            if if_preload == True:
                pass
            else:
                if (if_loaded == False):
                    logging.warning("Loading model...")
                    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
                    LoadModel(model_path= model_path, model_ver=model_ver)
                    if_loaded = True
                else:
                    pass
            global model
            if model is not None:
                model.infer(prompt_wav, speech_request.input, output_path)
            logging.warning(f"Speech generating at {output_path}")
           
        else:
            return ""

        if not output_path or not os.path.exists(output_path) or not os.access(output_path, os.R_OK):
            raise HTTPException(status_code=500, detail="Failed to generate speech")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # 使用FileResponse返回生成的语音文件
    return FileResponse(path=output_path, media_type='audio/wav', filename="output.wav")

# Define the configuration endpoint
@app.post("/config")
async def update_config(config: Config):
    global if_remove_think_tag, CORRECT_API_KEY, model_ver, prompt_wav, if_preload
    if config.if_remove_think_tag is not None:
        if_remove_think_tag = config.if_remove_think_tag
        logging.info(f"已设置去除思考标签功能: {if_remove_think_tag}")
    if config.prompt_wav:
        prompt_wav = config.prompt_wav
        logging.info(f"已设置音频输入文件: {prompt_wav}")
    if config.model_ver:
        model_ver = config.model_ver
        logging.info(f"已设置模型版本: {model_ver}")
    if config.CORRECT_API_KEY:
        CORRECT_API_KEY = config.CORRECT_API_KEY
        logging.info(f"已设置API密钥: {CORRECT_API_KEY}")
    if config.if_preload is not None:
        if_preload = config.if_preload
        logging.info(f"已设置预加载: {if_preload}")
        if if_preload == True:
            global model, if_loaded
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
            model = LoadModel(model_path= model_path, model_ver=model_ver)
            if_loaded = True
            logging.info("模型已预加载")
    return {"配置已更新"}

if __name__ == "__main__":
    logging.warning("This is a model ,you can't run this seperately.")