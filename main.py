# 有关tts的详细配置请移步service.py
from astrbot.api.event import filter, AstrMessageEvent              # pyright: ignore[reportMissingImports]
from astrbot.api.star import Context, Star, register                # pyright: ignore[reportMissingImports]
from astrbot.api.provider import ProviderRequest                    # pyright: ignore[reportMissingImports]
from astrbot.api.message_components import *                        # pyright: ignore[reportMissingImports]
from astrbot.core.utils.astrbot_path import get_astrbot_data_path   # pyright: ignore[reportMissingImports]
from multiprocessing import Process
from astrbot.api import logger                                      # pyright: ignore[reportMissingImports] 
from typing import Callable, Optional, Any
from pathlib import Path
import aiohttp
import asyncio
import atexit
import os
from random import random

#端口在这里改
port = "5210"  # 默认端口号

# 锁文件路径
lock_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "child_process.lock")
temp_dir = os.path.join(get_astrbot_data_path(), "temp")
output_path = os.path.join(temp_dir,"output.wav")

class Misc:
    def _get_prompt_wav_path(self, generation_config):
        """获取提示音频文件路径"""
        prompt_wav_filename = generation_config.get("prompt_wav", "prompt_白芷.wav")
        sounds_dir = Path(__file__).parent / "sounds"
        return os.path.join(sounds_dir, prompt_wav_filename)
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        # 确保sounds目录存在
        sounds_dir = Path(__file__).parent / "sounds"
        os.makedirs(sounds_dir, exist_ok=True)
        
        # 确保outputs目录存在
        outputs_dir = Path(__file__).parent / "outputs"
        os.makedirs(outputs_dir, exist_ok=True)

class RandomTTS:
    """随机将文字转为语音"""
    def __init__(self) -> None:
        pass
    
    @staticmethod
    async def random_replace(chain: list, random_factor: float, server_ip: str, CORRECT_API_KEY: str):
        randfloat = random()
        if randfloat <= random_factor:
            logger.info(f"Random TTS triggered,factor: {randfloat} in {random_factor}")
            if all(isinstance(e, Plain) for e in chain):                                # type: ignore  #如果chain里都为Plain元素
                text_parts = [msg.text for msg in chain if isinstance(msg, Plain)]      # type: ignore  #将Plain中的text提取出来，组成列表
                text_whole = ''.join(text_parts)                                                        # 合并所有 Plain 消息的文本
                chain.clear()                                                                           #清除chain里的所有元素
                if text_whole != "":
                    try:
                        wav_path = await manager.post_generate_request_with_session_auth(
                            server_ip,
                            port,
                            text_whole,
                            CORRECT_API_KEY,
                            output_path,                                            #这里用全局定义的
                            timeout_seconds=120.0                                   #不够用再改 
                            )
                        chain = [
                            Record.fromFileSystem(wav_path) # type: ignore
                        ]
                    except Exception as e:                                          #返回纯文字以防止影响体验
                        logger.error("Error when trying randomly send vocal message turn to plain text instead")
                        chain = [
                            Plain(text_whole) # type: ignore
                        ]
                    return chain
                else:
                    pass
            else:
                pass
        else:
            return chain

randomtts = RandomTTS()

class TTSManager:
    """TTS功能管理类"""
    def __init__(self):
        self.on_init = True  # 标记是否为初始化阶段
    
    async def async_retry_request(
            self,
            request_func: Callable,
            max_retries: int = 20,
            initial_retry_delay: float = 1.0,
            max_retry_delay: float = 60.0,
            backoff_factor: float = 2.0,
            retry_exceptions: tuple = (asyncio.TimeoutError, aiohttp.ClientError),
            **kwargs
        ) -> Any:
        """
        通用的异步重试请求函数
        
        Args:
            request_func: 要执行的请求函数
            max_retries: 最大重试次数
            initial_retry_delay: 初始重试延迟（秒）
            max_retry_delay: 最大重试延迟（秒）
            backoff_factor: 退避因子
            retry_exceptions: 需要重试的异常类型
            **kwargs: 传递给请求函数的参数
        
        Returns:
            请求函数的返回结果
        
        Raises:
            ConnectionError: 所有重试都失败时抛出
            Exception: 不可重试的异常
        """
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                result = await request_func(**kwargs)
                return result
                
            except retry_exceptions as e:
                last_error = e
                retry_count += 1
                if retry_count > max_retries:
                    break
                    
                delay = min(
                    initial_retry_delay * (backoff_factor ** (retry_count - 1)),
                    max_retry_delay
                )
                
                logger.warning(
                    f"请求失败({str(e)}), 正在进行第 {retry_count}/{max_retries} 次重试, "
                    f"等待 {delay:.1f} 秒后重试..."
                )
                
                await asyncio.sleep(delay)
                
            except Exception as e:
                logger.error(f"发生不可重试的错误: {str(e)}")
                raise
        
        logger.error(f"所有重试均失败, 最后错误: {str(last_error)}")
        raise ConnectionError(f"无法完成请求, 重试 {max_retries} 次后失败") from last_error


    # 具体的请求函数
    async def _post_config_request(
        self,
        server_ip: str,
        port: str,
        prompt_wav: str,
        CORRECT_API_KEY: str,
        model_ver: str,
        max_text_tokens_per_sentence: int,
        timeout_seconds: Optional[float] = 60.0,
        **kwargs
    ) -> dict:
        """具体的POST配置请求实现"""
        url = f"http://{server_ip}:{port}/config"
        
        # 构建基础payload
        payload = {
            "prompt_wav": prompt_wav,
            "model_ver": model_ver,
            "max_text_tokens_per_sentence": max_text_tokens_per_sentence,
            "CORRECT_API_KEY": CORRECT_API_KEY
        }
        
        # 添加所有可选参数（如果存在于kwargs中）
        config_fields = [
            "if_remove_think_tag", "if_preload", "if_split_text", "if_remove_emoji",
            "use_fp16", "use_cuda_kernel", "use_deepspeed", "use_emo_text", "use_random", "verbose",
            "emo_alpha", "emo_audio_prompt", "emo_vector", "emo_text", "interval_silence"
        ]
        
        for field in config_fields:
            if field in kwargs:
                payload[field] = kwargs[field]
        
        headers = {
            'Authorization': f'Bearer {CORRECT_API_KEY}',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                logger.info(f"请求成功: {result}")
                return result

    async def _post_generate_request(
            self,
            server_ip: str,
            port: str,
            text: str,
            CORRECT_API_KEY: str,
            output_path: str,
            timeout_seconds: Optional[float] = 60.0,
        ) -> str:
            """具体的POST生成请求实现"""
            url = f"http://{server_ip}:{port}/audio/speech"
            payload = {
                "model": "",
                "input": text,
                "voice": ""
            }

            headers = {
                'Authorization': f'Bearer {CORRECT_API_KEY}',
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
            
            timeout = aiohttp.ClientTimeout(total=timeout_seconds)
            async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    
                    # 检查响应内容类型
                    content_type = response.headers.get('Content-Type', '')
                    
                    if 'audio/wav' in content_type or 'audio/x-wav' in content_type:
                        # 处理音频文件响应
                        output_path_path = Path(output_path)
                        output_path_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        with open(output_path_path, 'wb') as f:
                            while True:
                                chunk = await response.content.read(1024)
                                if not chunk:
                                    break
                                f.write(chunk)
                        
                        logger.info(f"音频文件成功保存到: {output_path_path}")
                        return str(output_path_path)
                    else:
                        # 如果不是音频文件，尝试解析为JSON
                        result = await response.json()
                        logger.info(f"请求成功: {result}")
                    return result

    # 重构后的原始方法
    async def post_config_with_session_auth(
            self,
            server_ip: str,
            port: str,
            prompt_wav: str,
            CORRECT_API_KEY: str,
            model_ver: str,
            max_text_tokens_per_sentence: int,
            timeout_seconds: Optional[float] = 60.0,
            max_retries: int = 20,
            initial_retry_delay: float = 1.0,
            max_retry_delay: float = 60.0,
            backoff_factor: float = 2.0,
            **kwargs
        ) -> dict:
        """发送带认证的POST请求到指定服务器，具有自动重试机制"""
        return await self.async_retry_request(
            request_func=self._post_config_request,
            max_retries=max_retries,
            initial_retry_delay=initial_retry_delay,
            max_retry_delay=max_retry_delay,
            backoff_factor=backoff_factor,
            server_ip=server_ip,
            port=port,
            prompt_wav=prompt_wav,
            CORRECT_API_KEY=CORRECT_API_KEY,
            model_ver=model_ver,
            max_text_tokens_per_sentence=max_text_tokens_per_sentence,
            timeout_seconds=timeout_seconds,
            **kwargs
        )

    async def post_generate_request_with_session_auth(
        self,
        server_ip: str,
        port: str,
        text: str,
        CORRECT_API_KEY: str,
        output_path: str,
        timeout_seconds: Optional[float] = 60.0,
        max_retries: int = 20,
        initial_retry_delay: float = 1.0,
        max_retry_delay: float = 60.0,
        backoff_factor: float = 2.0
    ) -> str:
        """发送带认证的POST请求到指定服务器，具有自动重试机制，返回保存的音频文件路径"""
        return await self.async_retry_request(
            request_func=self._post_generate_request,
            max_retries=max_retries,
            initial_retry_delay=initial_retry_delay,
            max_retry_delay=max_retry_delay,
            backoff_factor=backoff_factor,
            server_ip=server_ip,
            port=port,
            text=text,
            CORRECT_API_KEY=CORRECT_API_KEY,
            output_path=output_path,
            timeout_seconds=timeout_seconds
        )

    @staticmethod
    def cleanup():
        """清理锁文件"""
        if os.path.exists(lock_file_path):
            os.remove(lock_file_path)

    @staticmethod
    def child_process_function():
        """子进程执行的函数"""
        from .service import run_service
        run_service()  # 启动服务

    @staticmethod
    def terminate_child_process(child_process):
        """手动终止子进程"""
        if child_process and child_process.is_alive():
            child_process.terminate()
            child_process.join()
            logger.info("Service.py process terminated.")
        TTSManager.cleanup()

    @staticmethod
    def start_child_process():
        """启动子进程的函数"""
        if os.path.exists(lock_file_path):
            if manager.on_init:
                TTSManager.cleanup()
            else:
                logger.warning("Another instance is already running.")
                return None
        
        with open(lock_file_path, 'w') as f:
            f.write("Locked")
        
        atexit.register(TTSManager.cleanup)
        
        p = Process(target=TTSManager.child_process_function, args=())
        p.start()
        logger.info("Sub process started")
        return p
    
manager = TTSManager()
misc = Misc()

@register("astrbot_plugin_index_tts", "xiewoc ", "基于index-tts对AstrBot的语音转文字(TTS)补充", "1.0.4", "https://github.com/xiewoc/astrbot_plugin_spark_tts")
class AstrbotPluginIndexTTS(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        # 基础配置项
        self.reduce_parenthesis = config.get('if_reduce_parenthesis', False)
        self.if_remove_think_tag = config.get("if_remove_think_tag", False)
        self.if_split_text = config.get("if_split_text", False)
        self.if_remove_emoji = config.get("if_remove_emoji", False)
        self.use_fp16 = config.get("use_fp16", False)
        self.use_cuda_kernel = config.get("use_cuda_kernel", False)
        self.use_deepspeed = config.get("use_deepspeed", False)
        self.use_emo_text = config.get("use_emo_text", False)
        self.use_random = config.get("use_random", False)
        self.verbose = config.get("verbose", False)
        self.child_process = None

        # 生成配置子项
        sub_config_generation = config.get('generation', {})
        self.prompt_wav = misc._get_prompt_wav_path(sub_config_generation)
        self.emo_alpha = sub_config_generation.get("emo_alpha", 1.0)
        self.interval_silence = sub_config_generation.get("interval_silence", 120)
        self.emo_audio_prompt = sub_config_generation.get("emo_audio_prompt", "")
        self.emo_text = sub_config_generation.get("emo_text", "")
        self.emo_vector = sub_config_generation.get("emo_vector", [0, 0, 0, 0, 0, 0, 0, 0])
        self.model_ver = sub_config_generation.get("model_ver", "1.5")
        self.if_preload = sub_config_generation.get("if_preload", False)
        self.max_text_tokens_per_sentence = sub_config_generation.get("max_text_tokens_per_sentence", 100)
        self.if_random_tts = sub_config_generation.get("if_random_tts", False)
        self.random_factor = sub_config_generation.get("random_factor", 0.3)

        # 服务器配置子项
        sub_config_serve = config.get('serve_config', {})
        self.server_ip = sub_config_serve.get("server_ip", "127.0.0.1")
        self.if_seperate_serve = sub_config_serve.get("if_seperate_serve", False)
        self.CORRECT_API_KEY = sub_config_serve.get("CORRECT_API_KEY", "")
        
        # 确保必要的目录存在
        misc._ensure_directories()
        
    async def initialize(self):
        try:
            logger.info("插件初始化中...")
            if not self.if_seperate_serve:
                self.child_process = manager.start_child_process()
                manager.on_init = False
                if self.child_process:
                    logger.info("TTS服务子进程已启动")

            try:
                params = {
                    "if_remove_think_tag": self.if_remove_think_tag,
                    "if_preload": self.if_preload,
                    "if_split_text": self.if_split_text,
                    "if_remove_emoji": self.if_remove_emoji,
                    "use_fp16": self.use_fp16,  # 添加其他可能需要的参数
                    "use_cuda_kernel": self.use_cuda_kernel,
                    "use_deepspeed": self.use_deepspeed,
                    "use_emo_text": self.use_emo_text,
                    "use_random": self.use_random,
                    "verbose": self.verbose,
                    "emo_alpha": self.emo_alpha,
                    "interval_silence": self.interval_silence,
                    "emo_audio_prompt": self.emo_audio_prompt,
                    "emo_text": self.emo_text,
                    "emo_vector": self.emo_vector
                }

                await manager.post_config_with_session_auth(
                    self.server_ip,
                    port,
                    self.prompt_wav,
                    self.CORRECT_API_KEY,
                    self.model_ver,
                    self.max_text_tokens_per_sentence,
                    timeout_seconds = 30.0,  # 首次连接使用较短超时
                    max_retries = 20,
                    initial_retry_delay = 3.0,
                    max_retry_delay = 80.0,
                    backoff_factor = 2.0,
                    **params
                )
            except Exception as e:
                logger.error(f"初始服务连接失败: {str(e)}")
                if not self.if_seperate_serve and self.child_process:
                    self.child_process.terminate()
                raise

        except Exception as e:
            logger.error(f"插件初始化失败: {str(e)}")
            raise

    async def terminate(self): 
        logger.info("已调用方法:Terminate,正在关闭")
        manager.on_init = False
        manager.terminate_child_process(self.child_process)

    @filter.command_group("tts_cfg_it")
    def tts_cfg_it(self):
        pass
    
    @tts_cfg_it.group("set")
    def set(self):
        pass
    
    @set.command("voice")
    async def voice(self, event: AstrMessageEvent, voice_name:str):
        self.prompt_wav = str ( Path(__file__).parent / "sounds" / voice_name )
        try:
            await manager.post_config_with_session_auth(
                self.server_ip,
                port,
                self.prompt_wav,
                self.CORRECT_API_KEY,
                self.model_ver,
                self.max_text_tokens_per_sentence,
                timeout_seconds = 30.0,  # 首次连接使用较短超时
                max_retries = 10,
                initial_retry_delay = 1.0,
                max_retry_delay = 20.0,
                backoff_factor = 2.0
            )
            await event.reply(f"语音源已更改为: {voice_name}")
        except Exception as e:
            logger.error(f"初始服务连接失败: {str(e)}")
            raise

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        '''用于触发随机文字转语音'''
        chain = event.get_result().chain
        if self.if_random_tts:
            chain = await randomtts.random_replace(
                chain,
                self.random_factor,
                self.server_ip,
                self.CORRECT_API_KEY
                )
        else:
            pass
        event.get_result().chain = chain

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        if self.reduce_parenthesis:
            req.system_prompt += "请简化输出文本，仅保留口语内容，删除描述动作、表情或心情的附加信息（如括号内的补充说明）。"

    @filter.llm_tool(name="send_vocal_msg_it")
    async def send_vocal_msg_it(self, event: AstrMessageEvent, text: str):
        '''发送语音消息

        Args:
            text (string): 要转换为语音的文本内容
        '''
        if text != "" and text is not None:
            try:
                params = {
                    "use_emo_text":True
                }
                await manager.post_config_with_session_auth(
                    self.server_ip,
                    port,
                    self.prompt_wav,
                    self.CORRECT_API_KEY,
                    self.model_ver,
                    self.max_text_tokens_per_sentence,
                    timeout_seconds = 30.0,  # 首次连接使用较短超时
                    max_retries = 10,
                    initial_retry_delay = 1.0,
                    max_retry_delay = 20.0,
                    backoff_factor = 2.0,
                    **params
                )

                wav_path = await manager.post_generate_request_with_session_auth(
                    self.server_ip,
                    port,
                    text,
                    self.CORRECT_API_KEY,
                    output_path,
                    timeout_seconds=90.0
                    )
                chain = [
                    Record.fromFileSystem(wav_path) # type: ignore
                ]
            except Exception as e:
                logger.error(f"语音消息生成失败: {e}")
                chain = [
                    Plain(text) # type: ignore
                ]
        else:
            pass    
        yield event.chain_result(chain)