# 有关tts的详细配置请移步service.py
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api.provider import ProviderRequest
from astrbot.api.message_components import *
from multiprocessing import Process
from astrbot.api import logger
from typing import Optional
from pathlib import Path
import subprocess
import aiohttp
import asyncio
import atexit
import sys
import os
from random import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

#端口在这里改
port = "5210"  # 默认端口号

# 锁文件路径
lock_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "child_process.lock")
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"temp","output.wav")

class RandomTTS:
    """随机将文字转为语音"""
    def __init__(self) -> None:
        pass
    
    @staticmethod
    async def RandomReplace(chain: list, random_factor: float, server_ip: str, CORRECT_API_KEY: str):
        randfloat = random()
        if randfloat <= random_factor:
            logger.info(f"Random TTS triggered,factor: {randfloat} in {random_factor}")
            if all(isinstance(e, Plain) for e in chain):                            #如果chain里都为Plain元素
                text_parts = [msg.text for msg in chain if isinstance(msg, Plain)]     #将Plain中的text提取出来，组成列表
                text_whole = ''.join(text_parts)                                    # 合并所有 Plain 消息的文本
                chain.clear()                                                       #清除chain里的所有元素
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
                            Record.fromFileSystem(wav_path)
                        ]
                    except Exception as e:                                          #返回纯文字以防止影响体验
                        logger.error("Error when trying randomly send vocal message turn to plain text instead")
                        chain = [
                            Plain(text_whole)
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
    
    @staticmethod
    async def post_config_with_session_auth(
        server_ip: str,
        port: str,
        if_remove_think_tag: bool,
        if_preload: bool,
        prompt_wav: str,
        CORRECT_API_KEY: str,
        model_ver: str,
        max_text_tokens_per_sentence: int,
        timeout_seconds: Optional[float] = 60.0,
        max_retries: int = 20,
        initial_retry_delay: float = 1.0,
        max_retry_delay: float = 60.0,
        backoff_factor: float = 2.0
    ) -> dict:
        """发送带认证的POST请求到指定服务器，具有自动重试机制"""
        url = f"http://{server_ip}:{port}/config"
        payload = {
            "if_remove_think_tag": if_remove_think_tag,
            "if_preload": if_preload,
            "prompt_wav": prompt_wav,
            "model_ver": model_ver,
            "max_text_tokens_per_sentence": max_text_tokens_per_sentence,
            "CORRECT_API_KEY": CORRECT_API_KEY
        }

        headers = {
            'Authorization': f'Bearer {CORRECT_API_KEY}',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            timeout = aiohttp.ClientTimeout(total=timeout_seconds)
            
            try:
                async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                    async with session.post(url, json=payload) as response:
                        response.raise_for_status()
                        result = await response.json()
                        logger.info(f"请求成功: {result}")
                        return result
                        
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
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
        raise ConnectionError(f"无法连接到服务器, 重试 {max_retries} 次后失败") from last_error
    
    @staticmethod
    async def post_generate_request_with_session_auth(
        server_ip: str,
        port: str,
        text: str,
        CORRECT_API_KEY: str,
        output_path: str = output_path,  # 添加输出路径参数
        timeout_seconds: Optional[float] = 60.0,
        max_retries: int = 20,
        initial_retry_delay: float = 1.0,
        max_retry_delay: float = 60.0,
        backoff_factor: float = 2.0
    ) -> str:
        """发送带认证的POST请求到指定服务器，具有自动重试机制，返回保存的音频文件路径"""
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
        
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            timeout = aiohttp.ClientTimeout(total=timeout_seconds)
            
            try:
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
                        
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
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
        raise ConnectionError(f"无法连接到服务器, 重试 {max_retries} 次后失败") from last_error

    @staticmethod
    async def download_repo() -> None:
        """异步安全地下载 Index TTS 仓库"""
        repo_dir = Path(__file__).parent / "index-tts"
        repo_url = "https://github.com/index-tts/index-tts.git"
        
        try:
            if repo_dir.exists():
                logger.info("Index TTS Github Repo found, skipping download.")
                return
                
            repo_dir.mkdir(parents=True, exist_ok=True)
            
            if not await TTSManager.is_git_available():
                raise RuntimeError("Git is not installed or not in PATH")
            
            logger.info("Downloading Index TTS Github Repo...")
            
            await TTSManager.run_command(
                f"git clone --recursive {repo_url} {repo_dir}",
                timeout=600
            )
            
            logger.info("Successfully downloaded Index TTS Github Repo")
            
        except asyncio.TimeoutError:
            logger.error("Git clone operation timed out")
            raise
        except Exception as e:
            logger.error(f"Failed to download repository: {str(e)}")
            if repo_dir.exists():
                try:
                    await TTSManager.run_command(f"rm -rf {repo_dir}")
                except:
                    pass
            raise

    @staticmethod
    async def is_git_available() -> bool:
        """检查系统是否安装了git"""
        try:
            await TTSManager.run_command("git --version", timeout=5)
            return True
        except:
            return False

    @staticmethod
    async def run_command(command: str, timeout: Optional[float] = None) -> str:
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
                logger.error(f"Command failed with error: {error_msg}")
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
    def cleanup():
        """清理锁文件"""
        if os.path.exists(lock_file_path):
            os.remove(lock_file_path)

    @staticmethod
    def child_process_function():
        """子进程执行的函数"""
        import service 
        service.run_service()

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

@register("astrbot_plugin_index_tts", "xiewoc ", "基于index-tts对AstrBot的语音转文字(TTS)补充", "1.0.1", "https://github.com/xiewoc/astrbot_plugin_spark_tts")
class AstrbotPluginIndexTTS(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.reduce_parenthesis = config.get('if_reduce_parenthesis', False)
        self.if_remove_think_tag = config.get("if_remove_think_tag", False)
        self.if_preload = False
        self.child_process = None

        sub_config_misc = config.get('misc', {})
        sub_config_serve = config.get('serve_config', {})
        
        # 确保sounds目录存在
        sounds_dir = Path(__file__).parent / "sounds"
        os.makedirs(sounds_dir, exist_ok=True)
        
        self.prompt_wav = os.path.join(
            sounds_dir,
            sub_config_misc.get("prompt_wav", "")
        )
        self.model_ver = sub_config_misc.get("model_ver", "1.5") 
        self.max_text_tokens_per_sentence = sub_config_misc.get("max_text_tokens_per_sentence",100)
        self.if_random_tts = sub_config_misc.get("if_random_tts",False)
        self.random_factor = sub_config_misc.get("random_factor",0.3)

        self.server_ip = sub_config_serve.get("server_ip", "127.0.0.1")  
        self.if_seperate_serve = sub_config_serve.get("if_seperate_serve", False)
        self.CORRECT_API_KEY = sub_config_serve.get("CORRECT_API_KEY", "")
        
        # 确保outputs目录存在
        outputs_dir = Path(__file__).parent / "outputs"
        os.makedirs(outputs_dir, exist_ok=True)
        
    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self):
        try:
            await manager.download_repo()
                
            if not self.if_seperate_serve:
                self.child_process = manager.start_child_process()
                manager.on_init = False
                if self.child_process:
                    logger.info("TTS服务子进程已启动")

            #await asyncio.sleep(2)  # 等待服务启动

            # 检查服务是否可用
            try:
                await manager.post_config_with_session_auth(
                    self.server_ip,
                    port,
                    self.if_remove_think_tag,
                    self.if_preload,
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
                self.if_remove_think_tag,
                self.if_preload,
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
        except Exception as e:
            logger.error(f"初始服务连接失败: {str(e)}")
            if not self.if_seperate_serve and self.child_process:
                self.child_process.terminate()
            raise

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        chain = event.get_result().chain
        if self.if_random_tts:
            chain = await randomtts.RandomReplace(
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
                wav_path = await manager.post_generate_request_with_session_auth(
                    self.server_ip,
                    port,
                    text,
                    self.CORRECT_API_KEY,
                    output_path,
                    timeout_seconds=90.0
                    )
                chain = [
                    Record.fromFileSystem(wav_path)
                ]
            except Exception as e:
                logger.error(f"语音消息生成失败: {e}")
                chain = [
                    Plain(text)
                ]
        else:
            pass    
        yield event.chain_result(chain)