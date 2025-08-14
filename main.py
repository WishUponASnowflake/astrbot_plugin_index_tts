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
from math import exp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 锁文件路径
lock_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "child_process.lock")
global on_init
on_init = True  # 标记是否为初始化阶段

class TTSManager:
    """TTS功能管理类"""
    
    @staticmethod
    async def post_with_session_auth(
        server_ip: str,
        port: str,
        if_remove_think_tag: bool,
        if_preload: bool,
        prompt_wav: str,
        CORRECT_API_KEY: str,
        model_ver: str,
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
    def terminate_child_process_on_exit(child_process):
        """注册函数，在主进程退出时终止子进程"""
        def cleanup_on_exit():
            if child_process and child_process.is_alive():
                child_process.terminate()
                child_process.join()
                logger.info("Service.py process terminated.")
            TTSManager.cleanup()
        atexit.register(cleanup_on_exit)

    @staticmethod
    def start_child_process():
        """启动子进程的函数"""
        if os.path.exists(lock_file_path):
            global on_init
            if on_init:
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

@register("astrbot_plugin_index_tts", "xiewoc ", "基于index-tts对AstrBot的语音转文字(TTS)补充", "1.0.0", "https://github.com/xiewoc/astrbot_plugin_spark_tts")
class AstrbotPluginIndexTTS(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.reduce_parenthesis = config.get('if_reduce_parenthesis', False)
        self.if_remove_think_tag = config.get("if_remove_think_tag", False)
        self.if_preload = False

        sub_config_misc = config.get('misc', {})
        sub_config_serve = config.get('serve_config', {})
        
        # 确保sounds目录存在
        sounds_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")
        os.makedirs(sounds_dir, exist_ok=True)
        
        self.prompt_wav = os.path.join(
            sounds_dir,
            sub_config_misc.get("prompt_wav", "")
        )
        self.model_ver = sub_config_misc.get("model_ver", "1.5")  # 添加默认值
        self.server_ip = sub_config_serve.get("server_ip", "127.0.0.1")  # 添加默认值
        self.if_seperate_serve = sub_config_serve.get("if_seperate_serve", False)
        self.CORRECT_API_KEY = sub_config_serve.get("CORRECT_API_KEY", "")
        
        # 确保outputs目录存在
        outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        
    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self):
        try:
            await TTSManager.download_repo()
                
            if not self.if_seperate_serve:
                child_process = TTSManager.start_child_process()
                global on_init
                on_init = False
                if child_process:
                    TTSManager.terminate_child_process_on_exit(child_process)
                    logger.info("TTS服务子进程已启动")

            await asyncio.sleep(3)  # 等待服务启动

                # 检查服务是否可用
            try:
                await TTSManager.post_with_session_auth(
                    self.server_ip,
                    "5210",
                    self.if_remove_think_tag,
                    self.if_preload,
                    self.prompt_wav,
                    self.CORRECT_API_KEY,
                    self.model_ver,
                    timeout_seconds=30.0  # 首次连接使用较短超时
                )
            except Exception as e:
                logger.error(f"初始服务连接失败: {str(e)}")
                if not self.if_seperate_serve and child_process:
                    child_process.terminate()
                raise

        except Exception as e:
            logger.error(f"TTS插件初始化失败: {str(e)}")
            raise

    @filter.on_llm_request()
    async def on_call_llm(self, event: AstrMessageEvent, req: ProviderRequest):
        if self.reduce_parenthesis:

            req.system_prompt += "请在输出的字段中减少使用括号括起对动作,心情,表情等的描写，尽量只剩下口语部分"
