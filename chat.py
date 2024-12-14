from dataclasses import dataclass
from typing import List
import json
import httpx
from pathlib import Path
from openai import OpenAI
from knowledge_retriever import KnowledgeRetriever, RetrievalMode
from config import *

# 全局开关
DEBUG_MODE = False  # 在这里控制debug模式

PLATFORM_KNOWLEDGE_BASE = {
    "bilibili": "./bilibili_knowledge_base",
    "weibo": "./weibo_knowledge_base",
    "zhihu": "./zhihu_knowledge_base",
}
PLATFORM_NAME = {"bilibili": "B站", "weibo": "微博", "zhihu": "知乎"}

GENERATE_QUERY_PROMPT = "prompt/generate_query.txt"


@dataclass
class QueryResult:
    """查询结果数据类"""

    mode: RetrievalMode
    query: str
    entities: List[str]


@dataclass
class DialogueEntry:
    """对话条目数据类"""

    speaker: str
    content: str


class DialogueContext:
    """对话上下文管理类"""

    def __init__(self, max_history: int = 15):
        self.history: List[DialogueEntry] = []
        self.max_history = max_history

        # 追踪策略使用情况
        self.strategy_history: List[str] = []  # 记录策略使用顺序
        self.strategy_counts = {
            "FAST": 0,  # 快速检索
            "ASSOCIATE": 0,  # 联想检索
            "RELATION": 0,  # 关系检索
            "COMMUNITY": 0,  # 社区检索
        }

    def add_dialogue_group(self, entries: List[DialogueEntry]):
        """添加一组对话"""
        self.history.extend(entries)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def format_history(self) -> str:
        if not self.history:
            return "无历史对话"
        return "\n".join(
            [f"{entry.speaker}: {entry.content}" for entry in self.history]
        )

    def add_strategy(self, strategy: str):
        """记录使用的策略"""
        self.strategy_history.append(strategy)
        self.strategy_counts[strategy] += 1
        # 只保留最近的策略历史
        if len(self.strategy_history) > self.max_history:
            oldest_strategy = self.strategy_history.pop(0)
            self.strategy_counts[oldest_strategy] -= 1

    def get_strategy_info(self) -> str:
        """获取策略使用情况的描述"""
        total_strategies = sum(self.strategy_counts.values())
        if total_strategies == 0:
            return "尚未使用任何策略"

        recent_strategies = self.strategy_history[-3:] if self.strategy_history else []

        strategy_info = []
        strategy_info.append("策略使用情况：")
        for strategy, count in self.strategy_counts.items():
            percentage = (count / total_strategies * 100) if total_strategies > 0 else 0
            strategy_info.append(f"- {strategy}: {count}次 ({percentage:.1f}%)")

        strategy_info.append(f"最近使用的策略: {', '.join(recent_strategies)}")

        return "\n".join(strategy_info)


class Chat:
    def __init__(
        self,
        character_name: str,
    ):
        self.character_name = character_name
        self.context = DialogueContext()
        self.client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

        # 初始化时创建prompt缓存
        self._initialize_prompt_cache()

        platform = character_name.lower()
        if platform in PLATFORM_KNOWLEDGE_BASE:
            knowledge_base_path = PLATFORM_KNOWLEDGE_BASE[platform]
            print(f"\n[正在加载 {character_name} 的知识库: {knowledge_base_path}]")
            self.knowledge_retriever = KnowledgeRetriever(knowledge_base_path)
        else:
            print(f"\n[错误] 未找到角色 {character_name} 对应的知识库")
            self.knowledge_retriever = None

    def _debug_print(self, *args, **kwargs):
        """Debug模式下的打印函数"""
        if DEBUG_MODE:
            print(*args, **kwargs)

    def _initialize_prompt_cache(self):
        """初始化prompt缓存

        读取generate_query.txt文件内容并创建缓存，用于后续查询时复用。
        缓存的TTL设置为300秒（5分钟），使用tag 'generate_query_prompt'标识。
        如果缓存创建失败会打印错误信息但不会中断程序运行。
        """
        try:
            # 1. 读取prompt文件内容
            with open(GENERATE_QUERY_PROMPT, "r", encoding="utf-8") as f:
                prompt_content = f.read()

            # 2. 设置缓存URL - 确保没有重复的斜杠
            base_url = str(self.client.base_url).rstrip("/")
            cache_url = f"{base_url}/caching"

            # 3. 准备请求头
            headers = {
                "Authorization": f"Bearer {self.client.api_key}",
                "Content-Type": "application/json",
            }

            # 4. 准备请求体
            payload = {
                "model": "moonshot-v1",
                "messages": [{"role": "system", "content": prompt_content}],
                "ttl": 300,  # 缓存5分钟
                "tags": ["generate_query_prompt"],
            }

            # 5. 发送创建缓存请求
            cache_response = httpx.post(
                cache_url, headers=headers, json=payload, timeout=10.0  # 设置10秒超时
            )

            # 6. 检查响应状态
            if cache_response.status_code != 200:
                error_msg = cache_response.text
                raise Exception(
                    f"创建缓存失败，状态码: {cache_response.status_code}, 错误信息: {error_msg}"
                )

            if self._debug_print:
                self._debug_print(f"[Debug] Prompt缓存初始化成功")

        except FileNotFoundError:
            print(f"Prompt文件未找到: {GENERATE_QUERY_PROMPT}")
        except httpx.RequestError as e:
            print(f"发送缓存请求失败: {str(e)}")
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {str(e)}")
        except Exception as e:
            print(f"初始化prompt缓存失败: {str(e)}")

    def _get_mode_specific_prompt(self, mode: RetrievalMode) -> str:
        """根据检索模式获取特定的系统提示词"""
        try:
            prompt_path = f"prompt/{mode.name.lower()}_debate.txt"
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_template = f.read()
            return prompt_template.format(
                platform_name=PLATFORM_NAME[self.character_name.lower()]
            )
        except Exception as e:
            print(f"读取{mode.name}模式提示词失败: {str(e)}")
            return ""

    def _analyze_input(self, topic: str) -> QueryResult:
        """分析需要回复的输入"""
        history_str = self.context.format_history()
        strategy_info = self.context.get_strategy_info()

        # 使用缓存的prompt
        messages = [
            # 使用缓存的system prompt
            {"role": "cache", "content": "tag=generate_query_prompt;reset_ttl=300"},
            {
                "role": "user",
                "content": f"你代表的平台是：{PLATFORM_NAME[self.character_name.lower()]}\n"
                f"策略使用情况：{strategy_info}\n"
                f"辩论主题：{topic}\n"
                f"历史对话：\n{history_str}",
            },
        ]

        if DEBUG_MODE:
            self._debug_print(f"查询信息: {messages[-1]['content']}")

        response = self.client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=messages,
            temperature=0.5,
            response_format={"type": "json_object"},
        )

        reply = response.choices[0].message.content

        try:
            result = json.loads(reply)
            mode = RetrievalMode[result["mode"]]
            # 记录使用的策略
            self.context.add_strategy(mode.name)

            self._debug_print(
                f"[Debug] 查询结果: mode={mode}, query={result['query']}, entities={result['entities']}"
            )

            return QueryResult(
                mode=mode, query=result["query"], entities=result["entities"]
            )
        except Exception as e:
            self._debug_print(f"[Debug] 查询生成失败: {str(e)}")

    def _stream_chat_response(self, character_prompt: str, input_context: str):
        """流式生成回复"""
        try:
            messages = [
                {"role": "system", "content": character_prompt},
                {"role": "user", "content": input_context},
            ]

            response = self.client.chat.completions.create(
                model="moonshot-v1-auto",
                messages=messages,
                temperature=0.7,
                stream=True,
            )

            full_response = ""
            current_sentence = ""
            end_marks = "。！？!?.;"
            consecutive_marks = False  # 标记是否正在处理连续的标点

            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content

                    for char in content:
                        current_sentence += char

                        if char in end_marks:
                            if not consecutive_marks:
                                consecutive_marks = True
                            continue

                        # 如果遇到非标点字符，且之前有标点
                        if consecutive_marks:
                            consecutive_marks = False
                            # 输出累积的句子
                            if sentence := current_sentence[
                                :-1
                            ].strip():  # 去掉最后一个字符
                                yield sentence
                            current_sentence = char  # 新句子从当前字符开始

            # 处理最后一个句子
            if current_sentence.strip():
                yield current_sentence.strip()

            return full_response

        except Exception as e:
            error_msg = f"生成回复时出错: {str(e)}"
            print(error_msg)
            return error_msg

    def generate_response(self, dialogue_group: List[DialogueEntry], topic):
        """根据新的一组对话生成响应"""
        try:
            if dialogue_group:
                # 添加历史对话
                self.context.add_dialogue_group(dialogue_group)

            # 分析输入
            query_result = self._analyze_input(topic)

            # 执行知识检索
            self._debug_print("\n[Debug] 开始知识检索...")
            retrieval_result = (
                self.knowledge_retriever.retrieve(
                    mode=query_result.mode,
                    query=query_result.query,
                    entities=query_result.entities,
                )
                if self.knowledge_retriever
                else None
            )

            # 构建输入上下文
            history_str = self.context.format_history()

            if retrieval_result:
                input_context = f"以下是检索结果：\n{retrieval_result}"
            else:
                input_context = "没有检索到相关信息，请表示对这个问题不了解或不清楚。"

            input_context += f"\n\n辩论主题：{topic}\n历史对话：\n{history_str}\n\n请按照上述要求完成你这一轮的回复："

            # 获取角色提示词
            character_prompt = self._get_mode_specific_prompt(query_result.mode)
            full_response = ""

            # 使用yield返回每个生成的句子
            for sentence in self._stream_chat_response(character_prompt, input_context):
                full_response += sentence
                yield sentence

            # 将完整回复加入历史
            self.context.add_dialogue_group(
                [DialogueEntry(speaker=self.character_name, content=full_response)]
            )

        except Exception as e:
            yield f"生成响应时出错: {str(e)}"
