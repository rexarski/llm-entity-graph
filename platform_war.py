from typing import List
from chat import Chat, DialogueEntry
from platform_war_UI import PlatformUI

PLATFORM_NAME = {"bilibili": "B站", "weibo": "微博", "zhihu": "知乎"}


class PlatformWar:
    def __init__(self):
        self.display = PlatformUI()
        self.characters = {}
        self._initialize_characters()
        self.dialogue_history: List[DialogueEntry] = []
        self.is_running = True
        self.debate_topic = ""

    def _initialize_characters(self):
        """初始化各平台角色"""
        for platform in PLATFORM_NAME.keys():
            self.characters[platform] = Chat(character_name=platform)

        print(f"\n已初始化 {len(self.characters)} 个平台角色")

    def _process_character_response(self, character: Chat) -> None:
        """处理单个角色的响应，使用生成器获取并显示响应"""
        display_name = PLATFORM_NAME[character.character_name]
        full_response = ""

        # 使用生成器逐句获取响应
        for sentence in character.generate_response(
            dialogue_group=self.dialogue_history.copy(),  # 传入副本避免修改
            topic=self.debate_topic,
        ):
            # 更新UI显示当前句子
            self.display.update_progress(character.character_name, sentence)
            full_response += sentence + " "

        # 将完整响应添加到对话历史
        if full_response:
            self.dialogue_history.append(
                DialogueEntry(speaker=display_name, content=full_response.strip())
            )

    def start(self):
        """启动平台大战"""
        try:
            # 在启动curses之前获取主题
            self.debate_topic = self.display.get_topic()
            if not self.debate_topic:
                print("辩论主题不能为空！")
                return

            # 启动UI
            self.display.start()

            current_idx = 0
            platforms = list(PLATFORM_NAME.keys())

            while self.is_running and self.display.is_running:
                current_platform = platforms[current_idx]
                current_character = self.characters[current_platform]

                # 处理当前角色的响应
                self._process_character_response(current_character)

                # 更新下一个发言的角色索引
                current_idx = (current_idx + 1) % len(platforms)

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"运行时发生错误: {str(e)}")
        finally:
            self.stop()

    def stop(self):
        """停止平台大战"""
        self.is_running = False
        if self.display:
            self.display.stop()


def main():
    debate = PlatformWar()
    try:
        debate.start()
    except Exception as e:
        print(f"程序发生错误: {str(e)}")
    finally:
        debate.stop()


if __name__ == "__main__":
    main()
