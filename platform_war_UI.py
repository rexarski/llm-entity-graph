import curses
import threading
import time
from datetime import datetime
import os

PLATFORM_NAME = {"bilibili": "B站", "weibo": "微博", "zhihu": "知乎"}


class PlatformUI:
    def __init__(self):
        self.screen = None
        self.is_running = True
        self.lock = threading.Lock()
        self.topic = ""
        self.displayed_texts = []  # 存储所有已显示的文本
        self.scroll_offset = 0  # 滚动偏移量

        # 为Windows终端设置UTF-8编码
        if os.name == "nt":  # 仅在Windows系统上执行
            os.system("chcp 65001")

    def get_topic(self):
        """在启动curses之前获取主题"""
        print("\n请输入辩论主题: ", end="")
        self.topic = input().strip()
        return self.topic

    def start(self):
        """启动UI"""
        if not self.topic:
            return None

        try:
            # 初始化curses
            self.screen = curses.initscr()
            curses.start_color()
            curses.use_default_colors()

            # 定义颜色对
            curses.init_pair(1, curses.COLOR_BLUE, curses.COLOR_BLACK)  # bilibili蓝
            curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)  # 微博红
            curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)  # 知乎青
            curses.init_pair(4, curses.COLOR_GREEN, curses.COLOR_BLACK)  # 系统绿

            # 设置curses选项
            curses.noecho()
            curses.cbreak()
            self.screen.keypad(True)
            curses.curs_set(0)  # 隐藏光标

            # 获取窗口大小
            height, width = self.screen.getmaxyx()

            # 创建主窗口和状态栏
            self.main_win = curses.newwin(height - 2, width, 0, 0)
            self.status_win = curses.newwin(2, width, height - 2, 0)
            self.main_win.scrollok(True)

            # 启动UI线程
            self.ui_thread = threading.Thread(target=self._run_ui)
            self.ui_thread.daemon = True
            self.ui_thread.start()

            return self.topic

        except Exception as e:
            self.cleanup()
            raise e

    def _wrap_text(self, text: str, width: int) -> list:
        """将文本按指定宽度换行，考虑中文字符"""
        lines = []
        current_line = ""
        current_width = 0

        for char in text:
            # 更准确地计算字符宽度
            char_width = 2 if ord(char) > 127 else 1

            if current_width + char_width > width:
                lines.append(current_line)
                current_line = char
                current_width = char_width
            else:
                current_line += char
                current_width += char_width

        if current_line:
            lines.append(current_line)

        return lines

    def _calculate_display_lines(self):
        """计算所有要显示的行，包括换行后的文本"""
        display_lines = []
        height, width = self.main_win.getmaxyx()

        # 添加标题和分隔线
        current_time = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        time_text = f"System Time: {current_time}"
        title = f"[SYS] >>> {self.topic} <<<"
        display_lines.append({"type": "header", "content": (time_text, title)})
        display_lines.append({"type": "separator", "content": "=" * (width - 2)})

        # 处理每条消息
        for entry in self.displayed_texts:
            platform = entry["platform"]
            text = entry["text"]
            timestamp = entry["timestamp"]

            timestamp_str = f"[{timestamp}]> "
            platform_label = f"[{PLATFORM_NAME[platform]}]> "
            total_label_length = len(timestamp_str) + len(platform_label)

            # 计算可用宽度并换行文本
            available_width = width - total_label_length - 6
            wrapped_lines = self._wrap_text(text, available_width)

            # 添加消息的第一行（包含时间戳和平台标签）
            display_lines.append(
                {
                    "type": "message_start",
                    "content": (timestamp_str, platform_label, wrapped_lines[0]),
                    "platform": platform,
                }
            )

            # 添加消息的后续行
            for line in wrapped_lines[1:]:
                display_lines.append(
                    {
                        "type": "message_continuation",
                        "content": line,
                        "platform": platform,
                    }
                )

            # 添加消息间的空行
            display_lines.append({"type": "blank"})

        return display_lines

    def _update_display(self):
        """更新显示内容"""
        try:
            with self.lock:
                height, width = self.main_win.getmaxyx()
                self.main_win.clear()

                # 计算所有显示行
                display_lines = self._calculate_display_lines()

                # 计算可显示的行数（减去标题和分隔线）
                visible_height = height - 2

                # 确保滚动偏移量在有效范围内
                max_scroll = max(0, len(display_lines) - visible_height)
                self.scroll_offset = min(max_scroll, self.scroll_offset)
                self.scroll_offset = max(0, self.scroll_offset)

                # 显示可见的行
                current_y = 0
                for line in display_lines[
                    self.scroll_offset : self.scroll_offset + visible_height
                ]:
                    if current_y >= height:
                        break

                    try:
                        if line["type"] == "header":
                            time_text, title = line["content"]

                            # 先填充整行背景
                            padding = " " * (width - 2)
                            self.main_win.addstr(
                                current_y, 0, padding, curses.color_pair(4)
                            )

                            # 显示时间
                            self.main_win.addstr(
                                current_y,
                                2,
                                time_text,
                                curses.color_pair(4) | curses.A_BOLD,
                            )

                            # 计算标题位置并显示
                            time_length = len(time_text) + 4
                            available_width = width - time_length
                            title_pos = time_length + max(
                                0, (available_width - len(title)) // 2
                            )

                            if title_pos + len(title) > width - 2:
                                title = title[: width - title_pos - 5] + "..."
                            self.main_win.addstr(
                                current_y,
                                title_pos,
                                title,
                                curses.color_pair(4) | curses.A_BOLD,
                            )

                        elif line["type"] == "separator":
                            # 分隔线填充整行
                            separator = "=" * (width - 2)
                            self.main_win.addstr(
                                current_y, 1, separator, curses.color_pair(4)
                            )

                        elif line["type"] == "message_start":
                            timestamp_str, platform_label, text = line["content"]
                            platform = line["platform"]

                            # 设置平台颜色
                            color = (
                                curses.color_pair(1)
                                if platform == "bilibili"
                                else (
                                    curses.color_pair(2)
                                    if platform == "weibo"
                                    else curses.color_pair(3)
                                )
                            )

                            # 先填充整行背景
                            padding = " " * (width - 2)
                            self.main_win.addstr(current_y, 0, padding, color)

                            # 显示时间戳
                            self.main_win.addstr(
                                current_y,
                                2,
                                timestamp_str,
                                curses.color_pair(4) | curses.A_BOLD,
                            )

                            # 显示平台标签
                            self.main_win.addstr(
                                current_y,
                                2 + len(timestamp_str),
                                platform_label,
                                color | curses.A_BOLD,
                            )

                            # 显示文本内容
                            total_label_length = len(timestamp_str) + len(
                                platform_label
                            )
                            if text:  # 确保文本不为空
                                self.main_win.addstr(
                                    current_y,
                                    total_label_length + 3,
                                    text,
                                    color | curses.A_BOLD,
                                )

                        elif line["type"] == "message_continuation":
                            platform = line["platform"]
                            color = (
                                curses.color_pair(1)
                                if platform == "bilibili"
                                else (
                                    curses.color_pair(2)
                                    if platform == "weibo"
                                    else curses.color_pair(3)
                                )
                            )

                            # 先填充整行背景
                            padding = " " * (width - 2)
                            self.main_win.addstr(current_y, 0, padding, color)

                            # 显示续行内容
                            total_label_length = len("[HH:MM:SS.xxx]> ") + len(
                                "[平台]> "
                            )
                            if line["content"]:  # 确保文本不为空
                                self.main_win.addstr(
                                    current_y,
                                    total_label_length + 3,
                                    line["content"],
                                    color | curses.A_BOLD,
                                )

                        elif line["type"] == "blank":
                            # 空行处理
                            self.main_win.addstr(current_y, 0, " " * (width - 1))

                        current_y += 1

                    except curses.error:
                        # 忽略curses绘制错误，继续处理下一行
                        continue

                # 更新状态栏
                self.status_win.clear()
                status_text = "[SYS]> Press 'q' to exit | ESC to stop | ↑↓ to scroll"
                self.status_win.addstr(
                    0, 2, status_text, curses.color_pair(4) | curses.A_BOLD
                )

                # 刷新窗口
                self.main_win.refresh()
                self.status_win.refresh()

        except curses.error:
            pass

    def _run_ui(self):
        """运行UI主循环"""
        try:
            while self.is_running:
                self._update_display()

                self.screen.timeout(100)
                try:
                    key = self.screen.getch()
                    if key == ord("q") or key == 27:  # q 或 ESC
                        self.is_running = False
                    elif key == curses.KEY_UP:
                        self.scroll_offset = max(0, self.scroll_offset - 1)
                    elif key == curses.KEY_DOWN:
                        self.scroll_offset += 1  # _update_display会处理最大值
                except curses.error:
                    pass

                time.sleep(0.05)

        except Exception as e:
            self.stop()
            raise e

    def update_progress(self, character_name: str, text: str):
        """更新显示的文本"""
        with self.lock:
            # 将新的文本添加到显示列表中，包含时间戳
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-4]  # 精确到毫秒
            self.displayed_texts.append(
                {"platform": character_name, "text": text, "timestamp": timestamp}
            )

            # 自动滚动到底部
            display_lines = self._calculate_display_lines()
            height, _ = self.main_win.getmaxyx()
            self.scroll_offset = max(0, len(display_lines) - (height - 2))

    def cleanup(self):
        """清理curses设置"""
        if self.screen:
            self.screen.keypad(False)
            curses.nocbreak()
            curses.echo()
            curses.endwin()

    def stop(self):
        """停止UI显示"""
        self.is_running = False
        if hasattr(self, "ui_thread") and self.ui_thread.is_alive():
            self.ui_thread.join()
        self.cleanup()
