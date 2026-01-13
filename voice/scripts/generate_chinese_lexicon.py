#!/usr/bin/env python3
"""
Generate Chinese phoneme lexicon from misaki - Kokoro's native G2P.

IMPORTANT: misaki IS Kokoro's native G2P, so we use its exact output format.
This replaces the hand-crafted pypinyin->IPA conversion with accurate phonemes.

Key insight from manager directive (commit 328830c):
- misaki outputs phonemes that Kokoro was trained on
- espeak-ng has BROKEN tone data (你好 outputs both rising - WRONG)
- misaki outputs CORRECT tones (你好 -> ni↓xau↓ - both dipping)

Usage:
    python scripts/generate_chinese_lexicon.py > stream-tts-cpp/include/chinese_misaki_lexicon.hpp
    python scripts/generate_chinese_lexicon.py --test  # Test a few words

Copyright 2025 Andrew Yates. All rights reserved.
"""

import sys
import argparse
try:
    from misaki.zh import ZHG2P
except ImportError:
    print("Error: Install misaki with: pip install misaki jieba cn2an", file=sys.stderr)
    sys.exit(1)


# Common Chinese words for the lexicon
# These are words likely to appear in Claude Code voice output
COMMON_WORDS = [
    # Greetings and common phrases
    "你好", "再见", "谢谢", "不客气", "对不起", "没关系",
    "早上好", "晚上好", "下午好", "晚安",
    "欢迎", "请", "好的", "是的", "不是",

    # Pronouns and demonstratives
    "我", "你", "他", "她", "它", "我们", "你们", "他们",
    "这", "那", "这个", "那个", "这里", "那里", "哪里",
    "什么", "谁", "怎么", "为什么", "多少", "几",
    "一个", "两个", "三个", "几个",  # ADDED: counters (jieba splits "一个" as word)
    "中文", "英文", "日文",  # ADDED: languages

    # Numbers
    "零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
    "百", "千", "万", "亿",
    "第一", "第二", "第三",

    # Time
    "今天", "明天", "昨天", "后天", "前天",
    "现在", "以前", "以后", "刚才", "马上",
    "早上", "中午", "下午", "晚上", "半夜",
    "年", "月", "日", "号", "星期", "周",
    "小时", "分钟", "秒",

    # Programming and technology terms
    "代码", "程序", "函数", "变量", "常量",
    "程序员", "工程师", "开发者",  # ADDED: job titles
    "文件", "文件夹", "目录", "路径",
    "数据", "数据库", "服务器", "客户端",
    "网络", "网站", "网页", "链接",
    "输入", "输出", "读取", "写入", "保存",
    "编译", "运行", "执行", "调试", "测试",
    "错误", "警告", "成功", "失败",
    "开始", "结束", "暂停", "继续", "取消",
    "创建", "删除", "修改", "修复", "更新", "查询",  # ADDED: 修复
    "安装", "卸载", "下载", "上传",
    "登录", "登出", "注册", "密码", "用户",

    # Actions
    "做", "看", "说", "听", "想", "知道", "认为",
    "去", "来", "走", "跑", "站", "坐",
    "吃", "喝", "睡", "醒",
    "给", "拿", "放", "找", "用", "要", "能", "会",
    "等", "帮", "帮助", "教", "学", "学习", "工作", "休息",  # ADDED: 帮助, 学习

    # Adjectives
    "好", "坏", "大", "小", "多", "少", "长", "短",
    "高", "低", "快", "慢", "新", "旧", "老", "年轻",
    "热", "冷", "难", "容易", "重要", "简单", "复杂",
    "正确", "错误", "完成", "完整",

    # Common nouns
    "人", "事", "物", "地方", "时间", "问题", "答案",
    "方法", "结果", "原因", "目的", "意思", "例子",
    "情况", "状态", "条件", "步骤", "过程",
    "名字", "内容", "格式", "类型", "版本",

    # Connectors and particles
    "的", "地", "得", "了", "着", "过",
    "和", "与", "或", "但", "但是", "因为", "所以",
    "如果", "那么", "虽然", "不过", "然后",
    "还", "也", "都", "就", "才", "又", "再",

    # Locations
    "中国", "北京", "上海", "深圳", "广州",
    "东", "西", "南", "北", "中",
    "上", "下", "左", "右", "前", "后", "里", "外",

    # Status words for coding context
    "正在", "已经", "将要", "需要", "可以", "应该",
    "必须", "可能", "一定", "确定", "不确定",
    "处理中", "等待中", "完成", "进行中",

    # Common phrases and question words
    "好的", "没问题", "可以", "不可以",
    "请稍等", "马上", "立刻", "等一下",
    "非常好", "太好了", "不错", "很好",
    "请问", "能不能", "可不可以",  # ADDED: question phrases

    # Common sentences (include jieba word boundaries via misaki)
    # These get proper spaces from jieba segmentation
    "今天天气很好", "今天天气", "天气很好",
    "你好吗", "我很好", "你叫什么名字", "谢谢你",
    "我不知道", "我知道了", "没有问题",
    "请等一下", "请帮帮我", "请告诉我",
    "我正在工作", "我正在学习", "我正在等待",
    "代码运行成功", "代码运行失败", "代码编译完成",
    "文件已保存", "文件已删除", "文件已创建",
    "程序正在运行", "程序已停止", "程序已完成",
    "请稍等一下", "马上就好", "立刻处理",
    # Test sentences - critical for quality tests
    "谢谢你的帮助", "我正在学习中文", "请问这个多少钱",
    "这是什么意思", "你能帮我吗", "我听不懂",
    "请再说一遍", "慢一点说", "我明白了",

    # World/Nature
    "世界", "天", "地", "山", "水", "河", "海",
    "风", "雨", "雪", "云", "太阳", "月亮",
    "花", "草", "树", "木",
    # Parks and outdoor activities (2025-12-08 Worker #419)
    "公园", "花园", "果园", "动物园", "植物园", "游乐园",
    "散步", "散会", "运动", "跑步", "健身",
    "广场", "操场", "球场", "停车场", "商场", "市场",

    # Programming/Development expanded (2025-12-08)
    # These words were found missing during quality testing
    "代码审查", "审查", "审核", "评审", "检查", "验证",
    "抛出异常", "异常", "抛出", "捕获", "捕获错误",
    "逻辑运算", "逻辑", "运算", "算法",
    "内存泄漏", "泄漏", "内存", "溢出",
    "线程", "进程", "协程",
    "推送", "推送远程", "远程", "拉取", "拉取更新",
    "仓库", "分支", "合并", "冲突", "解决冲突",
    "索引", "引擎", "框架", "模块", "组件", "插件",
    "栈", "堆", "队列", "链表", "哈希表", "字典",
    "递归", "迭代", "循环", "遍历",
    "参数", "返回值", "返回", "声明", "定义", "赋值",
    "注释", "文档", "说明",
    "兼容", "适配", "迁移", "部署", "发布", "上线",
    "监控", "告警", "日志", "跟踪", "追踪", "分析", "统计",
    "权限", "授权", "认证", "身份", "令牌",
    "缓存", "刷新", "清除", "释放", "分配",
    "并发", "异步", "同步", "阻塞", "唤醒",
    "加密", "解密", "签名", "校验", "哈希",
    "配置文件", "配置", "设置", "选项",
]

# Additional characters for fallback (single-char lookups)
# Include ALL common Chinese characters - approx 3500 most frequent
# This is the HSK 1-6 character set plus common additional characters
# Without this, ANY unknown character causes fallback to espeak (WRONG tones!)
COMMON_CHINESE_CHARS = """
一二三四五六七八九十百千万亿零
是不的了有在我你他她它们这那哪什么谁怎么为什么
多少几个些种样
去来走跑站坐飞开关打找看见说听想知道觉得认为
给拿放用要能会可以应该必须需要
吃喝睡醒穿住买卖借还做工作学习教读写画听说问答
好坏大小多少长短高低快慢新旧老年轻热冷难容易重轻
美丑胖瘦对错真假
红黄蓝绿白黑
今天明天昨天后天前天现在以前以后马上立刻刚才然后
早上中午下午晚上半夜
年月日号星期周
小时分钟秒
东西南北中上下左右前后里外内中间旁边附近
人物事情地方时间问题答案
方法结果原因目的意思例子情况状态条件步骤过程
名字内容格式类型版本
家房间屋子门窗户床桌椅子
学校公司医院银行商店饭店酒店机场车站
街道路口红绿灯
车船飞机火车地铁公交出租车
手机电脑电视电话网络网站邮件
书本报纸杂志
衣服裤子鞋袜帽子
水茶咖啡牛奶果汁
饭菜面包蛋肉鱼鸡猪牛羊
苹果香蕉橙子葡萄西瓜
山水河海湖泉
天地云雨雪风雷电
太阳月亮星星
花草树木叶
鸟鱼虫猫狗
春夏秋冬
父母儿女兄弟姐妹爷奶叔婶
朋友同学老师学生医生护士
工人农民商人
国王总统部长经理主任
中国美国英国法国德国日本韩国
北京上海广州深圳香港台湾
省市县区镇村
政府军队警察法院
教育经济政治文化历史科学
数学语文英语物理化学生物
音乐美术体育
电影电视节目新闻广告
比赛竞争考试成绩
钱元块毛分
斤两克千克公斤
米厘米毫米公里英里
度摄氏华氏
和与或但是因为所以如果那么虽然不过然后
的地得了着过
把被比从到对给跟向往在朝
吧吗呢啊哦嗯
爱恨怕怪希望相信担心害怕生气高兴
注意小心记住忘记理解明白
准备开始结束继续停止等待
欢迎感谢道歉祝贺
帮助支持反对同意拒绝
成功失败赢输
正确错误对不对行不行好不好
接受送给打开关闭进入离开
安静热闹干净脏乱整齐
简单复杂重要特别普通
漂亮美丽帅
聪明笨傻
勤快懒惰
便宜贵
远近深浅
满空
先后
常很太非更最
只都也又再还就才已
可能一定肯定确实
突然忽然终于居然
其实实际真正
大概可能也许
当然自然
经常有时偶尔
总是永远从来
一起单独
首先然后最后
一般特别尤其
比如例如
另外此外
总之总而言之
因此所以
不然否则

# Programming/Development terms - expanded (2025-12-08)
# These characters were found missing during quality testing
审查阅批判评估览浏检验证
抛捕获取截拦顺挡挂推拉压挤缩
逻辑算输键盘触摸滚轮轴柱
泄漏溢缺陷故障损坏崩溃宕荡卡冻
仓库索引引擎框架模块组件插件
栈堆队列表单字典映射排序
递归迭代循环遍增减除乘
注释标签属性参数返回声明定义赋值
调换兼容适配迁移部署发布
监控告警日志跟踪追踪分析统计
权限授权认证鉴别身份令牌票据
缓存刷新清除释放分配管理优
并发异步同步阻塞挂起唤醒
加密解密签名校验哈希摘要编
# More missing characters found in testing (2025-12-08 round 2)
全部完整启动停止暂境况态势响
析使令载初始终默认自默准备转换
端口地址接命请求响应客户服务端
克隆提交拉取合并冲突解决恢复撤销重置
构建打包压缩解压缩安卸载升级降级
线程池连接超时重试回滚事务提
# Missing characters from comprehensive test (2025-12-08 round 3)
临时依赖导出入操作效率敏感添加访问率软件收发
# Missing AI/ML/Tech characters (2025-12-08 round 4)
研究机器学习深度神经网络人工智能模型训练推理预测
识别分类聚类回归优化损失梯度反向传播
卷积池化激活函数层输入输出隐藏嵌入向量矩阵张量
特征提取选择降维正则化过拟合欠拟合泛化
批量随机梯度下降学习率权重偏置
注意力机制变换器编码解码
生成对抗判别器
强化奖励策略动作状态
# Additional missing characters (2025-12-08 round 5)
资源计算能力容量存储
蓝牙无线网络云端服务
算法效率复杂度时间空间
程序设计架构设计模式
接口实现继承多态封装抽象
虚拟环境变量常量
# Missing characters causing fallback (2025-12-08 round 6 - Worker #416)
# From test sentence: "这个函数有一个严重的bug需要立即修复"
严仔即圾垃尽段洞瓶细致锁颈
# Additional common programming/debugging characters
漏洞陷阱崩溃宕机闪退卡顿
调试断点跟踪堆栈追踪调用
仔细检查详细信息警告提示
立即马上尽快紧急严重致命
性能瓶颈优化压测负载吞吐
垃圾回收内存泄露释放分配
容器镜像集群节点副本实例
完美完善完整齐全健全稳定
临界互斥原子锁竞态死锁活锁
# Missing common characters (2025-12-08 Worker #417)
# Found from testing: 够 流 是常用字
够流足够能够够了够用

交流河流电流气流人流客流物流工作流水流血流
# More common words that should be in lexicon
高兴兴奋兴趣感兴趣有兴趣没兴趣
喜欢愿意乐意倾向偏好
# Common conversation words
聊天说话讲话对话沟通交流交谈会话
# Common verbs (that may be missing)
见面遇到碰到遇见
# Ability/capability words
足够够用充足充分完备齐备具备配备
# Flow-related words
流动流程工作流程流水线流畅流利流行
# More conversation particles
呵呵哈哈嘻嘻嘿嘿呀啦
# Common auxiliary verbs
能够可能可以能不能会不会
# Time expressions
刚刚才刚才才刚刚刚好
# Politeness and honorifics
您贵姓啥咋呢嘛吖嗯哟哎
# More common verbs
喊叫嚷骂吵闹响震惊吓怕担忧
# Communication verbs
通知告知通告宣布发表声明表示表达表明
# Emotions
幸福快乐开心难过伤心痛苦失望满意
# Common adverbs
非常十分特别格外尤其极其异常相当稍微略微
# Numbers and quantities
几十几百几千若干各种各样各位
# Location/direction words
这儿那儿哪儿这边那边旁边对面背后底下
# Technology words that might be missing
虚拟现实增强混合云计算边缘计算
# Missing characters causing fallback (2025-12-08 Worker #419)
# Found from testing: 园 散 - common location/activity words
园公园花园果园花园里动物园植物园儿童乐园游乐园
散散步散会分散扩散散热播散散开
# More location/activity words likely to appear
逛街购物旅游游玩玩耍娱乐运动健身跑步走路
广场操场球场停车场机场商场市场市集农场牧场战场
# More common conversation words
稀有趣搞笑可笑滑稽尴尬
请求邀请邀请函请客做客客人宾客嘉宾
# Additional words found in test sentences
讨论议论争论谈论评论讨价还价
仓促匆忙忙碌繁忙轻松悠闲空闲
# More programming-related words
远程本地局部全局静态动态私有公开公共受保护
"""

# Combine with characters from COMMON_WORDS
SINGLE_CHARS = list(set(''.join(COMMON_WORDS) + COMMON_CHINESE_CHARS.replace('\n', '')))

# English words commonly embedded in Chinese tech text (Worker #418)
# These need IPA phonemes so Kokoro can pronounce them when embedded in Chinese
ENGLISH_TECH_WORDS = [
    # Common programming terms
    "bug", "debug", "fix", "code", "app", "API", "SDK", "UI", "UX",
    "CPU", "GPU", "RAM", "ROM", "SSD", "HDD", "USB", "HDMI",
    "HTTP", "HTTPS", "URL", "DNS", "IP", "TCP", "UDP", "FTP", "SSH",
    "HTML", "CSS", "JSON", "XML", "YAML", "SQL",
    "JavaScript", "Python", "Java", "Ruby", "Go", "Rust", "Swift",
    "React", "Vue", "Angular", "Node", "Django", "Flask",
    "Docker", "Kubernetes", "AWS", "Azure", "GCP",
    "Git", "GitHub", "GitLab",
    "IDE", "VSCode", "Vim", "Emacs",
    "Linux", "Windows", "macOS", "iOS", "Android",
    "README", "TODO", "FIXME",
    "OK", "yes", "no", "test", "demo", "beta", "alpha",
    "login", "logout", "email", "password",
    "server", "client", "database", "cache", "queue",
    "push", "pull", "commit", "merge", "branch", "fork",
    "null", "undefined", "true", "false",
    "error", "warning", "info", "success", "fail",
    "start", "stop", "pause", "resume", "restart",
    "load", "save", "delete", "update", "create",
    "input", "output", "config", "setting", "option",
    "async", "sync", "callback", "promise", "await",
    "class", "function", "method", "variable", "constant",
    "array", "list", "map", "set", "object", "string", "number",
    "if", "else", "for", "while", "switch", "case", "return",
    "try", "catch", "throw", "exception",
    "public", "private", "static", "final",
    "import", "export", "module", "package",
    "interface", "extends", "implements",
    "thread", "process", "lock", "mutex",
    "memory", "heap", "stack", "pointer",
    "compile", "build", "run", "execute", "deploy",
    "framework", "library", "plugin", "extension",
    "model", "view", "controller", "service",
    "token", "session", "cookie", "header", "body",
    "request", "response", "status", "endpoint",
    "encrypt", "decrypt", "hash", "sign", "verify",
    "AI", "ML", "NLP", "CNN", "RNN", "LSTM", "GAN", "GPT", "LLM",
    "training", "inference", "predict",
    "regex", "pattern", "match", "replace", "search", "find",
    "print", "log", "debug", "trace", "profile",
    "mock", "stub", "test", "fixture",
    "unit", "integration", "smoke", "regression",
    "CI", "CD", "DevOps", "QA",
]


def generate_lexicon():
    """Generate the C++ header file content."""

    g2p = ZHG2P()

    # Skip English G2P import - spacy is broken with Python 3.14
    # English tech words are handled by the main lexicon.hpp file
    has_en_g2p = False
    print("// Skipping English G2P (spacy broken with Python 3.14)", file=sys.stderr)

    # Collect all entries
    entries = {}

    # Process English tech words first (using English G2P)
    if has_en_g2p:
        for word in ENGLISH_TECH_WORDS:
            try:
                result = en_g2p(word)
                # g2p returns (phonemes, None) tuple or just phonemes
                phonemes = result[0] if isinstance(result, tuple) else result
                if phonemes:
                    entries[word] = phonemes
                    # Also add lowercase version
                    lower = word.lower()
                    if lower != word and lower not in entries:
                        entries[lower] = phonemes
            except Exception as e:
                print(f"// Warning: Failed to process English '{word}': {e}", file=sys.stderr)

    # Process compound words first (higher priority)
    for word in COMMON_WORDS:
        try:
            result = g2p(word)
            # g2p returns (phonemes, None) tuple
            phonemes = result[0] if isinstance(result, tuple) else result
            if phonemes:
                entries[word] = phonemes
        except Exception as e:
            print(f"// Warning: Failed to process '{word}': {e}", file=sys.stderr)

    # Process single characters as fallback
    for char in SINGLE_CHARS:
        if char not in entries:
            try:
                result = g2p(char)
                # g2p returns (phonemes, None) tuple
                phonemes = result[0] if isinstance(result, tuple) else result
                if phonemes:
                    entries[char] = phonemes
            except Exception:
                pass

    # Sort entries: longer words first, then alphabetically
    sorted_entries = sorted(entries.items(), key=lambda x: (-len(x[0]), x[0]))

    # Generate header
    print("""#pragma once
// Chinese phoneme lexicon - AUTO-GENERATED from misaki (Kokoro's native G2P)
// Generated by scripts/generate_chinese_lexicon.py using misaki.zh.ZHG2P
//
// IMPORTANT: misaki IS Kokoro's native G2P, so these phonemes are exactly
// what the TTS model expects. This replaces hand-crafted pinyin->IPA conversion.
//
// Comparison:
//   espeak-ng (BROKEN):     你好 -> ni↗hˈɑu↗ (both rising - WRONG)
//   misaki (underlying):    你好 -> ni↓xau↓ (both dipping - TRAINING DATA)
//
// JIEBA INTEGRATION (2025-12-08):
// Python Kokoro uses jieba for word segmentation to get natural prosody.
// C++ now uses cppjieba for runtime segmentation to match Python behavior.
// This adds spaces at word boundaries for proper prosodic grouping.
//
// Copyright 2025 Andrew Yates. All rights reserved.

#include <string>
#include <unordered_map>
#include <sstream>
#include <vector>
#include "chinese_segmenter.hpp"

namespace chinese_misaki {

// Phoneme lexicon: Chinese text -> misaki phonemes with tones
// Tone markers: ↗ (rising/tone 2), ↓ (dipping/tone 3), ↘ (falling/tone 4), → (level/tone 1)
inline const std::unordered_map<std::string, std::string> PHONEME_LEXICON = {""")

    for word, phonemes in sorted_entries:
        # Skip non-Chinese entries that would break C++ syntax
        # (punctuation, ASCII digits/letters that aren't useful)
        if len(word) == 1 and ord(word[0]) < 0x4E00:
            # Skip ASCII and non-CJK single chars (punctuation, digits, etc.)
            # These would need complex escaping and aren't needed for Chinese G2P
            continue
        # Escape any special characters in both word and phonemes
        word_escaped = word.replace('\\', '\\\\').replace('"', '\\"')
        phonemes_escaped = phonemes.replace('\\', '\\\\').replace('"', '\\"')
        print(f'    {{"{word_escaped}", "{phonemes_escaped}"}},')

    print("""};

// UTF-8 character length helper
inline int utf8_char_length(unsigned char c) {
    if ((c & 0x80) == 0) return 1;       // ASCII
    if ((c & 0xE0) == 0xC0) return 2;    // 2-byte
    if ((c & 0xF0) == 0xE0) return 3;    // 3-byte
    if ((c & 0xF8) == 0xF0) return 4;    // 4-byte
    return 1;  // Invalid, treat as 1
}

// Check if a character is a CJK Unified Ideograph (Chinese character)
inline bool is_chinese_char(const std::string& ch) {
    if (ch.size() != 3) return false;
    unsigned char c1 = static_cast<unsigned char>(ch[0]);
    unsigned char c2 = static_cast<unsigned char>(ch[1]);
    // CJK Unified Ideographs: U+4E00-U+9FFF
    if (c1 == 0xE4 && c2 >= 0xB8) return true;
    if (c1 >= 0xE5 && c1 <= 0xE8) return true;
    if (c1 == 0xE9 && c2 <= 0xBF) return true;
    return false;
}

// Look up a single word/character in the lexicon
// Returns empty string if not found
inline std::string lookup_word(const std::string& word) {
    auto it = PHONEME_LEXICON.find(word);
    if (it != PHONEME_LEXICON.end()) {
        return it->second;
    }
    return "";
}

// Look up Chinese text and return misaki phonemes
// NEW: Uses jieba segmentation for natural word boundaries (matches Python Kokoro behavior)
// Returns empty string if no match found (signals espeak fallback)
inline std::string lookup_phonemes(const std::string& text) {
    // Try the full text first (for common phrases in lexicon)
    auto it = PHONEME_LEXICON.find(text);
    if (it != PHONEME_LEXICON.end()) {
        return it->second;
    }

    // Use jieba to segment the text into words (matches Python behavior)
    // This is the key to natural prosody - word boundaries determine prosodic groups
    std::vector<std::string> words = chinese_segmenter::segment(text);

    std::stringstream result;
    bool found_any = false;
    bool has_unknown_chinese = false;

    for (const auto& word : words) {
        // Skip empty words
        if (word.empty()) continue;

        // Skip punctuation
        if (word == " " || word == "," || word == "." || word == "!" || word == "?" ||
            word == "，" || word == "。" || word == "！" || word == "？" || word == "、" ||
            word == "\\n" || word == "\\t") {
            continue;
        }

        // Try to look up the word
        std::string phonemes = lookup_word(word);

        if (!phonemes.empty()) {
            // Found in lexicon
            if (found_any) {
                result << " ";  // Space between word groups for prosodic boundary
            }
            result << phonemes;
            found_any = true;
        } else {
            // Word not in lexicon - try character-by-character fallback
            bool word_ok = true;
            std::stringstream word_phonemes;
            bool first_char = true;

            size_t i = 0;
            while (i < word.size()) {
                unsigned char c = static_cast<unsigned char>(word[i]);
                int char_len = utf8_char_length(c);
                if (i + char_len > word.size()) break;

                std::string ch = word.substr(i, char_len);
                std::string char_phoneme = lookup_word(ch);

                if (!char_phoneme.empty()) {
                    // Character found in lexicon
                    if (!first_char) word_phonemes << "";  // No space within word
                    word_phonemes << char_phoneme;
                    first_char = false;
                } else if (is_chinese_char(ch)) {
                    // Unknown Chinese character - mark for fallback
                    has_unknown_chinese = true;
                    word_ok = false;
                    break;
                } else {
                    // Non-Chinese character (ASCII, etc) - pass through
                    if (!first_char) word_phonemes << "";
                    word_phonemes << ch;
                    first_char = false;
                }

                i += char_len;
            }

            if (word_ok && !first_char) {
                if (found_any) {
                    result << " ";  // Space between word groups
                }
                result << word_phonemes.str();
                found_any = true;
            }
        }
    }

    // If we encountered unknown Chinese characters, return empty to trigger fallback
    if (has_unknown_chinese) {
        return "";
    }

    return found_any ? result.str() : "";
}

// Check if text contains any Chinese characters
inline bool contains_chinese(const std::string& text) {
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        int len = utf8_char_length(c);
        if (i + len > text.size()) break;

        std::string ch = text.substr(i, len);
        if (is_chinese_char(ch)) {
            return true;
        }
        i += len;
    }
    return false;
}

}  // namespace chinese_misaki
""")


def test_lexicon():
    """Test the misaki G2P outputs on a few examples (no sandhi)."""
    g2p = ZHG2P()

    test_cases = [
        ("你好", "tone 3+3 stays as training data (no sandhi)"),
        ("好好", "tone 3+3 stays as training data (no sandhi)"),
        ("可以", "tone 3+3 stays as training data (no sandhi)"),
        ("好", "Single tone 3 - no change"),
        ("世界", "Tone 4+4 - no change"),
        ("不好意思", "Underlying tones only"),
        ("谢谢", "Thank you"),
        ("对不起", "Sorry"),
        ("代码", "Code"),
        ("中国", "China"),
    ]

    print("Testing misaki Chinese G2P (no sandhi applied):")
    print("=" * 70)
    for text, description in test_cases:
        raw_phonemes = g2p(text)
        print(f"  '{text}'")
        print(f"    Raw (misaki):   {raw_phonemes}")
        print(f"    Note: {description}")
        print()
    print("=" * 70)

    # Key comparison
    print("\nKey fix for 你好 (tone 3+3 kept as training data):")
    raw = g2p('你好')
    print(f"  espeak-ng (BROKEN):     ni↗hˈɑu↗ (both rising - WRONG)")
    print(f"  misaki (underlying):    {raw} (both dipping - TRAINING DATA)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Chinese misaki lexicon')
    parser.add_argument('--test', action='store_true', help='Test G2P on examples')
    args = parser.parse_args()

    if args.test:
        test_lexicon()
    else:
        generate_lexicon()
