"""
每日Agent开发动态和知识科普生成脚本
使用LLM生成高质量的Agent开发动态和技术科普文章
"""

import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)
MODEL = os.getenv("MODEL_ID", "glm-5")

# 输出目录
DOCS_DIR = Path(__file__).parent / "docs"
DOCS_DIR.mkdir(exist_ok=True)


def generate_article(topic: str, article_type: str) -> str:
    """使用LLM生成文章"""
    
    if article_type == "agent_news":
        system_prompt = """你是一位AI Agent领域的技术专家和科技博主。
你的任务是撰写关于AI Agent开发最新动态的文章。

要求：
1. 内容要涵盖最新的Agent框架、工具、研究进展
2. 包含实际的技术洞察和开发建议
3. 语言专业但易懂，适合开发者阅读
4. 结构清晰，包含代码示例（如适用）
5. 文章长度约1500-2000字
6. 使用Markdown格式输出"""

        user_prompt = f"""请撰写一篇关于"{topic}"的文章，内容应包括：
- 背景介绍
- 最新进展和动态
- 技术细节分析
- 实践建议
- 未来展望

今天是{datetime.now().strftime("%Y年%m月%d日")}，请基于你的知识库生成最新、最有价值的内容。"""

    else:  # knowledge
        system_prompt = """你是一位资深的技术科普作者，擅长将复杂的技术概念讲得深入浅出。
你的任务是撰写高质量的AI/LLM/Agent技术科普文章。

要求：
1. 用通俗易懂的语言解释技术概念
2. 配合生动的比喻和实际案例
3. 循序渐进，由浅入深
4. 适当使用图表描述（Mermaid语法）
5. 文章长度约1500-2000字
6. 使用Markdown格式输出"""

        user_prompt = f"""请撰写一篇关于"{topic}"的技术科普文章，内容应包括：
- 什么是{topic}？（基础概念）
- 为什么{topic}很重要？
- {topic}是如何工作的？（原理讲解）
- 实际应用场景
- 学习资源推荐

请用生动有趣的方式来讲解，让初学者也能理解。"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=4000,
        temperature=0.7,
    )
    
    return response.choices[0].message.content


def save_article(content: str, title: str, article_type: str) -> str:
    """保存文章到文件"""
    today = datetime.now().strftime("%Y-%m-%d")
    
    # 生成文件名
    safe_title = title.replace(" ", "_").replace("/", "_")[:30]
    filename = f"{today}_{article_type}_{safe_title}.md"
    filepath = DOCS_DIR / filename
    
    # 添加元信息
    full_content = f"""---
title: {title}
date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
type: {article_type}
model: {MODEL}
generated_by: daily_news.py
---

# {title}

> 📅 生成日期：{datetime.now().strftime("%Y年%m月%d日")}
> 🤖 生成模型：{MODEL}

---

{content}
"""
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(full_content)
    
    return str(filepath)


def main():
    """主函数"""
    print("=" * 60)
    print("🤖 每日Agent动态 & 知识科普生成器")
    print(f"📅 日期: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"🔧 模型: {MODEL}")
    print("=" * 60)
    
    # 定义要生成的主题
    topics = [
        {
            "type": "agent_news",
            "title": "AI Agent开发最新动态",
            "topic": "2024-2025年AI Agent开发领域的最新进展，包括多Agent系统、工具使用、记忆机制等"
        },
        {
            "type": "knowledge", 
            "title": "深入理解Function Calling",
            "topic": "大语言模型的Function Calling（函数调用）机制"
        },
        {
            "type": "knowledge",
            "title": "RAG技术详解",
            "topic": "检索增强生成（RAG）技术原理与实践"
        },
        {
            "type": "agent_news",
            "title": "Agent框架对比分析",
            "topic": "主流Agent框架（LangChain、AutoGPT、CrewAI等）的对比与选型建议"
        },
    ]
    
    generated_files = []
    
    for i, item in enumerate(topics, 1):
        print(f"\n[{i}/{len(topics)}] 正在生成: {item['title']}...")
        
        try:
            content = generate_article(item["topic"], item["type"])
            filepath = save_article(content, item["title"], item["type"])
            generated_files.append(filepath)
            print(f"  ✅ 已保存: {filepath}")
        except Exception as e:
            print(f"  ❌ 生成失败: {e}")
    
    # 生成索引文件
    index_path = generate_index(generated_files)
    
    print("\n" + "=" * 60)
    print(f"🎉 完成！共生成 {len(generated_files)} 篇文章")
    print(f"📁 输出目录: {DOCS_DIR}")
    print(f"📋 索引文件: {index_path}")
    print("=" * 60)


def generate_index(files: list) -> str:
    """生成文章索引"""
    index_path = DOCS_DIR / "README.md"
    
    content = f"""# 📚 Agent开发知识库

> 自动生成于 {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}

## 📖 文章列表

| 日期 | 类型 | 标题 |
|------|------|------|
"""
    
    # 按日期倒序排列
    for filepath in sorted(files, reverse=True):
        filename = Path(filepath).name
        parts = filename.split("_")
        date = parts[0] if parts else "unknown"
        article_type = parts[1] if len(parts) > 1 else "unknown"
        title = "_".join(parts[2:]).replace(".md", "") if len(parts) > 2 else filename
        
        type_emoji = "📰" if article_type == "agent_news" else "💡"
        content += f"| {date} | {type_emoji} {article_type} | [{title}]({filename}) |\n"
    
    content += f"""
## 📂 目录结构

```
docs/
├── README.md          # 本索引文件
├── *_agent_news_*.md  # Agent开发动态
└── *_knowledge_*.md   # 技术科普文章
```

## 🔧 使用方法

运行以下命令生成最新文章：

```bash
python daily_news.py
```

## 📝 说明

- 所有文章由 **{MODEL}** 模型自动生成
- 文章类型：
  - `agent_news`: AI Agent开发最新动态
  - `knowledge`: 高质量技术科普
- 建议每天运行一次获取最新内容
"""
    
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    return str(index_path)


if __name__ == "__main__":
    main()