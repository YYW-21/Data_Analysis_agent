import json

from openai import OpenAI

from datapilot.core.config import settings


def generate_report(context: dict) -> str:
    base_report = deterministic_report(context)
    if not settings.enable_llm_report or not settings.openai_api_key:
        return base_report

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    prompt = (
        "You are a data science assistant. Rewrite the following structured analysis into a "
        "clear Markdown report in Chinese. Do not invent metrics or columns. Keep all numbers "
        "consistent with the provided JSON.\n\n"
        f"Structured context:\n{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
        f"Fallback report:\n{base_report}"
    )
    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = response.choices[0].message.content
        return content or base_report
    except Exception as exc:
        return f"{base_report}\n\n## LLM Report Note\n\nLLM enhancement failed: `{exc}`\n"


def deterministic_report(context: dict) -> str:
    profile = context["profile"]
    task = context["task"]
    ml = context["ml"]
    lines = [
        "# DataPilot 自动分析报告",
        "",
        "## 1. 数据集概览",
        "",
        f"- 行数: {profile['rows']}",
        f"- 列数: {profile['columns']}",
        f"- 重复行: {profile['duplicate_rows']}",
        f"- 目标列: `{task['target_column']}`",
        f"- 任务类型: `{task['task_type']}`",
        f"- 任务理解来源: `{task.get('source', 'rules')}`",
        f"- 任务理解置信度: {task.get('confidence', 0):.2f}",
        "",
        "## 2. Agent 工作流计划",
        "",
    ]
    for step in task.get("workflow_steps", []):
        lines.append(f"- {step}")
    if task.get("reason"):
        lines.append(f"- Reason: {task['reason']}")

    lines.extend(
        [
            "",
            "## 3. 数据质量",
        "",
        ]
    )
    if profile["warnings"]:
        lines.extend([f"- {warning}" for warning in profile["warnings"]])
    else:
        lines.append("- 未发现明显的高风险数据质量问题。")

    lines.extend(
        [
            "",
            "## 4. 特征工程",
            "",
            f"- 数值特征: {', '.join(ml['features']['numeric']) or '无'}",
            f"- 类别特征: {', '.join(ml['features']['categorical']) or '无'}",
            "",
            "## 5. 模型训练结果",
            "",
            f"- 最佳模型: `{ml['best_model']}`",
            f"- 模型文件: `{ml['model_path']}`",
            "",
            "### 候选模型指标",
            "",
        ]
    )
    for item in ml["candidate_metrics"]:
        metrics = ", ".join(f"{k}={v:.4f}" for k, v in item["metrics"].items())
        lines.append(f"- `{item['model']}`: {metrics}")

    lines.extend(
        [
            "",
            "## 6. 结论",
            "",
            "本次分析完成了数据画像、EDA、确定性预处理、模型训练和指标评估。"
            "建议结合业务背景进一步检查目标列定义、样本偏差和重要特征的可解释性。",
        ]
    )
    return "\n".join(lines) + "\n"
