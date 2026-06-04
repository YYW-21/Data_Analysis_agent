const i18n = {
  zh: {
    subtitle: "可控的表格数据自动分析与建模工作流",
    navUpload: "数据集",
    navConfigure: "任务配置",
    navResults: "分析结果",
    navDocs: "API 文档",
    languageLabel: "语言",
    eyebrow: "Agent + AutoML 工作台",
    heroTitle: "从表格数据到可解释模型报告",
    heroCopy:
      "上传数据，选择目标列，DataPilot 会自动完成画像、EDA、建模、评估、特征重要性和报告生成。",
    systemReady: "本地服务已就绪",
    flowUpload: "上传数据",
    flowProfile: "数据画像",
    flowTrain: "模型训练",
    flowReport: "报告生成",
    uploadTitle: "上传数据集",
    uploadHint: "支持 CSV、XLSX 和 XLS 文件。初版建议使用干净的表格数据。",
    dropTitle: "选择数据文件",
    dropHint: "点击选择 CSV / Excel 文件",
    profileTitle: "数据画像 JSON",
    uploadButton: "上传并分析字段",
    uploading: "正在上传并生成数据画像...",
    uploadDone: "上传完成。数据集 ID：",
    uploadMissing: "请先选择一个数据文件。",
    jobTitle: "创建分析任务",
    jobHint: "目标列会作为 y，其他列会作为特征 X。",
    goalLabel: "分析目标",
    goalPlaceholder: "例如：分析哪些因素影响客户流失，并训练一个预测模型",
    targetLabel: "目标列",
    runButton: "开始自动分析",
    running: "正在运行自动分析工作流，这可能需要几十秒...",
    runMissing: "请先上传数据集，并填写分析目标。",
    runDone: "分析完成。报告已生成。",
    resultTitle: "分析结果",
    resultHint: "指标、Agent 计划、图表和报告会在任务完成后展示。",
    taskType: "任务类型",
    targetColumn: "目标列",
    bestModel: "最佳模型",
    metricsTitle: "模型指标",
    leaderboardTitle: "模型排行榜",
    leaderboardHint: "按交叉验证均值排序，留出集指标用于最终泛化检查。",
    traceTitle: "任务 Trace",
    rank: "排名",
    model: "模型",
    cvMean: "交叉验证均值",
    cvStd: "标准差",
    holdoutScore: "留出集得分",
    folds: "折数",
    duration: "耗时",
    agentPlanTitle: "Agent 计划",
    artifactsTitle: "分析图表",
    predictTitle: "新数据预测",
    predictHint: "任务完成后上传新的 CSV / Excel，系统会加载最佳模型并输出预测结果。",
    predictButton: "上传并预测",
    predictMissingJob: "请先完成一次训练任务。",
    predictMissingFile: "请先选择一个新数据文件。",
    predicting: "正在加载模型并生成预测...",
    predictDone: "预测完成。",
    predictionPath: "预测文件",
    predictionRows: "预测行数",
    reportTitle: "报告",
    errorPrefix: "错误：",
  },
  en: {
    subtitle: "A controllable tabular data analysis and modeling workflow",
    navUpload: "Dataset",
    navConfigure: "Task",
    navResults: "Results",
    navDocs: "API Docs",
    languageLabel: "Language",
    eyebrow: "Agent + AutoML Workspace",
    heroTitle: "From tabular data to model insight reports",
    heroCopy:
      "Upload data, choose a target column, and DataPilot profiles, explores, trains, evaluates, explains, and reports.",
    systemReady: "Local service ready",
    flowUpload: "Upload",
    flowProfile: "Profile",
    flowTrain: "Train",
    flowReport: "Report",
    uploadTitle: "Upload Dataset",
    uploadHint: "CSV, XLSX, and XLS files are supported. Clean tabular data works best for v1.",
    dropTitle: "Choose Data File",
    dropHint: "Select a CSV / Excel file",
    profileTitle: "Dataset Profile JSON",
    uploadButton: "Upload and Profile",
    uploading: "Uploading and profiling dataset...",
    uploadDone: "Upload complete. Dataset ID: ",
    uploadMissing: "Please choose a dataset file first.",
    jobTitle: "Create Analysis Job",
    jobHint: "The target column is y; other columns are used as X features.",
    goalLabel: "Analysis Goal",
    goalPlaceholder: "Example: analyze churn factors and train a prediction model",
    targetLabel: "Target Column",
    runButton: "Start Analysis",
    running: "Running the analysis workflow. This may take a few dozen seconds...",
    runMissing: "Please upload a dataset and enter an analysis goal first.",
    runDone: "Analysis complete. Report generated.",
    resultTitle: "Results",
    resultHint: "Metrics, agent plan, charts, and report appear after completion.",
    taskType: "Task Type",
    targetColumn: "Target Column",
    bestModel: "Best Model",
    metricsTitle: "Model Metrics",
    leaderboardTitle: "Model Leaderboard",
    leaderboardHint: "Ranked by cross-validation mean; holdout score checks final generalization.",
    traceTitle: "Task Trace",
    rank: "Rank",
    model: "Model",
    cvMean: "CV Mean",
    cvStd: "Std Dev",
    holdoutScore: "Holdout Score",
    folds: "Folds",
    duration: "Duration",
    agentPlanTitle: "Agent Plan",
    artifactsTitle: "Analysis Charts",
    predictTitle: "Predict New Data",
    predictHint:
      "After training, upload a new CSV / Excel file and DataPilot will predict with the best model.",
    predictButton: "Upload and Predict",
    predictMissingJob: "Please complete a training job first.",
    predictMissingFile: "Please choose a new data file first.",
    predicting: "Loading model and generating predictions...",
    predictDone: "Prediction complete.",
    predictionPath: "Prediction file",
    predictionRows: "Prediction rows",
    reportTitle: "Report",
    errorPrefix: "Error: ",
  },
};

let currentLanguage = localStorage.getItem("datapilot-language") || "zh";
let datasetId = null;
let currentJobId = null;

const languageSelect = document.querySelector("#languageSelect");
const fileInput = document.querySelector("#fileInput");
const uploadButton = document.querySelector("#uploadButton");
const runButton = document.querySelector("#runButton");
const predictButton = document.querySelector("#predictButton");
const datasetStatus = document.querySelector("#datasetStatus");
const jobStatus = document.querySelector("#jobStatus");
const predictStatus = document.querySelector("#predictStatus");
const profilePreview = document.querySelector("#profilePreview");
const predictFileInput = document.querySelector("#predictFileInput");
const goalInput = document.querySelector("#goalInput");
const targetInput = document.querySelector("#targetInput");
const taskType = document.querySelector("#taskType");
const targetColumn = document.querySelector("#targetColumn");
const bestModel = document.querySelector("#bestModel");
const metricsBox = document.querySelector("#metricsBox");
const agentPlanBox = document.querySelector("#agentPlanBox");
const artifacts = document.querySelector("#artifacts");
const report = document.querySelector("#report");
const metricCards = document.querySelector("#metricCards");
const flowUpload = document.querySelector("#flowUpload");
const flowProfile = document.querySelector("#flowProfile");
const flowTrain = document.querySelector("#flowTrain");
const flowReport = document.querySelector("#flowReport");
const predictionSummary = document.querySelector("#predictionSummary");
const predictionTable = document.querySelector("#predictionTable");
const leaderboardTable = document.querySelector("#leaderboardTable");
const traceList = document.querySelector("#traceList");

function t(key) {
  return i18n[currentLanguage][key] || key;
}

function applyLanguage() {
  document.documentElement.lang = currentLanguage === "zh" ? "zh-CN" : "en";
  languageSelect.value = currentLanguage;
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = t(node.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
    node.placeholder = t(node.dataset.i18nPlaceholder);
  });
}

function setStatus(node, message, kind = "") {
  node.className = `status ${kind}`;
  node.textContent = message;
}

function setFlowState(states) {
  [
    ["upload", flowUpload],
    ["profile", flowProfile],
    ["train", flowTrain],
    ["report", flowReport],
  ].forEach(([key, node]) => {
    node.classList.remove("active", "done");
    if (states[key]) {
      node.classList.add(states[key]);
    }
  });
}

function storageUrl(path) {
  const normalized = path.replaceAll("\\", "/");
  const marker = "storage/";
  const index = normalized.indexOf(marker);
  if (index >= 0) {
    return `/${normalized.slice(index)}`;
  }
  return normalized;
}

async function parseError(response) {
  try {
    const body = await response.json();
    return body.detail || response.statusText;
  } catch {
    return response.statusText;
  }
}

uploadButton.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) {
    setStatus(datasetStatus, t("uploadMissing"), "warn");
    return;
  }

  const form = new FormData();
  form.append("file", file);
  uploadButton.disabled = true;
  setStatus(datasetStatus, t("uploading"));
  setFlowState({ upload: "active" });

  try {
    const response = await fetch("/datasets/upload", {
      method: "POST",
      body: form,
    });
    if (!response.ok) {
      throw new Error(await parseError(response));
    }
    const data = await response.json();
    datasetId = data.dataset_id;
    profilePreview.textContent = JSON.stringify(data.profile, null, 2);
    if (data.profile.target_candidates?.length && !targetInput.value) {
      targetInput.value = data.profile.target_candidates[0];
    }
    setStatus(datasetStatus, `${t("uploadDone")}${datasetId}`, "ok");
    setFlowState({ upload: "done", profile: "done", train: "active" });
  } catch (error) {
    setStatus(datasetStatus, `${t("errorPrefix")}${error.message}`, "warn");
    setFlowState({ upload: "active" });
  } finally {
    uploadButton.disabled = false;
  }
});

runButton.addEventListener("click", async () => {
  if (!datasetId || !goalInput.value.trim()) {
    setStatus(jobStatus, t("runMissing"), "warn");
    return;
  }

  runButton.disabled = true;
  setStatus(jobStatus, t("running"));
  setFlowState({ upload: "done", profile: "done", train: "active" });
  artifacts.innerHTML = "";
  report.textContent = "";
  agentPlanBox.textContent = "";
  metricsBox.textContent = "";
  metricCards.innerHTML = "";
  predictionSummary.textContent = "";
  predictionTable.innerHTML = "";
  leaderboardTable.innerHTML = "";
  traceList.innerHTML = "";
  currentJobId = null;
  predictButton.disabled = true;

  try {
    const response = await fetch("/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_id: datasetId,
        user_goal: goalInput.value.trim(),
        target_column: targetInput.value.trim() || null,
      }),
    });
    if (!response.ok) {
      throw new Error(await parseError(response));
    }
    const data = await response.json();
    currentJobId = data.job_id;
    taskType.textContent = data.task_type;
    targetColumn.textContent = data.target_column || "-";
    bestModel.textContent = data.best_model || "-";
    metricsBox.textContent = JSON.stringify(data.metrics, null, 2);
    renderMetricCards(data.metrics || {});
    renderLeaderboard(data.model_leaderboard || []);
    renderTrace(data.trace || []);
    agentPlanBox.textContent = JSON.stringify(data.agent_plan || {}, null, 2);
    data.artifacts.forEach((artifact) => {
      const figure = document.createElement("figure");
      const image = document.createElement("img");
      image.src = storageUrl(artifact);
      image.alt = artifact.split(/[\\/]/).pop();
      const caption = document.createElement("figcaption");
      caption.textContent = image.alt;
      figure.appendChild(image);
      figure.appendChild(caption);
      artifacts.appendChild(figure);
    });

    const reportResponse = await fetch(`/jobs/${data.job_id}/report`);
    const reportData = await reportResponse.json();
    report.textContent = reportData.markdown;
    setStatus(jobStatus, t("runDone"), "ok");
    predictButton.disabled = false;
    setFlowState({ upload: "done", profile: "done", train: "done", report: "done" });
  } catch (error) {
    setStatus(jobStatus, `${t("errorPrefix")}${error.message}`, "warn");
    setFlowState({ upload: "done", profile: "done", train: "active" });
  } finally {
    runButton.disabled = false;
  }
});

predictButton.addEventListener("click", async () => {
  if (!currentJobId) {
    setStatus(predictStatus, t("predictMissingJob"), "warn");
    return;
  }
  const file = predictFileInput.files[0];
  if (!file) {
    setStatus(predictStatus, t("predictMissingFile"), "warn");
    return;
  }

  const form = new FormData();
  form.append("file", file);
  predictButton.disabled = true;
  setStatus(predictStatus, t("predicting"));
  predictionSummary.textContent = "";
  predictionTable.innerHTML = "";

  try {
    const response = await fetch(`/jobs/${currentJobId}/predict`, {
      method: "POST",
      body: form,
    });
    if (!response.ok) {
      throw new Error(await parseError(response));
    }
    const data = await response.json();
    setStatus(predictStatus, t("predictDone"), "ok");
    predictionSummary.textContent = `${t("predictionRows")}: ${data.row_count} | ${t(
      "predictionPath",
    )}: ${data.prediction_path}`;
    renderPredictionTable(data.preview || []);
  } catch (error) {
    setStatus(predictStatus, `${t("errorPrefix")}${error.message}`, "warn");
  } finally {
    predictButton.disabled = false;
  }
});

function renderMetricCards(metrics) {
  metricCards.innerHTML = "";
  Object.entries(metrics).forEach(([key, value]) => {
    const card = document.createElement("div");
    card.className = "metric-card";
    const label = document.createElement("span");
    label.textContent = key;
    const metric = document.createElement("strong");
    metric.textContent = typeof value === "number" ? value.toFixed(4) : value;
    card.appendChild(label);
    card.appendChild(metric);
    metricCards.appendChild(card);
  });
}

function formatScore(value) {
  return typeof value === "number" ? value.toFixed(4) : "-";
}

function renderLeaderboard(rows) {
  leaderboardTable.innerHTML = "";
  if (!rows.length) {
    return;
  }
  const columns = [
    ["rank", t("rank")],
    ["model", t("model")],
    ["cv_mean", t("cvMean")],
    ["cv_std", t("cvStd")],
    ["holdout_score", t("holdoutScore")],
    ["cv_folds", t("folds")],
  ];
  const thead = document.createElement("thead");
  const header = document.createElement("tr");
  columns.forEach(([, label]) => {
    const th = document.createElement("th");
    th.textContent = label;
    header.appendChild(th);
  });
  thead.appendChild(header);
  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    if (row.is_best) {
      tr.classList.add("best-model-row");
    }
    columns.forEach(([key]) => {
      const td = document.createElement("td");
      td.textContent =
        key === "model" && row.is_best
          ? `${row[key]} ★`
          : typeof row[key] === "number" && key !== "rank" && key !== "cv_folds"
            ? formatScore(row[key])
            : (row[key] ?? "-");
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  leaderboardTable.appendChild(thead);
  leaderboardTable.appendChild(tbody);
}

function renderTrace(rows) {
  traceList.innerHTML = "";
  rows.forEach((row, index) => {
    const item = document.createElement("div");
    item.className = `trace-item ${row.status}`;
    const marker = document.createElement("span");
    marker.className = "trace-marker";
    marker.textContent = String(index + 1).padStart(2, "0");
    const content = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = row.stage;
    const detail = document.createElement("span");
    detail.textContent = `${row.status} · ${t("duration")}: ${formatScore(
      row.duration_seconds,
    )}s${row.error ? ` · ${row.error}` : ""}`;
    content.appendChild(title);
    content.appendChild(detail);
    item.appendChild(marker);
    item.appendChild(content);
    traceList.appendChild(item);
  });
}

function renderPredictionTable(rows) {
  predictionTable.innerHTML = "";
  if (!rows.length) {
    return;
  }
  const columns = Object.keys(rows[0]);
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  columns.forEach((column) => {
    const th = document.createElement("th");
    th.textContent = column;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);

  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    columns.forEach((column) => {
      const td = document.createElement("td");
      td.textContent = row[column] ?? "";
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  predictionTable.appendChild(thead);
  predictionTable.appendChild(tbody);
}

languageSelect.addEventListener("change", () => {
  currentLanguage = languageSelect.value;
  localStorage.setItem("datapilot-language", currentLanguage);
  applyLanguage();
});

applyLanguage();
setFlowState({ upload: "active" });
