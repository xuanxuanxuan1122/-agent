const form = document.getElementById("reportForm");
const mainTitle = document.getElementById("mainTitle");
const researchDirection = document.getElementById("researchDirection");
const llmProfile = document.getElementById("llmProfile");
const skipReformatter = document.getElementById("skipReformatter");
const submitButton = document.getElementById("submitButton");
const health = document.getElementById("health");
const jobTitle = document.getElementById("jobTitle");
const jobDirection = document.getElementById("jobDirection");
const jobId = document.getElementById("jobId");
const jobStatus = document.getElementById("jobStatus");
const elapsed = document.getElementById("elapsed");
const notice = document.getElementById("notice");
const preview = document.getElementById("preview");
const files = document.getElementById("files");
const fileActions = document.getElementById("fileActions");
const tabs = Array.from(document.querySelectorAll(".tab"));

let currentJob = null;
let activeKind = "review_draft";
let pollTimer = null;

const statusText = {
  queued: "排队中",
  running: "生成中",
  completed: "已完成",
  needs_review: "待复核",
  failed: "失败"
};

function setNotice(text, tone = "") {
  notice.textContent = text;
  notice.className = `notice ${tone}`.trim();
}

async function getJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || data.error || response.statusText);
  }
  return data;
}

async function checkHealth() {
  try {
    await getJson("/api/health");
    health.textContent = "已连接";
  } catch (error) {
    health.textContent = "未连接";
  }
}

function artifact(kind) {
  return currentJob && currentJob.artifacts ? currentJob.artifacts[kind] : null;
}

function renderActions(kind) {
  fileActions.innerHTML = "";
  const item = artifact(kind);
  if (!item || !item.exists) {
    return;
  }
  const download = document.createElement("a");
  download.href = item.download_url;
  download.textContent = "下载";
  download.className = kind === "final" ? "primary" : "";
  fileActions.appendChild(download);
}

async function renderPreview(kind) {
  files.classList.add("hidden");
  preview.classList.remove("hidden");
  renderActions(kind);

  if (kind === "log") {
    preview.textContent = (currentJob.log_tail || []).join("\n") || "暂无日志。";
    return;
  }

  const item = artifact(kind);
  if (!item || !item.preview_url) {
    preview.textContent = kind === "final" ? "最终通过报告尚未生成。" : "该产物尚未生成。";
    return;
  }
  try {
    const response = await fetch(item.preview_url);
    preview.textContent = await response.text();
  } catch (error) {
    preview.textContent = String(error);
  }
}

function renderFiles() {
  preview.classList.add("hidden");
  fileActions.innerHTML = "";
  files.classList.remove("hidden");
  files.innerHTML = "";

  const entries = Object.values(currentJob.artifacts || {});
  if (!entries.length) {
    files.textContent = "暂无产物文件。";
    return;
  }

  for (const item of entries) {
    const row = document.createElement("div");
    row.className = "file-row";

    const label = document.createElement("strong");
    label.textContent = item.label;

    const filename = document.createElement("span");
    filename.textContent = `${item.filename} · ${formatSize(item.size)}`;

    const link = document.createElement("a");
    link.href = item.download_url;
    link.textContent = "打开";

    row.append(label, filename, link);
    files.appendChild(row);
  }
}

function formatSize(size) {
  if (!size) return "0 B";
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

function summarize(text, maxLength = 72) {
  const normalized = String(text || "").replace(/\s+/g, " ").trim();
  if (!normalized) return "-";
  if (normalized.length <= maxLength) return normalized;
  return `${normalized.slice(0, maxLength - 1)}…`;
}

function renderJob(job) {
  currentJob = job;
  jobTitle.textContent = summarize(job.main_title, 80);
  jobDirection.textContent = summarize(job.research_direction, 96);
  jobId.textContent = job.id || "-";
  jobStatus.textContent = statusText[job.status] || job.status || "未知";
  elapsed.textContent = `${(job.elapsed_seconds || 0).toFixed(1)}s`;
  submitButton.disabled = job.status === "queued" || job.status === "running";

  if (job.status === "completed") {
    setNotice("最终通过报告已生成。", "good");
  } else if (job.status === "needs_review") {
    setNotice("已生成待复核产物，请查看草稿或诊断报告。", "warn");
  } else if (job.status === "failed") {
    setNotice(job.error || "任务执行失败，请查看日志。", "bad");
  } else if (job.status === "running") {
    setNotice("报告生成中。");
  } else {
    setNotice("任务已创建。");
  }

  if (activeKind === "files") {
    renderFiles();
  } else {
    renderPreview(activeKind);
  }
}

async function pollJob(id) {
  try {
    const job = await getJson(`/api/reports/${id}`);
    renderJob(job);
    if (job.status === "queued" || job.status === "running") {
      pollTimer = window.setTimeout(() => pollJob(id), 2500);
    } else {
      pollTimer = null;
    }
  } catch (error) {
    setNotice(String(error), "bad");
    submitButton.disabled = false;
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const title = mainTitle.value.trim();
  const direction = researchDirection.value.trim();
  if (!title) {
    setNotice("请输入主标题。", "warn");
    mainTitle.focus();
    return;
  }

  if (pollTimer) {
    window.clearTimeout(pollTimer);
    pollTimer = null;
  }

  submitButton.disabled = true;
  preview.textContent = "";
  files.innerHTML = "";
  setNotice("正在创建任务。");

  try {
    const job = await getJson("/api/reports", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        main_title: title,
        research_direction: direction,
        llm_profile: llmProfile.value.trim(),
        skip_reformatter: skipReformatter.checked
      })
    });
    renderJob(job);
    pollJob(job.id);
  } catch (error) {
    setNotice(String(error), "bad");
    submitButton.disabled = false;
  }
});

for (const tab of tabs) {
  tab.addEventListener("click", () => {
    activeKind = tab.dataset.kind;
    tabs.forEach((item) => item.classList.toggle("active", item === tab));
    if (!currentJob) return;
    if (activeKind === "files") {
      renderFiles();
    } else {
      renderPreview(activeKind);
    }
  });
}

checkHealth();
