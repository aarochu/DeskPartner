"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  attempts: [],
  activeId: null,
  detail: null,
  selectedIds: new Set(),
  hasMore: false,
  matching: 0,
  total: 0,
  controlToken: "",
  searchTimer: null,
};

function showNotice(message, kind = "info") {
  const node = $("notice");
  node.textContent = message;
  node.className = `notice${kind === "error" ? " error" : ""}`;
}

function clearNotice() {
  $("notice").textContent = "";
  $("notice").className = "notice hidden";
}

async function api(url, options = {}) {
  const response = await fetch(url, { cache: "no-store", ...options });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `${response.status} ${response.statusText}`);
  return payload;
}

function tagClass(tag) {
  if (tag === "Bad episode") return "tag bad";
  if (tag === "Needs review") return "tag review";
  return "tag";
}

function addDefinition(list, term, value) {
  const dt = document.createElement("dt");
  const dd = document.createElement("dd");
  dt.textContent = term;
  dd.textContent = value ?? "—";
  list.append(dt, dd);
}

function formatBytes(bytes) {
  const value = Number(bytes || 0);
  if (!value) return "missing";
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(1)} GB`;
  if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  return `${Math.round(value / 1024)} KB`;
}

function setVideo(video, artifact, rawFramesPreserved = false) {
  video.pause();
  video.removeAttribute("src");
  if (artifact?.available && artifact.url) video.src = artifact.url;
  video.load();
  const shell = video.parentElement;
  shell.classList.toggle("missing", !artifact?.available);
  const message = shell.querySelector(".media-unavailable");
  if (message) message.textContent = rawFramesPreserved
    ? "MP4 unavailable · raw frames preserved"
    : "Video unavailable";
}

function activeAttempt() {
  if (state.detail?.attempt_id === state.activeId) return state.detail;
  return state.attempts.find((item) => item.attempt_id === state.activeId) || null;
}

function updateResultCount() {
  $("result-count").textContent = `${state.attempts.length} shown · ${state.matching} matching · ${state.total} total · ${state.selectedIds.size} selected`;
}

async function loadDetail(attemptId) {
  if (!attemptId) return;
  try {
    const payload = await api(`/api/rerun/attempt?attempt_id=${encodeURIComponent(attemptId)}`);
    if (state.activeId !== attemptId) return;
    state.detail = payload.attempt || null;
    renderDetail();
  } catch (error) {
    if (state.activeId === attemptId) showNotice(`Run details are unavailable: ${error.message}`, "error");
  }
}

function renderDetail() {
  const attempt = activeAttempt();
  $("empty-detail").classList.toggle("hidden", Boolean(attempt));
  $("attempt-detail").classList.toggle("hidden", !attempt);
  if (!attempt) return;

  $("detail-tag").className = tagClass(attempt.tag);
  $("detail-tag").textContent = attempt.tag;
  $("detail-id").textContent = attempt.attempt_id;
  $("detail-task").textContent = attempt.task || "No task text recorded";
  $("selection-checkbox").checked = state.selectedIds.has(attempt.attempt_id);
  setVideo($("front-video"), attempt.artifacts?.front, attempt.raw_frames_preserved);
  setVideo($("side-video"), attempt.artifacts?.side, attempt.raw_frames_preserved);
  $("open-rerun-button").disabled = !attempt.artifacts?.rerun?.available;
  $("review-button").href = `/training?attempt_id=${encodeURIComponent(attempt.attempt_id)}#attempt-review`;

  const run = $("run-summary");
  run.replaceChildren();
  addDefinition(run, "Dataset", attempt.dataset);
  addDefinition(run, "Outcome", String(attempt.disposition || "unknown").replaceAll("_", " "));
  addDefinition(run, "LeRobot", attempt.training_included ? `Episode ${attempt.training_episode_index}` : "Excluded");
  addDefinition(run, "Created", attempt.created_at);
  addDefinition(run, "Failure", attempt.failure_label || "—");
  addDefinition(run, "RRD", formatBytes(attempt.artifacts?.rerun?.bytes));

  const timing = $("timing-summary");
  timing.replaceChildren();
  addDefinition(timing, "Frames", String(attempt.frames || 0));
  addDefinition(timing, "Duration", `${Number(attempt.duration_s || 0).toFixed(1)} s`);
  addDefinition(timing, "Dataset rate", attempt.timing?.dataset_fps ? `${attempt.timing.dataset_fps} FPS` : "—");
  addDefinition(timing, "Motor loop", Number(attempt.timing?.actual_control_hz || 0) > 0 ? `${attempt.timing.actual_control_hz} Hz` : "unavailable");
  addDefinition(timing, "Requested", attempt.timing?.requested_control_hz ? `${attempt.timing.requested_control_hz} Hz` : "—");

  const cameras = $("camera-summary");
  cameras.replaceChildren();
  [
    ["Front video", attempt.artifacts?.front],
    ["Side video", attempt.artifacts?.side],
  ].forEach(([label, artifact]) => addDefinition(cameras, label, formatBytes(artifact?.bytes)));
  ["front", "side"].forEach((key) => {
    const freshness = attempt.camera_freshness?.[key];
    if (!freshness) return;
    const age = freshness.max_age_ms_seen == null ? "age unknown" : `max ${Number(freshness.max_age_ms_seen).toFixed(0)} ms`;
    addDefinition(cameras, `${key} freshness`, `${String(freshness.status).replaceAll("_", " ")} · ${age}`);
  });
  if (attempt.raw_frames_preserved && !attempt.archive_complete) {
    addDefinition(cameras, "Recovery", "Raw frames preserved · archive incomplete");
  }

  const joints = $("joint-summary");
  joints.replaceChildren();
  (attempt.joints?.names || []).forEach((name) => {
    const chip = document.createElement("span");
    const start = attempt.joints?.start_positions_deg?.[name];
    chip.textContent = start == null ? name : `${name} ${Number(start).toFixed(1)}°`;
    joints.append(chip);
  });
}

function renderAttempts() {
  const list = $("attempt-list");
  list.replaceChildren();
  if (!state.attempts.length) {
    const empty = document.createElement("p");
    empty.className = "empty-list";
    empty.textContent = "No runs match these filters. Recorded attempts will appear here without starting the robot.";
    list.append(empty);
    state.activeId = null;
    renderDetail();
    return;
  }
  if (!state.attempts.some((item) => item.attempt_id === state.activeId)) {
    state.activeId = state.attempts[0].attempt_id;
  }
  state.attempts.forEach((attempt) => {
    const row = document.createElement("div");
    row.className = `attempt-card${attempt.attempt_id === state.activeId ? " active" : ""}`;
    const selected = document.createElement("input");
    selected.type = "checkbox";
    selected.checked = state.selectedIds.has(attempt.attempt_id);
    selected.setAttribute("aria-label", `Select ${attempt.attempt_id} for sharing`);
    selected.addEventListener("click", (event) => event.stopPropagation());
    selected.addEventListener("change", () => {
      if (selected.checked) state.selectedIds.add(attempt.attempt_id);
      else state.selectedIds.delete(attempt.attempt_id);
      updateResultCount();
      renderDetail();
    });
    const copy = document.createElement("span");
    const id = document.createElement("strong");
    const tag = document.createElement("span");
    const meta = document.createElement("small");
    id.textContent = attempt.attempt_id;
    tag.className = tagClass(attempt.tag);
    tag.textContent = attempt.tag;
    meta.textContent = `${attempt.frames || 0} frames · ${Number(attempt.duration_s || 0).toFixed(1)}s · ${attempt.training_included ? `EP ${attempt.training_episode_index}` : "excluded"}`;
    copy.append(id, tag, meta);
    const open = document.createElement("button");
    open.type = "button";
    open.className = "attempt-open";
    open.append(copy);
    row.append(selected, open);
    open.addEventListener("click", () => {
      state.activeId = attempt.attempt_id;
      state.detail = null;
      renderAttempts();
      loadDetail(attempt.attempt_id);
    });
    list.append(row);
  });
  renderDetail();
}

function setOptions(select, values, firstLabel) {
  const current = select.value;
  select.replaceChildren();
  const all = document.createElement("option");
  all.value = "";
  all.textContent = firstLabel;
  select.append(all);
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.append(option);
  });
  if ([...select.options].some((option) => option.value === current)) select.value = current;
}

async function loadCatalog({ append = false } = {}) {
  clearNotice();
  const query = new URLSearchParams();
  if ($("dataset-filter").value) query.set("dataset", $("dataset-filter").value);
  if ($("tag-filter").value) query.set("tag", $("tag-filter").value);
  if ($("search-filter").value.trim()) query.set("q", $("search-filter").value.trim());
  query.set("limit", "50");
  query.set("offset", append ? String(state.attempts.length) : "0");
  try {
    const payload = await api(`/api/rerun/catalog?${query}`);
    const page = Array.isArray(payload.attempts) ? payload.attempts : [];
    state.attempts = append ? [...state.attempts, ...page] : page;
    state.hasMore = Boolean(payload.has_more);
    state.matching = Number(payload.visible ?? state.attempts.length);
    state.total = Number(payload.total ?? state.matching);
    setOptions($("dataset-filter"), payload.datasets || [], "All datasets");
    setOptions($("tag-filter"), payload.tags || [], "All tags");
    updateResultCount();
    $("load-more-button").classList.toggle("hidden", !state.hasMore);
    renderAttempts();
    await loadDetail(state.activeId);
  } catch (error) {
    if (!append) state.attempts = [];
    $("result-count").textContent = "Catalog unavailable";
    renderAttempts();
    showNotice(`Rerun Library API is not connected yet: ${error.message}`, "error");
  }
}

async function openRerun() {
  const attempt = activeAttempt();
  if (!attempt?.artifacts?.rerun?.available) return;
  try {
    await api("/api/training/attempt/replay", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-ReBot-Control": state.controlToken },
      body: JSON.stringify({ attempt_id: attempt.attempt_id }),
    });
    showNotice(`Opening ${attempt.attempt_id} in the native Rerun viewer. No robot motion was requested.`);
  } catch (error) {
    showNotice(`Rerun visualization could not open: ${error.message}`, "error");
  }
}

async function prepareShare() {
  const ids = [...state.selectedIds];
  if (!ids.length && state.activeId) ids.push(state.activeId);
  if (!ids.length) {
    showNotice("Select at least one run first.", "error");
    return;
  }
  try {
    const result = await api("/api/rerun/share-preview", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-ReBot-Control": state.controlToken },
      body: JSON.stringify({ attempt_ids: ids, mode: "verified_lerobot_share" }),
    });
    showNotice(result.message || `Prepared ${ids.length} run${ids.length === 1 ? "" : "s"} for verified sharing.`);
  } catch (error) {
    showNotice(`Share preparation is not available yet; your ${ids.length}-run selection is preserved. ${error.message}`, "error");
  }
}

function bindSynchronizedVideos() {
  const front = $("front-video");
  const side = $("side-video");
  let synchronizing = false;
  const mirror = async (source, target, mode) => {
    if (synchronizing || !source.currentSrc || !target.currentSrc) return;
    synchronizing = true;
    try {
      if (Math.abs((target.currentTime || 0) - (source.currentTime || 0)) > 0.12) {
        target.currentTime = source.currentTime;
      }
      if (mode === "play" && target.paused) await target.play().catch(() => {});
      if (mode === "pause" && !target.paused) target.pause();
    } finally {
      synchronizing = false;
    }
  };
  [[front, side], [side, front]].forEach(([source, target]) => {
    source.addEventListener("play", () => mirror(source, target, "play"));
    source.addEventListener("pause", () => mirror(source, target, "pause"));
    source.addEventListener("seeked", () => mirror(source, target, "seek"));
    source.addEventListener("ratechange", () => { if (target.currentSrc) target.playbackRate = source.playbackRate; });
  });
  window.setInterval(() => {
    const source = !front.paused ? front : (!side.paused ? side : null);
    if (source) mirror(source, source === front ? side : front, "seek");
    const drift = front.currentSrc && side.currentSrc
      ? Math.abs((front.currentTime || 0) - (side.currentTime || 0))
      : null;
    $("video-drift").textContent = drift == null
      ? "Camera drift: unavailable"
      : `Camera drift: ${(drift * 1000).toFixed(0)} ms`;
  }, 500);
}

function bindEvents() {
  bindSynchronizedVideos();
  $("refresh-button").addEventListener("click", loadCatalog);
  $("load-more-button").addEventListener("click", () => loadCatalog({ append: true }));
  $("dataset-filter").addEventListener("change", loadCatalog);
  $("tag-filter").addEventListener("change", loadCatalog);
  $("search-filter").addEventListener("input", () => {
    window.clearTimeout(state.searchTimer);
    state.searchTimer = window.setTimeout(loadCatalog, 180);
  });
  $("clear-selection-button").addEventListener("click", () => {
    state.selectedIds.clear();
    updateResultCount();
    renderAttempts();
  });
  $("select-all-button").addEventListener("click", () => {
    state.attempts.forEach((attempt) => state.selectedIds.add(attempt.attempt_id));
    updateResultCount();
    renderAttempts();
  });
  $("selection-checkbox").addEventListener("change", (event) => {
    if (!state.activeId) return;
    if (event.target.checked) state.selectedIds.add(state.activeId);
    else state.selectedIds.delete(state.activeId);
    updateResultCount();
    renderAttempts();
  });
  $("open-rerun-button").addEventListener("click", openRerun);
  $("share-button").addEventListener("click", prepareShare);
}

async function boot() {
  bindEvents();
  try {
    const status = await api("/api/training/status");
    state.controlToken = status.control_token || "";
  } catch (_) {
    // Read-only catalog browsing remains useful if the control status endpoint is unavailable.
  }
  await loadCatalog();
}

document.addEventListener("DOMContentLoaded", boot);
