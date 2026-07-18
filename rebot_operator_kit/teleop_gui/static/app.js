"use strict";

const JOINT_LABELS = {
  shoulder_pan: "Shoulder pan",
  shoulder_lift: "Shoulder lift",
  elbow_flex: "Elbow flex",
  wrist_flex: "Wrist flex",
  wrist_yaw: "Wrist yaw",
  wrist_roll: "Wrist roll",
  gripper: "Gripper",
};

// v3 intentionally resets older saved drafts once so the verified 240 Hz /
// 2000°/s / 8.4° profile becomes the actual default on upgraded machines.
const STORAGE_KEY = "rebot.teleop.draft.v3";
const DEFAULT_PRESET = "hand_tracking";

const elements = {};
let joints = Object.keys(JOINT_LABELS);
let presets = {};
let activePreset = DEFAULT_PRESET;
let speedBaseline = null;
let appliedMultiplier = 1;
let controlToken = "";
let latestStatus = null;
let latestDevices = null;
let latestLogSequence = 0;
let pollCounter = 0;
let actionPending = false;
let noticeTimer = null;

function byId(id) {
  return document.getElementById(id);
}

function cacheElements() {
  [
    "state-pill", "state-label", "top-actual-hz", "top-target-hz", "command-title",
    "command-subtitle", "start-button", "stop-button", "notice", "notice-message",
    "notice-close", "preset-strip", "draft-badge", "apply-mode-label", "hz-input",
    "step-input", "force-input", "duration-input", "global-velocity-input",
    "apply-all-button", "joint-grid", "tracking-cap", "motor-range", "effective-cap",
    "baseline-summary", "multiplier-status", "multiplier-input", "multiplier-preview",
    "apply-multiplier-button", "set-baseline-button",
    "mode-chip", "gauge-ring", "gauge-hz", "gauge-status", "gauge-target",
    "gauge-delta", "runtime-value", "pid-value", "clamp-value", "issue-value",
    "follower-port", "leader-port", "follower-state", "leader-state", "port-owner",
    "ports-state", "log-window", "log-empty", "copy-log-button", "clear-log-button",
  ].forEach((id) => { elements[id] = byId(id); });
}

async function api(path, options = {}) {
  const headers = { Accept: "application/json", ...(options.headers || {}) };
  if (options.method === "POST") {
    headers["Content-Type"] = "application/json";
    headers["X-ReBot-Control"] = controlToken;
  }
  const response = await fetch(path, { cache: "no-store", ...options, headers });
  const data = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
  if (!response.ok) {
    const error = new Error(data.error || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return data;
}

function showNotice(message, type = "info", timeout = 5000) {
  window.clearTimeout(noticeTimer);
  elements.notice.classList.remove("hidden", "error", "success");
  if (type === "error" || type === "success") elements.notice.classList.add(type);
  elements["notice-message"].textContent = message;
  if (timeout > 0) {
    noticeTimer = window.setTimeout(() => elements.notice.classList.add("hidden"), timeout);
  }
}

function hideNotice() {
  window.clearTimeout(noticeTimer);
  elements.notice.classList.add("hidden");
}

function formatNumber(value, digits = 1) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
  return Number(value).toFixed(digits).replace(/\.0$/, "");
}

function formatDuration(totalSeconds) {
  const seconds = Math.max(0, Math.floor(Number(totalSeconds) || 0));
  const hours = String(Math.floor(seconds / 3600)).padStart(2, "0");
  const minutes = String(Math.floor((seconds % 3600) / 60)).padStart(2, "0");
  const remainder = String(seconds % 60).padStart(2, "0");
  return `${hours}:${minutes}:${remainder}`;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function roundMotion(value, digits = 3) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function formatMultiplier(value) {
  return `${formatNumber(value, 2)}×`;
}

function createSpeedBaseline(draft, label = "Custom baseline", presetKey = null) {
  return {
    label,
    presetKey,
    hz: Number(draft.hz),
    max_step: Number(draft.max_step),
    velocities: Object.fromEntries(
      joints.map((joint) => [joint, Number(draft.velocities?.[joint])]),
    ),
  };
}

function baselineIsValid(baseline) {
  return Boolean(
    baseline
    && Number.isFinite(Number(baseline.hz))
    && Number(baseline.hz) > 0
    && Number.isFinite(Number(baseline.max_step))
    && Number(baseline.max_step) > 0
    && joints.every((joint) => Number.isFinite(Number(baseline.velocities?.[joint])) && Number(baseline.velocities[joint]) > 0),
  );
}

function calculateMultiplier(multiplier) {
  if (!baselineIsValid(speedBaseline)) return null;
  const requested = Number(multiplier);
  if (!Number.isFinite(requested) || requested < 0.1 || requested > 10) return null;

  const baselineHz = Number(speedBaseline.hz);
  const baselineStep = Number(speedBaseline.max_step);
  const baselineTrackingCap = baselineHz * baselineStep;
  const rawHz = Math.round(baselineHz * requested);
  const hz = clamp(rawHz, 1, 240);
  const rawStep = (baselineTrackingCap * requested) / hz;
  const maxStep = roundMotion(clamp(rawStep, 0.01, 45), 4);
  const velocities = {};
  const achieved = [];

  joints.forEach((joint) => {
    const baselineVelocity = Number(speedBaseline.velocities[joint]);
    const rawVelocity = baselineVelocity * requested;
    const velocity = roundMotion(clamp(rawVelocity, 0.1, 2000));
    velocities[joint] = velocity;

    const baselineEffective = Math.min(baselineVelocity, baselineTrackingCap);
    const scaledEffective = Math.min(velocity, hz * maxStep);
    achieved.push(baselineEffective > 0 ? scaledEffective / baselineEffective : 0);
  });

  const achievedMin = Math.min(...achieved);
  const achievedMax = Math.max(...achieved);
  const tolerance = Math.max(0.02, requested * 0.01);
  const limited = achieved.some((value) => Math.abs(value - requested) > tolerance);

  return {
    requested,
    hz,
    max_step: maxStep,
    velocities,
    achievedMin,
    achievedMax,
    limited,
  };
}

function renderJointFields() {
  elements["joint-grid"].replaceChildren();
  joints.forEach((joint, index) => {
    const label = document.createElement("label");
    label.className = "joint-field";

    const indexLabel = document.createElement("span");
    indexLabel.className = "joint-index";
    indexLabel.textContent = `J${index + 1}`;

    const title = document.createElement("span");
    title.className = "field-title";
    title.textContent = JOINT_LABELS[joint] || joint.replaceAll("_", " ");

    const shell = document.createElement("span");
    shell.className = "input-shell";

    const input = document.createElement("input");
    input.id = `velocity-${joint}`;
    input.dataset.joint = joint;
    input.type = "number";
    input.min = "0.1";
    input.max = "2000";
    input.step = "1";
    input.value = "2000";
    input.inputMode = "decimal";
    input.addEventListener("input", handleSpeedDraftEdit);

    const unit = document.createElement("span");
    unit.className = "unit";
    unit.textContent = "°/s";

    shell.append(input, unit);
    label.append(indexLabel, title, shell);
    elements["joint-grid"].append(label);
  });
}

function renderPresets() {
  elements["preset-strip"].replaceChildren();
  Object.entries(presets).forEach(([key, preset]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "preset-button";
    button.dataset.preset = key;

    const title = document.createElement("strong");
    title.textContent = preset.label;
    const details = document.createElement("span");
    details.textContent = `${preset.hz} Hz · ${formatNumber(preset.velocity, 0)}°/s`;
    button.append(title, details);
    button.addEventListener("click", () => applyPreset(key));
    elements["preset-strip"].append(button);
  });
  updatePresetSelection();
}

function getDraft() {
  const velocities = {};
  joints.forEach((joint) => {
    velocities[joint] = Number(byId(`velocity-${joint}`).value);
  });
  return {
    hz: Number(elements["hz-input"].value),
    max_step: Number(elements["step-input"].value),
    gripper_force: Number(elements["force-input"].value),
    duration_s: Number(elements["duration-input"].value),
    velocities,
  };
}

function setDraft(draft) {
  elements["hz-input"].value = draft.hz;
  elements["step-input"].value = draft.max_step;
  elements["force-input"].value = draft.gripper_force;
  elements["duration-input"].value = draft.duration_s ?? 0;
  joints.forEach((joint) => {
    const value = draft.velocities?.[joint] ?? draft.global_velocity ?? 2000;
    byId(`velocity-${joint}`).value = value;
  });
  const values = joints.map((joint) => Number(byId(`velocity-${joint}`).value));
  if (values.every((value) => value === values[0])) {
    elements["global-velocity-input"].value = values[0];
  }
  updateDerived();
}

function applyPreset(key, persist = true) {
  const preset = presets[key];
  if (!preset) return;
  activePreset = key;
  const draft = {
    hz: preset.hz,
    max_step: preset.max_step,
    gripper_force: preset.gripper_force,
    duration_s: 0,
    velocities: Object.fromEntries(joints.map((joint) => [joint, preset.velocity])),
  };
  speedBaseline = createSpeedBaseline(draft, preset.label, key);
  appliedMultiplier = 1;
  elements["multiplier-input"].value = "1";
  setDraft(draft);
  if (persist) saveDraft();
  updatePresetSelection();
  updateMultiplierPanel();
}

function updatePresetSelection() {
  document.querySelectorAll(".preset-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.preset === activePreset);
  });
  if (appliedMultiplier !== null && appliedMultiplier !== 1 && speedBaseline) {
    elements["draft-badge"].textContent = `${formatMultiplier(appliedMultiplier)} ${speedBaseline.label}`;
  } else {
    elements["draft-badge"].textContent = activePreset === "custom"
      ? "Custom profile"
      : (presets[activePreset]?.label || "Custom profile");
  }
}

function handleSpeedDraftEdit() {
  activePreset = "custom";
  appliedMultiplier = null;
  updatePresetSelection();
  updateDerived();
  updateMultiplierPanel();
  saveDraft();
}

function handleAuxiliaryEdit() {
  activePreset = "custom";
  updatePresetSelection();
  updateDerived();
  saveDraft();
}

function updateMultiplierPanel() {
  if (!baselineIsValid(speedBaseline)) return;
  const baselineVelocities = Object.values(speedBaseline.velocities).map(Number);
  const baselineMinimum = Math.min(...baselineVelocities);
  const baselineMaximum = Math.max(...baselineVelocities);
  const velocityLabel = baselineMinimum === baselineMaximum
    ? `${formatNumber(baselineMinimum)}°/s`
    : `${formatNumber(baselineMinimum)}–${formatNumber(baselineMaximum)}°/s`;
  elements["baseline-summary"].textContent = `${speedBaseline.label} · ${formatNumber(speedBaseline.hz, 0)} Hz · ${velocityLabel} · ${formatNumber(speedBaseline.max_step)}°/cycle`;

  const preview = calculateMultiplier(elements["multiplier-input"].value);
  elements["multiplier-preview"].classList.toggle("limited", Boolean(preview?.limited));
  if (preview) {
    const previewVelocities = Object.values(preview.velocities);
    const minimum = Math.min(...previewVelocities);
    const maximum = Math.max(...previewVelocities);
    const previewVelocity = minimum === maximum
      ? `${formatNumber(minimum)}°/s`
      : `${formatNumber(minimum)}–${formatNumber(maximum)}°/s`;
    const achieved = Math.abs(preview.achievedMin - preview.achievedMax) < 0.01
      ? formatMultiplier(preview.achievedMin)
      : `${formatNumber(preview.achievedMin, 2)}–${formatNumber(preview.achievedMax, 2)}×`;
    elements["multiplier-preview"].textContent = `${formatMultiplier(preview.requested)} → ${preview.hz} Hz · ${previewVelocity} · ${formatNumber(preview.max_step)}°/cycle${preview.limited ? ` · achieved ${achieved} LIMITED` : ""}`;
  } else {
    elements["multiplier-preview"].textContent = "Enter a multiplier from 0.1× to 10×";
  }

  elements["multiplier-status"].classList.remove("limited", "override");
  if (appliedMultiplier === null) {
    elements["multiplier-status"].textContent = "Custom override";
    elements["multiplier-status"].classList.add("override");
  } else {
    const applied = calculateMultiplier(appliedMultiplier);
    elements["multiplier-status"].textContent = applied?.limited
      ? `${formatMultiplier(appliedMultiplier)} limited`
      : `${formatMultiplier(appliedMultiplier)} applied`;
    if (applied?.limited) elements["multiplier-status"].classList.add("limited");
  }

  document.querySelectorAll(".multiplier-chip").forEach((button) => {
    button.classList.toggle(
      "active",
      appliedMultiplier !== null && Number(button.dataset.multiplier) === appliedMultiplier,
    );
  });
}

function applySpeedMultiplier(value) {
  const result = calculateMultiplier(value);
  if (!result) {
    showNotice("Speed multiplier must be from 0.1× to 10×.", "error", 0);
    return;
  }

  const current = getDraft();
  appliedMultiplier = result.requested;
  activePreset = result.requested === 1 && speedBaseline.presetKey
    ? speedBaseline.presetKey
    : "custom";
  elements["multiplier-input"].value = result.requested;
  setDraft({
    ...current,
    hz: result.hz,
    max_step: result.max_step,
    velocities: result.velocities,
  });
  updatePresetSelection();
  updateMultiplierPanel();
  saveDraft();

  if (result.limited) {
    const achieved = Math.abs(result.achievedMin - result.achievedMax) < 0.01
      ? formatMultiplier(result.achievedMin)
      : `${formatNumber(result.achievedMin, 2)}–${formatNumber(result.achievedMax, 2)}×`;
    showNotice(`${formatMultiplier(result.requested)} was applied with limits; effective result is ${achieved}.`, "info", 5000);
  } else {
    showNotice(`${formatMultiplier(result.requested)} applied from ${speedBaseline.label}.`, "success", 2200);
  }
}

function setCurrentAsBaseline() {
  const draft = validateDraft(true);
  if (!draft) return;
  speedBaseline = createSpeedBaseline(draft, "Custom baseline", null);
  appliedMultiplier = 1;
  activePreset = "custom";
  elements["multiplier-input"].value = "1";
  updatePresetSelection();
  updateMultiplierPanel();
  saveDraft();
  showNotice("Current Hz, step limit, and seven motor velocities are now the 1× baseline.", "success", 3200);
}

function saveDraft() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      activePreset,
      appliedMultiplier,
      speedBaseline,
      draft: getDraft(),
    }));
  } catch (_) {
    // Local storage is optional; the controls remain fully functional without it.
  }
}

function loadSavedDraft() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    const saved = JSON.parse(stored);
    if (saved?.draft) {
      activePreset = saved.activePreset || "custom";
      setDraft(saved.draft);
      const preset = presets[activePreset];
      speedBaseline = baselineIsValid(saved.speedBaseline)
        ? createSpeedBaseline(
          saved.speedBaseline,
          saved.speedBaseline.label || "Custom baseline",
          saved.speedBaseline.presetKey || null,
        )
        : createSpeedBaseline(
          saved.draft,
          preset?.label || "Custom baseline",
          preset ? activePreset : null,
        );
      appliedMultiplier = saved.appliedMultiplier === null
        ? null
        : (Number.isFinite(Number(saved.appliedMultiplier)) ? Number(saved.appliedMultiplier) : 1);
      elements["multiplier-input"].value = appliedMultiplier ?? 1;
      updatePresetSelection();
      updateMultiplierPanel();
      saveDraft();
      return true;
    }
  } catch (_) {
    // Ignore stale or malformed local settings.
  }
  return false;
}

function markValidity(input, valid) {
  input.closest(".input-shell")?.classList.toggle("invalid", !valid);
}

function validateDraft(showError = false) {
  const draft = getDraft();
  const checks = [
    [elements["hz-input"], Number.isInteger(draft.hz) && draft.hz >= 1 && draft.hz <= 240, "Hz must be a whole number from 1 to 240."],
    [elements["step-input"], Number.isFinite(draft.max_step) && draft.max_step >= 0.01 && draft.max_step <= 45, "Step limit must be from 0.01 to 45 degrees per cycle."],
    [elements["force-input"], Number.isFinite(draft.gripper_force) && draft.gripper_force >= 0 && draft.gripper_force <= 1, "Gripper force ratio must be from 0 to 1."],
    [elements["duration-input"], Number.isFinite(draft.duration_s) && draft.duration_s >= 0 && draft.duration_s <= 86400, "Duration must be 0 to 86400 seconds."],
  ];
  joints.forEach((joint) => {
    const input = byId(`velocity-${joint}`);
    const value = draft.velocities[joint];
    checks.push([input, Number.isFinite(value) && value >= 0.1 && value <= 2000, `${JOINT_LABELS[joint] || joint} velocity must be from 0.1 to 2000°/s.`]);
  });

  let firstError = "";
  checks.forEach(([input, valid, message]) => {
    markValidity(input, valid);
    if (!valid && !firstError) firstError = message;
  });
  if (showError && firstError) showNotice(firstError, "error", 0);
  return firstError ? null : draft;
}

function updateDerived() {
  const draft = getDraft();
  const velocities = Object.values(draft.velocities).filter(Number.isFinite);
  const trackingCap = draft.hz * draft.max_step;
  const minimum = velocities.length ? Math.min(...velocities) : 0;
  const maximum = velocities.length ? Math.max(...velocities) : 0;
  const effectiveValues = velocities.map((velocity) => Math.min(
    velocity,
    Number.isFinite(trackingCap) ? trackingCap : 0,
  ));
  const effectiveMinimum = effectiveValues.length ? Math.min(...effectiveValues) : 0;
  const effectiveMaximum = effectiveValues.length ? Math.max(...effectiveValues) : 0;
  elements["tracking-cap"].textContent = `${formatNumber(trackingCap)} °/s`;
  elements["motor-range"].textContent = `${formatNumber(minimum)}–${formatNumber(maximum)} °/s`;
  elements["effective-cap"].textContent = `${formatNumber(effectiveMinimum)}–${formatNumber(effectiveMaximum)} °/s`;
  elements["top-target-hz"].textContent = Number.isFinite(draft.hz) ? draft.hz : "—";
  if (!latestStatus?.running) elements["gauge-target"].textContent = Number.isFinite(draft.hz) ? draft.hz : "—";
  validateDraft(false);
}

function setConnectionTag(element, text, className = "") {
  element.textContent = text;
  element.classList.remove("good", "warn", "bad");
  if (className) element.classList.add(className);
}

function renderStatus(status) {
  latestStatus = status;
  if (status.control_token) controlToken = status.control_token;
  const state = status.state || "FAULT";
  const running = Boolean(status.running);
  const transitional = state === "STARTING" || state === "STOPPING";
  elements["state-pill"].dataset.state = state;
  elements["state-label"].textContent = state.replaceAll("_", " ");
  elements["start-button"].disabled = running || transitional || actionPending;
  elements["stop-button"].disabled = !running || state === "STOPPING" || actionPending;
  elements["apply-mode-label"].textContent = running ? "Changes apply next Start" : "Applies on Start";

  const requestedHz = status.requested?.hz ?? Number(elements["hz-input"].value);
  const actualHz = status.actual_hz;
  elements["top-actual-hz"].textContent = formatNumber(actualHz);
  elements["top-target-hz"].textContent = formatNumber(requestedHz, 0);
  elements["gauge-hz"].textContent = formatNumber(actualHz);
  elements["gauge-target"].textContent = formatNumber(requestedHz, 0);
  elements["runtime-value"].textContent = formatDuration(status.runtime_s);
  elements["pid-value"].textContent = status.pid || "—";
  elements["clamp-value"].textContent = status.clamp_count ?? 0;
  elements["issue-value"].textContent = `${status.warning_count ?? 0} / ${status.error_count ?? 0}`;
  elements["mode-chip"].textContent = status.simulate ? "Simulation" : "Hardware";

  const progress = actualHz && requestedHz ? Math.min(1, actualHz / requestedHz) * 360 : 0;
  elements["gauge-ring"].style.setProperty("--progress", `${progress}deg`);
  if (actualHz !== null && actualHz !== undefined && requestedHz) {
    const delta = actualHz - requestedHz;
    elements["gauge-delta"].textContent = `${delta >= 0 ? "+" : ""}${formatNumber(delta)} Hz from target`;
  } else {
    elements["gauge-delta"].textContent = "Measured loop rate appears here";
  }

  const messages = {
    READY: ["Ready for manual control", "Choose your numbers, then start. Motion begins as soon as the robot connects.", "Waiting for Start"],
    STOPPED: ["Teleop stopped", "Adjust any setting and press Start when you want control again.", "Stopped cleanly"],
    STARTING: ["Connecting both arms", "Opening the leader and follower buses with your selected settings.", "Starting control loop"],
    RUNNING: ["Manual teleop is live", "Move the leader arm. Your draft edits will be used on the next Start.", "Tracking leader input"],
    STOPPING: ["Stopping teleop", "Waiting for the follower and leader disconnect routines to finish.", "Disconnecting"],
    FAULT: ["Teleop needs attention", status.fault || "Review the event log, then correct the issue and try again.", "Fault"],
  };
  const [title, subtitle, gauge] = messages[state] || messages.FAULT;
  elements["command-title"].textContent = title;
  elements["command-subtitle"].textContent = subtitle;
  elements["gauge-status"].textContent = gauge;

  const connected = status.connected || {};
  if (running) {
    setConnectionTag(elements["follower-state"], connected.follower ? "Connected" : "Opening", connected.follower ? "good" : "warn");
    setConnectionTag(elements["leader-state"], connected.leader ? "Connected" : "Opening", connected.leader ? "good" : "warn");
  } else if (latestDevices?.ok) {
    setConnectionTag(elements["follower-state"], "Detected", "good");
    setConnectionTag(elements["leader-state"], "Detected", "good");
  }
}

function renderDevices(devices) {
  latestDevices = devices;
  if (!devices.ok) {
    elements["follower-port"].textContent = devices.error || "Not found";
    elements["leader-port"].textContent = devices.error || "Not found";
    setConnectionTag(elements["follower-state"], "Unavailable", "bad");
    setConnectionTag(elements["leader-state"], "Unavailable", "bad");
    setConnectionTag(elements["ports-state"], "Unknown", "bad");
    elements["port-owner"].textContent = "Connect both arm controllers";
    return;
  }

  elements["follower-port"].textContent = devices.ports?.follower || "—";
  elements["leader-port"].textContent = devices.ports?.leader || "—";
  if (!latestStatus?.running) {
    setConnectionTag(elements["follower-state"], devices.calibration?.follower ? "Ready" : "No calibration", devices.calibration?.follower ? "good" : "bad");
    setConnectionTag(elements["leader-state"], devices.calibration?.leader ? "Ready" : "No calibration", devices.calibration?.leader ? "good" : "bad");
  }

  if (latestStatus?.running && !devices.ports_free) {
    elements["port-owner"].textContent = "Owned by this teleop process";
    setConnectionTag(elements["ports-state"], "Teleop", "good");
  } else if (devices.ports_free) {
    elements["port-owner"].textContent = "Both serial ports are available";
    setConnectionTag(elements["ports-state"], "Free", "good");
  } else {
    elements["port-owner"].textContent = "Another process owns an arm port";
    setConnectionTag(elements["ports-state"], "Busy", "bad");
  }
}

function appendLogs(entries) {
  if (!entries?.length) return;
  const logWindow = elements["log-window"];
  const nearBottom = logWindow.scrollHeight - logWindow.scrollTop - logWindow.clientHeight < 60;
  elements["log-empty"]?.remove();

  entries.forEach((entry) => {
    latestLogSequence = Math.max(latestLogSequence, Number(entry.seq) || 0);
    const row = document.createElement("div");
    row.className = "log-row";
    row.dataset.level = entry.level;

    const time = document.createElement("span");
    time.className = "log-time";
    time.textContent = entry.time;
    const level = document.createElement("span");
    level.className = "log-level";
    level.textContent = entry.level;
    const message = document.createElement("span");
    message.className = "log-message";
    message.textContent = entry.message;
    row.append(time, level, message);
    logWindow.append(row);
  });

  while (logWindow.children.length > 500) logWindow.firstElementChild?.remove();
  if (nearBottom) logWindow.scrollTop = logWindow.scrollHeight;
}

async function refreshStatus() {
  const status = await api("/api/status");
  renderStatus(status);
  return status;
}

async function refreshDevices() {
  renderDevices(await api("/api/devices"));
}

async function refreshLogs() {
  const result = await api(`/api/logs?since=${latestLogSequence}`);
  appendLogs(result.logs);
}

async function startTeleop() {
  if (actionPending) return;
  const draft = validateDraft(true);
  if (!draft) return;
  actionPending = true;
  renderStatus(latestStatus || { state: "READY" });
  try {
    if (!controlToken) await refreshStatus();
    const status = await api("/api/start", { method: "POST", body: JSON.stringify(draft) });
    renderStatus(status);
    showNotice(`Teleop is starting at ${draft.hz} Hz.`, "success", 2500);
    saveDraft();
  } catch (error) {
    showNotice(error.message, "error", 0);
    await refreshStatus().catch(() => {});
  } finally {
    actionPending = false;
    if (latestStatus) renderStatus(latestStatus);
  }
}

async function stopTeleop() {
  if (actionPending || !latestStatus?.running) return;
  actionPending = true;
  renderStatus(latestStatus);
  showNotice("Stopping teleop and waiting for both arms to disconnect…", "info", 0);
  try {
    const status = await api("/api/stop", { method: "POST", body: "{}" });
    renderStatus(status);
    showNotice("Stop command completed.", "success", 2800);
  } catch (error) {
    showNotice(error.message, "error", 0);
  } finally {
    actionPending = false;
    await Promise.allSettled([refreshStatus(), refreshDevices(), refreshLogs()]);
  }
}

async function copyLogs() {
  const rows = [...elements["log-window"].querySelectorAll(".log-row")];
  const text = rows.map((row) => [...row.children].map((part) => part.textContent).join(" ")).join("\n");
  try {
    await navigator.clipboard.writeText(text);
    showNotice("Event log copied.", "success", 1800);
  } catch (_) {
    showNotice("Clipboard access was not available.", "error", 2500);
  }
}

function clearLogs() {
  elements["log-window"].replaceChildren();
  const empty = document.createElement("div");
  empty.className = "log-empty";
  empty.id = "log-empty";
  empty.textContent = "Log display cleared. New events will appear here.";
  elements["log-window"].append(empty);
  elements["log-empty"] = empty;
}

async function pollLoop() {
  try {
    await Promise.all([
      refreshStatus(),
      refreshLogs(),
      pollCounter % 4 === 0 ? refreshDevices() : Promise.resolve(),
    ]);
    pollCounter += 1;
  } catch (error) {
    elements["state-pill"].dataset.state = "FAULT";
    elements["state-label"].textContent = "Offline";
    elements["start-button"].disabled = true;
    elements["stop-button"].disabled = true;
    showNotice(`GUI backend unavailable: ${error.message}`, "error", 0);
  } finally {
    window.setTimeout(pollLoop, 600);
  }
}

function bindEvents() {
  [elements["hz-input"], elements["step-input"]]
    .forEach((input) => input.addEventListener("input", handleSpeedDraftEdit));
  [elements["force-input"], elements["duration-input"]]
    .forEach((input) => input.addEventListener("input", handleAuxiliaryEdit));

  elements["apply-all-button"].addEventListener("click", () => {
    const value = Number(elements["global-velocity-input"].value);
    if (!Number.isFinite(value) || value < 0.1 || value > 2000) {
      showNotice("Global motor velocity must be from 0.1 to 2000°/s.", "error", 0);
      return;
    }
    joints.forEach((joint) => { byId(`velocity-${joint}`).value = value; });
    handleSpeedDraftEdit();
  });
  elements["global-velocity-input"].addEventListener("keydown", (event) => {
    if (event.key === "Enter") elements["apply-all-button"].click();
  });

  document.querySelectorAll(".multiplier-chip").forEach((button) => {
    button.addEventListener("click", () => {
      const value = Number(button.dataset.multiplier);
      elements["multiplier-input"].value = value;
      applySpeedMultiplier(value);
    });
  });
  elements["multiplier-input"].addEventListener("input", updateMultiplierPanel);
  elements["multiplier-input"].addEventListener("keydown", (event) => {
    if (event.key === "Enter") applySpeedMultiplier(elements["multiplier-input"].value);
  });
  elements["apply-multiplier-button"].addEventListener("click", () => {
    applySpeedMultiplier(elements["multiplier-input"].value);
  });
  elements["set-baseline-button"].addEventListener("click", setCurrentAsBaseline);

  elements["start-button"].addEventListener("click", startTeleop);
  elements["stop-button"].addEventListener("click", stopTeleop);
  elements["notice-close"].addEventListener("click", hideNotice);
  elements["copy-log-button"].addEventListener("click", copyLogs);
  elements["clear-log-button"].addEventListener("click", clearLogs);
  window.addEventListener("keydown", (event) => {
    const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);
    if (event.code === "Space" && !typing && latestStatus?.running) {
      event.preventDefault();
      stopTeleop();
    }
  });
}

async function initialize() {
  cacheElements();
  renderJointFields();
  bindEvents();
  try {
    const [presetData, status, devices] = await Promise.all([
      api("/api/presets"), api("/api/status"), api("/api/devices"),
    ]);
    joints = presetData.joints || joints;
    presets = presetData.presets || {};
    renderJointFields();
    if (!loadSavedDraft()) applyPreset(DEFAULT_PRESET, false);
    renderPresets();
    renderStatus(status);
    renderDevices(devices);
    await refreshLogs();
    pollLoop();
  } catch (error) {
    showNotice(`Could not connect to the teleop backend: ${error.message}`, "error", 0);
    elements["state-pill"].dataset.state = "FAULT";
    elements["state-label"].textContent = "Offline";
    elements["start-button"].disabled = true;
  }
}

document.addEventListener("DOMContentLoaded", initialize);
