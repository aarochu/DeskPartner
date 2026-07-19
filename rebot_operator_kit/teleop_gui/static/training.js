"use strict";

const $ = (id) => document.getElementById(id);
let controlToken = "";
let latestStatus = null;
let latestPreflight = null;
let latestProfile = null;
let appliedProfileDigest = "";
let latestDatasets = [];
let latestAttempts = [];
let failureLabels = [];
let attemptsFingerprint = "";
let latestLog = 0;
let selectedDataset = "";
let pollCount = 0;
let actionPending = false;
let noticeTimer = null;
let previousRecordPhase = null;

const fields = {
  task: "task-input", dataset: "dataset-input", episodes: "episodes-input",
  episode_time_s: "episode-time-input", reset_time_s: "reset-time-input",
  control_hz: "control-hz-input", fps: "fps-input", motor_velocity: "velocity-input",
  max_step: "max-step-input", gripper_force: "force-input", resume: "resume-input",
  front_camera: "front-camera-input", front_width: "front-width-input",
  front_height: "front-height-input", side_camera: "side-camera-input",
  side_width: "side-width-input", side_height: "side-height-input",
  excluded_camera: "excluded-camera-input",
};

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (options.body) headers["Content-Type"] = "application/json";
  if (options.method && options.method !== "GET") headers["X-ReBot-Control"] = controlToken;
  const response = await fetch(path, { cache: "no-store", ...options, headers });
  const data = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

function showNotice(message, kind = "info", timeout = 0) {
  const notice = $("notice");
  if (noticeTimer !== null) {
    window.clearTimeout(noticeTimer);
    noticeTimer = null;
  }
  $("notice-text").textContent = message;
  notice.dataset.kind = kind;
  notice.classList.remove("hidden");
  if (timeout) {
    noticeTimer = window.setTimeout(() => {
      notice.classList.add("hidden");
      noticeTimer = null;
    }, timeout);
  }
}

function number(id) { return Number($(id).value); }
function clearFailureDraft() {
  $("failure-label-input").value = "";
  $("failure-note-input").value = "";
  updateButtons();
}
function config() {
  const result = {};
  Object.entries(fields).forEach(([key, id]) => {
    const element = $(id);
    result[key] = element.type === "checkbox" ? element.checked : (element.type === "number" ? Number(element.value) : element.value.trim());
  });
  result.profile_digest = appliedProfileDigest;
  return result;
}

function renderProfile(profile) {
  const previousDigest = latestProfile?.profile_digest || "";
  latestProfile = profile || null;
  const passed = Boolean(profile?.passed);
  const card = $("profile-lock");
  card.dataset.state = passed ? "LOCKED" : "BLOCKED";
  $("profile-state").textContent = passed ? "VERIFIED & LOCKED" : "BLOCKED";
  $("profile-id").textContent = profile?.profile_id || "Profile unavailable";
  $("profile-digest").textContent = profile?.profile_digest ? profile.profile_digest.slice(0, 12) : "—";
  const fileStates = Object.values(profile?.calibration_files || {});
  $("profile-calibration").textContent = fileStates.length
    ? `${fileStates.filter((item) => item.matches).length}/${fileStates.length} fingerprints match`
    : (profile?.error || "—");
  const contract = profile?.profile?.coordinate_contract || {};
  $("profile-joints").textContent = `${contract.action_dimension || "—"}D · follower degrees`;
  const cameras = profile?.profile?.camera_defaults || {};
  const frontName = cameras.front?.display_name || "overhead";
  const sideName = cameras.side?.display_name || "wrist";
  const excludedName = cameras.excluded_screen_name || "Mac camera";
  $("profile-cameras").textContent = cameras.front && cameras.side
    ? `${frontName} ${cameras.front.index} · ${sideName} ${cameras.side.index} · ${excludedName} ${cameras.excluded_screen_index} excluded`
    : "—";
  const training = profile?.profile?.training_defaults || {};
  $("profile-normalization").textContent = training.state_normalization
    ? `${training.state_normalization} state/action · gripper ${training.normalize_gripper ? "normalized" : "raw"}`
    : "—";
  if (previousDigest && profile?.profile_digest && previousDigest !== profile.profile_digest) {
    applyProfileDefaults(profile);
    showNotice(
      `Training profile changed to ${profile.profile_id}; verified defaults were reapplied. Review them before collection.`,
      "info",
      7000,
    );
  }
}

function applyProfileDefaults(profile = latestProfile, announce = false) {
  const defaults = profile?.defaults;
  if (!defaults) return;
  Object.entries(fields).forEach(([key, id]) => {
    if (!(key in defaults)) return;
    const element = $(id);
    if (element.type === "checkbox") element.checked = Boolean(defaults[key]);
    else element.value = defaults[key];
  });
  $("front-index-label").textContent = defaults.front_camera;
  $("side-index-label").textContent = defaults.side_camera;
  $("ready-checkbox").checked = false;
  appliedProfileDigest = profile.profile_digest || "";
  updateButtons();
  if (announce) showNotice(`Restored ${profile.profile_id} verified defaults.`, "success", 2400);
}

function formatBytes(value) {
  if (!Number.isFinite(value)) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let index = 0;
  let amount = value;
  while (amount >= 1024 && index < units.length - 1) { amount /= 1024; index += 1; }
  return `${amount.toFixed(index > 2 ? 2 : 1)} ${units[index]}`;
}

function formatDuration(seconds) {
  seconds = Math.max(0, Math.floor(Number(seconds) || 0));
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

function setGate(prefix, good, title, detail) {
  const strong = $(`${prefix}-gate`);
  strong.textContent = title;
  $(`${prefix}-detail`).textContent = detail || "—";
  strong.closest(".gate").classList.toggle("good", Boolean(good));
  strong.closest(".gate").classList.toggle("bad", !good);
}

function renderCamera(name, camera, minimumFps = 0) {
  const good = Boolean(camera?.opened
    && Number(camera?.measured_fps) >= Number(minimumFps || 0)
    && !camera?.dark_or_covered
    && !camera?.shape_mismatch
    && !camera?.shape_changed
    && !camera?.error);
  $(`${name}-camera-dot`).className = good ? "good" : "bad";
  $(`${name}-shape`).textContent = camera?.width ? `${camera.width} × ${camera.height}` : "—";
  $(`${name}-fps`).textContent = Number.isFinite(camera?.measured_fps) ? `${camera.measured_fps.toFixed(1)} FPS` : "—";
  $(`${name}-brightness`).textContent = Number.isFinite(camera?.brightness_mean) ? camera.brightness_mean.toFixed(1) : "—";
  if (camera?.index !== undefined) $(`${name}-index-label`).textContent = camera.index;
}

function renderPreflight(preflight) {
  latestPreflight = preflight;
  renderProfile(preflight.training_profile);
  const devices = preflight.devices || {};
  const calibration = devices.calibration || {};
  setGate("follower", devices.ok && calibration.follower, calibration.follower ? "Ready" : "Blocked", devices.ports?.follower || devices.error);
  setGate("leader", devices.ok && calibration.leader, calibration.leader ? "Ready" : "Blocked", devices.ports?.leader || devices.error);
  setGate("ports", devices.ok && devices.ports_free, devices.ports_free ? "Free" : "Busy", devices.ports_free ? "Exclusive access available" : "Stop teleop before collecting");
  const report = preflight.camera_report || {};
  setGate("camera", report.passed, report.passed ? "Passed" : "Blocked", report.checked_at || "Run a fresh check");
  setGate("disk", preflight.disk?.free_bytes >= 5 * 1024 ** 3, formatBytes(preflight.disk?.free_bytes), preflight.disk?.data_root || "—");
  renderCamera("front", report.cameras?.front, report.minimum_fps);
  renderCamera("side", report.cameras?.side, report.minimum_fps);
  $("data-root-footer").textContent = preflight.paths?.data || "—";
  if (latestStatus?.running) {
    return;
  }
  if (!preflight.training_profile?.passed) {
    showNotice(preflight.training_profile?.error || "The calibration/training profile is not verified.", "error", 0);
  } else if (!report.passed) {
    showNotice("Recording is blocked until a fresh dual-camera check passes and both snapshots are visually useful.", "error", 0);
  } else if (preflight.ready_to_record) {
    showNotice(
      "Ready to collect. Put both arms in the exact session-home pose, check readiness, then Start collection will capture those poses as it connects.",
      "success",
      5000,
    );
  }
  updateButtons();
}

function renderStatus(status) {
  const oldPhase = previousRecordPhase;
  latestStatus = status;
  previousRecordPhase = status.record_phase || null;
  if (status.control_token) controlToken = status.control_token;
  $("job-pill").dataset.state = status.state || "READY";
  $("job-state").textContent = status.state || "READY";
  const phaseLabels = {
    starting: "Connecting arms",
    recording: "Recording episode",
    awaiting_decision: "Waiting for keep / fail",
    saving_rerun: "Saving Rerun replay",
    saving_lerobot: "Checkpointing LeRobot",
    returning_home: "Returning to session home",
  };
  $("job-kind").textContent = phaseLabels[status.record_phase]
    || (status.kind ? status.kind.replaceAll("_", " ") : "Idle");
  $("job-runtime").textContent = formatDuration(status.runtime_s);
  updateButtons();
  if (status.fault) {
    const previousJob = !status.running && status.state === "FAULT";
    showNotice(
      previousJob
        ? `Previous job ended: ${status.fault}. This saved status does not block a corrected retry.`
        : status.fault,
      previousJob ? "info" : "error",
      0,
    );
  } else if (previousRecordPhase !== oldPhase) {
    if (previousRecordPhase === "saving_rerun") {
      showNotice("Keep accepted. Saving the two replay videos and Rerun episode now—do not press Stop.", "info", 0);
    } else if (previousRecordPhase === "saving_lerobot") {
      showNotice("Replay saved. Writing and fresh-loading the LeRobot checkpoint now—do not press Stop.", "info", 0);
    } else if (previousRecordPhase === "returning_home") {
      showNotice("Episode is durable in LeRobot. The arm is returning to the captured session-home pose.", "success", 0);
    } else if (previousRecordPhase === "recording" && oldPhase === "returning_home") {
      showNotice("Next episode is ready and recording. Move one can to a new reachable position.", "success", 5000);
    }
  }
}

function updateButtons() {
  const running = Boolean(latestStatus?.running);
  const recording = running && latestStatus.kind === "record";
  const decisionPending = Boolean(latestStatus?.decision_pending);
  const profileCurrent = Boolean(
    latestProfile?.passed
    && latestProfile.profile_digest
    && appliedProfileDigest === latestProfile.profile_digest
  );
  const ready = Boolean(latestPreflight?.ready_to_record && profileCurrent);
  $("camera-check-button").disabled = running || actionPending;
  $("start-record-button").disabled = running || actionPending || !ready || !$("ready-checkbox").checked;
  $("finish-button").disabled = !recording || decisionPending || actionPending;
  $("finish-stop-button").disabled = !recording || decisionPending || actionPending;
  const failureLabel = $("failure-label-input").value;
  const failureNote = $("failure-note-input").value.trim();
  const failureReady = Boolean(failureLabel && (failureLabel !== "other" || failureNote));
  $("rerecord-button").disabled = !recording || decisionPending || actionPending || !failureReady;
  $("stop-record-button").disabled = !recording || actionPending;
  $("validate-button").disabled = running || actionPending || !selectedDataset;
  const selected = latestDatasets.find((item) => item.name === selectedDataset);
  $("recipe-button").disabled = !selectedDataset || !selected?.validation_passed;
}

function failureLabelText(value) {
  return failureLabels.find((item) => item.value === value)?.label || value || "Unlabeled";
}

function setFailureLabelOptions(labels) {
  failureLabels = labels || [];
  const select = $("failure-label-input");
  const selected = select.value;
  select.replaceChildren();
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Choose after a failed attempt…";
  select.append(placeholder);
  failureLabels.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.value;
    option.textContent = item.label;
    select.append(option);
  });
  if (failureLabels.some((item) => item.value === selected)) select.value = selected;
  updateButtons();
}

function makeFailureSelect(value) {
  const select = document.createElement("select");
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Choose a failure reason…";
  placeholder.selected = !value;
  select.append(placeholder);
  failureLabels.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.value;
    option.textContent = item.label;
    option.selected = item.value === value;
    select.append(option);
  });
  return select;
}

function renderAttempts(payload) {
  const fingerprint = JSON.stringify(payload || {});
  if (fingerprint === attemptsFingerprint) return;
  attemptsFingerprint = fingerprint;
  latestAttempts = payload?.attempts || [];
  if (payload?.failure_labels) setFailureLabelOptions(payload.failure_labels);
  const list = $("attempt-list");
  list.replaceChildren();
  if (!latestAttempts.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "No archived attempts yet. Every new take will appear here.";
    list.append(empty);
    return;
  }
  latestAttempts.forEach((attempt) => {
    const state = attempt.disposition || "unknown";
    const manuallyFailed = state === "failed" || attempt.operator_disposition === "failed";
    const includedKept = state === "kept" && attempt.training_included === true;
    const alreadyExcluded = attempt.training_included === false && state !== "recording";
    const reviewable = attempt.archive_complete === true && (includedKept || alreadyExcluded);
    const card = document.createElement("article");
    card.className = "attempt-card";
    card.dataset.state = state;

    const summary = document.createElement("div");
    summary.className = "attempt-summary";
    const title = document.createElement("div");
    title.className = "attempt-title";
    const id = document.createElement("strong");
    id.textContent = attempt.attempt_id;
    const badge = document.createElement("b");
    badge.textContent = state === "kept" ? "Unlabeled success" : state.replaceAll("_", " ");
    title.append(id, badge);
    const meta = document.createElement("p");
    meta.className = "attempt-meta";
    const included = attempt.training_included ? `training episode ${attempt.training_episode_index}` : "excluded from training";
    meta.textContent = `${attempt.dataset || "—"} · ${attempt.samples || 0} frames · ${Number(attempt.duration_s || 0).toFixed(1)}s · ${included} · ${attempt.started_at || "—"}`;
    summary.append(title, meta);
    if (attempt.failure_label) {
      const label = document.createElement("p");
      label.className = "attempt-label";
      label.textContent = `Failure: ${failureLabelText(attempt.failure_label)}${attempt.failure_note ? ` — ${attempt.failure_note}` : ""}`;
      summary.append(label);
    }
    const actions = document.createElement("div");
    actions.className = "attempt-actions";
    if (attempt.artifacts?.rerun?.available) {
      const replay = document.createElement("button");
      replay.type = "button";
      replay.className = "button secondary";
      replay.textContent = "Replay in Rerun";
      replay.addEventListener("click", () => runAction("Rerun replay failed", async () => {
        await post("/api/training/attempt/replay", { attempt_id: attempt.attempt_id });
        showNotice(`Opening ${attempt.attempt_id} in Rerun.`, "success", 2600);
      }));
      actions.append(replay);
    }
    [["overhead", "Overhead MP4"], ["wrist", "Wrist MP4"]].forEach(([camera, label]) => {
      if (!attempt.artifacts?.[camera]?.available) return;
      const link = document.createElement("a");
      link.className = "button secondary";
      link.href = `/api/training/attempt/video?attempt_id=${encodeURIComponent(attempt.attempt_id)}&camera=${camera}`;
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = label;
      actions.append(link);
    });
    summary.append(actions);

    const edit = document.createElement("div");
    edit.className = `attempt-edit${reviewable ? "" : " hidden"}`;
    if (reviewable) {
      const select = makeFailureSelect(attempt.failure_label);
      const note = document.createElement("input");
      note.maxLength = 500;
      note.placeholder = "Failure note";
      note.value = attempt.failure_note || "";
      const help = document.createElement("small");
      help.textContent = includedKept
        ? "If review shows this take failed or was only a test, it will be removed from LeRobot. Both videos and the Rerun replay remain preserved."
        : "Save a review reason for this excluded take. Its recorded system outcome and raw files remain preserved.";
      const save = document.createElement("button");
      save.type = "button";
      save.className = "button danger-outline";
      save.textContent = includedKept
        ? "Mark failed & exclude from LeRobot"
        : (manuallyFailed ? "Update failure label" : "Label excluded attempt");
      save.addEventListener("click", () => runAction("Failure label could not be saved", async () => {
        if (!select.value) throw new Error("Choose a failure reason first");
        if (includedKept && !window.confirm(
          `Mark ${attempt.attempt_id} failed and remove training episode ${attempt.training_episode_index} from LeRobot? The MP4s and Rerun replay will be kept.`,
        )) return;
        showNotice(
          includedKept
            ? "Reclassifying this finished episode and rebuilding the success-only LeRobot dataset. Keep this page open…"
            : "Saving the review label…",
          "info",
          0,
        );
        const result = await post("/api/training/attempt/review", {
          attempt_id: attempt.attempt_id,
          action: includedKept ? "mark_failed" : "label_excluded",
          failure_label: select.value,
          failure_note: note.value.trim(),
          expected_revision: Number(attempt.review_revision || 0),
        });
        if (result.dataset_empty) $("resume-input").checked = false;
        const [attempts, datasets] = await Promise.all([
          api("/api/training/attempts"),
          api("/api/training/datasets"),
        ]);
        renderAttempts(attempts);
        renderDatasets(datasets.datasets);
        showNotice(result.message || "Finished-attempt review saved.", "success", 7000);
      }));
      edit.append(select, note, help, save);
    }
    card.append(summary, edit);
    list.append(card);
  });
}

function renderDatasets(datasets) {
  latestDatasets = datasets || [];
  const list = $("dataset-list");
  list.replaceChildren();
  if (!datasets?.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "No recorded datasets yet.";
    list.append(empty);
    selectedDataset = "";
    updateButtons();
    return;
  }
  if (!selectedDataset || !datasets.some((item) => item.name === selectedDataset)) selectedDataset = datasets.at(-1).name;
  datasets.forEach((item) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = `dataset-row${item.name === selectedDataset ? " selected" : ""}`;
    const copy = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = item.name;
    const meta = document.createElement("small");
    meta.textContent = item.ready
      ? `${item.frames || 0} frames · ${item.fps || "—"} FPS · ${(item.camera_keys || []).join(" + ")} · action ${JSON.stringify(item.action_shape)} · ${item.profile_compatible ? "PROFILE LOCKED" : "PROFILE MISSING/MISMATCH"} · ${item.validation_passed ? "VALIDATED" : "NOT VALIDATED"}`
      : "Dataset folder exists but is not finalized";
    const count = document.createElement("b");
    count.textContent = `${item.episodes || 0} EP`;
    copy.append(title, meta);
    row.append(copy, count);
    row.addEventListener("click", () => { selectedDataset = item.name; renderDatasets(datasets); });
    list.append(row);
  });
  updateButtons();
}

function appendLogs(entries) {
  if (!entries?.length) return;
  const log = $("training-log");
  const nearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 50;
  log.querySelector(".empty")?.remove();
  entries.forEach((entry) => {
    latestLog = Math.max(latestLog, Number(entry.seq) || 0);
    const row = document.createElement("div");
    row.className = "log-row";
    row.dataset.level = entry.level;
    [entry.time, entry.level, entry.message].forEach((value) => {
      const span = document.createElement("span");
      span.textContent = value;
      row.append(span);
    });
    log.append(row);
  });
  while (log.children.length > 600) log.firstElementChild?.remove();
  if (nearBottom) log.scrollTop = log.scrollHeight;
}

async function refreshAll() {
  const calls = [api("/api/training/status"), api(`/api/training/logs?since=${latestLog}`)];
  if (pollCount % 4 === 0) calls.push(
    api("/api/training/preflight"),
    api("/api/training/datasets"),
    api("/api/training/attempts"),
  );
  const results = await Promise.all(calls);
  renderStatus(results[0]);
  appendLogs(results[1].logs);
  if (results[2]) {
    renderPreflight(results[2]);
    renderDatasets(results[3].datasets);
    renderAttempts(results[4]);
    const stamp = Date.now();
    $("front-preview").src = `/camera/front.png?t=${stamp}`;
    $("side-preview").src = `/camera/side.png?t=${stamp}`;
  }
  pollCount += 1;
}

async function post(path, body) {
  if (!controlToken) renderStatus(await api("/api/training/status"));
  return api(path, { method: "POST", body: JSON.stringify(body || {}) });
}

async function runAction(label, action) {
  if (actionPending) return;
  actionPending = true;
  updateButtons();
  try { await action(); }
  catch (error) { showNotice(`${label}: ${error.message}`, "error", 0); }
  finally { actionPending = false; await refreshAll().catch(() => {}); updateButtons(); }
}

function bind() {
  $("notice-close").addEventListener("click", () => $("notice").classList.add("hidden"));
  $("refresh-button").addEventListener("click", () => runAction("Refresh failed", refreshAll));
  $("restore-defaults-button").addEventListener("click", () => applyProfileDefaults(latestProfile, true));
  $("refresh-attempts-button").addEventListener("click", () => runAction("Attempt archive refresh failed", async () => {
    renderAttempts(await api("/api/training/attempts"));
  }));
  $("ready-checkbox").addEventListener("change", updateButtons);
  $("camera-check-button").addEventListener("click", () => runAction("Camera check could not start", async () => {
    renderStatus(await post("/api/training/camera-check", config()));
    showNotice("Dual-camera check started. Keep both views still for six seconds.", "info", 7000);
  }));
  $("start-record-button").addEventListener("click", () => runAction("Collection could not start", async () => {
    renderStatus(await post("/api/training/record/start", config()));
    clearFailureDraft();
    showNotice("Collection is starting. Keep both arms in the desired home pose until they connect and that pose is captured; then begin teleoperation.", "success", 7500);
  }));
  $("finish-button").addEventListener("click", () => runAction("Episode control failed", async () => {
    await post("/api/training/record/control", { action: "finish" });
    clearFailureDraft();
    showNotice(
      "Save accepted. Writing this episode to Rerun and LeRobot now. Do not press Stop; wait until the next attempt is ready.",
      "info",
      0,
    );
  }));
  $("finish-stop-button").addEventListener("click", () => runAction("Episode control failed", async () => {
    await post("/api/training/record/control", { action: "finish_and_stop" });
    clearFailureDraft();
    showNotice(
      "Save-and-end accepted. This episode is being written to Rerun and LeRobot; the arms disconnect only after the durable save completes.",
      "info",
      0,
    );
  }));
  $("rerecord-button").addEventListener("click", () => runAction("Episode control failed", async () => {
    const failureLabel = $("failure-label-input").value;
    const failureNote = $("failure-note-input").value.trim();
    await post("/api/training/record/control", {
      action: "rerecord",
      failure_label: failureLabel,
      failure_note: failureNote,
    });
    clearFailureDraft();
    showNotice("Failed take is archived and excluded. The follower now returns home automatically; return the passive leader and reset the task objects before the next attempt.", "info", 8500);
  }));
  $("stop-record-button").addEventListener("click", () => runAction("Stop failed", async () => {
    if (!window.confirm("Discard the current take? It will be archived for review but will NOT be added to LeRobot.")) return;
    await post("/api/training/record/control", { action: "stop" });
    clearFailureDraft();
    showNotice("Discard requested. The current take is being archived as aborted and excluded from training; earlier saved episodes remain unchanged.", "info", 9000);
  }));
  $("validate-button").addEventListener("click", () => runAction("Validation could not start", async () => {
    renderStatus(await post("/api/training/validate", { dataset: selectedDataset, minimum_episodes: number("minimum-episodes-input") }));
    showNotice(`Validating ${selectedDataset}…`, "info", 4000);
  }));
  $("recipe-button").addEventListener("click", () => runAction("Recipe could not be generated", async () => {
    const recipe = await api(`/api/training/recipe?dataset=${encodeURIComponent(selectedDataset)}`);
    $("recipe-profile").textContent = `${recipe.training_profile?.id || "profile"} · ${recipe.training_profile?.digest?.slice(0, 12) || "—"}`;
    $("recipe-setup").textContent = (recipe.host_setup || []).join("\n");
    $("recipe-command").textContent = recipe.command_text;
    $("recipe-post").textContent = (recipe.post_training || []).join("\n");
    $("recipe-card").classList.remove("hidden");
    $("recipe-card").scrollIntoView({ behavior: "smooth", block: "center" });
  }));
  $("copy-command-button").addEventListener("click", async () => {
    const setup = $("recipe-setup").textContent || "";
    const command = $("recipe-command").textContent || "";
    const postTraining = $("recipe-post").textContent || "";
    await navigator.clipboard.writeText(`${setup}\n\n${command}\n\n${postTraining}`.trim());
    showNotice("GPU setup and training command copied.", "success", 1800);
  });
  $("copy-log-button").addEventListener("click", async () => {
    const text = [...$("training-log").querySelectorAll(".log-row")].map((row) => [...row.children].map((part) => part.textContent).join(" ")).join("\n");
    await navigator.clipboard.writeText(text);
    showNotice("Workspace log copied.", "success", 1800);
  });
  $("front-camera-input").addEventListener("input", () => $("front-index-label").textContent = $("front-camera-input").value);
  $("side-camera-input").addEventListener("input", () => $("side-index-label").textContent = $("side-camera-input").value);
  $("failure-label-input").addEventListener("change", updateButtons);
  $("failure-note-input").addEventListener("input", updateButtons);
  Object.values(fields).forEach((id) => $(id).addEventListener("input", updateButtons));
}

async function poll() {
  try { await refreshAll(); }
  catch (error) { showNotice(`Training GUI backend unavailable: ${error.message}`, "error", 0); }
  finally { window.setTimeout(poll, 700); }
}

document.addEventListener("DOMContentLoaded", async () => {
  bind();
  try {
    const [status, profile, preflight, datasets, logs, attempts] = await Promise.all([
      api("/api/training/status"), api("/api/training/profile"), api("/api/training/preflight"), api("/api/training/datasets"), api("/api/training/logs?since=0"),
      api("/api/training/attempts"),
    ]);
    renderStatus(status);
    renderProfile(profile);
    applyProfileDefaults(profile);
    renderPreflight(preflight);
    renderDatasets(datasets.datasets);
    appendLogs(logs.logs);
    renderAttempts(attempts);
    pollCount = 1;
    poll();
  } catch (error) {
    showNotice(`Could not initialize training workspace: ${error.message}`, "error", 0);
  }
});
