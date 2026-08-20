const form = document.querySelector("#recommendationForm");
const message = document.querySelector("#message");
const serviceStatus = document.querySelector("#serviceStatus");
const candidateList = document.querySelector("#candidateList");
const candidateCount = document.querySelector("#candidateCount");
const selectedPlayer = document.querySelector("#selectedPlayer");
const profileMetrics = document.querySelector("#profileMetrics");
const shortTermMetrics = document.querySelector("#shortTermMetrics");
const longTermMetrics = document.querySelector("#longTermMetrics");
const compensationBlock = document.querySelector("#compensationBlock");
const recentPerformance = document.querySelector("#recentPerformance");

let activeCandidateKey = "";

function setMessage(text, isError = false) {
  message.textContent = text || "";
  message.classList.toggle("error", isError);
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "N/A";
  }
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
  });
}

function formatCurrency(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "N/A";
  }
  return Number(value).toLocaleString(undefined, {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  });
}

function titleCase(text) {
  return String(text || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

async function postJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `${path} failed with status ${response.status}`);
  }
  return data;
}

function metric(label, value) {
  const item = document.createElement("div");
  item.className = "metric";
  item.innerHTML = `
    <div class="metric-label">${label}</div>
    <div class="metric-value">${value}</div>
  `;
  return item;
}

function clearReport() {
  selectedPlayer.textContent = "No player selected";
  profileMetrics.innerHTML = '<div class="empty">Select a candidate to load the report.</div>';
  shortTermMetrics.innerHTML = '<div class="empty">No forecast loaded.</div>';
  longTermMetrics.innerHTML = '<div class="empty">No forecast loaded.</div>';
  compensationBlock.innerHTML = '<div class="empty">No compensation data loaded.</div>';
  recentPerformance.innerHTML = '<div class="empty">No recent games loaded.</div>';
}

function renderCandidates(candidates) {
  candidateList.innerHTML = "";
  candidateCount.textContent = String(candidates.length);

  if (!candidates.length) {
    candidateList.innerHTML = '<div class="empty">No candidates found.</div>';
    return;
  }

  candidates.forEach((candidate, index) => {
    const playerId = candidate.player_id ?? candidate.candidate_player_id ?? "";
    const playerName = candidate.player_name ?? candidate.candidate_player_name ?? "Unknown player";
    const season = candidate.season ?? candidate.candidate_season ?? "";
    const score =
      candidate.recommendation_score ??
      candidate.role_similarity_score ??
      candidate.similarity_score ??
      candidate.score;
    const key = `${playerId}-${playerName}-${season}`;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "candidate-button";
    button.dataset.key = key;
    button.innerHTML = `
      <div>
        <div class="candidate-name">${index + 1}. ${playerName}</div>
        <div class="candidate-meta">${season || "latest"} · ${candidate.position_group || candidate.position || "position N/A"} · ${formatNumber(candidate.minutes, 0)} minutes</div>
      </div>
      <div class="candidate-score">${score === undefined ? "N/A" : formatNumber(score, 3)}</div>
    `;
    button.addEventListener("click", () => loadReport(candidate, key));
    candidateList.appendChild(button);
  });
}

function renderProfile(player) {
  profileMetrics.innerHTML = "";
  const items = [
    ["Season", player.season],
    ["Team", player.team_id],
    ["Position", player.position_group || player.position],
    ["Minutes", formatNumber(player.minutes, 0)],
    ["PTS / 100 poss", formatNumber(player.points_per_100)],
    ["AST / 100 poss", formatNumber(player.assists_per_100)],
    ["REB / 100 poss", formatNumber(player.rebounds_per_100)],
    ["Usage", formatNumber((player.usage_pct ?? 0) * 100, 1) + "%"],
  ];
  items.forEach(([label, value]) => profileMetrics.appendChild(metric(label, value ?? "N/A")));
}

function renderShortTerm(forecasts) {
  shortTermMetrics.innerHTML = "";
  if (!forecasts || !Object.keys(forecasts).length) {
    shortTermMetrics.innerHTML = '<div class="empty">No short-term forecast available.</div>';
    return;
  }
  Object.entries(forecasts).forEach(([task, forecast]) => {
    shortTermMetrics.appendChild(metric(titleCase(task), formatNumber(forecast.prediction)));
  });
}

function renderLongTerm(forecasts) {
  if (!forecasts || !Object.keys(forecasts).length) {
    longTermMetrics.innerHTML = '<div class="empty">No long-term forecast available.</div>';
    return;
  }

  const rows = [];
  Object.entries(forecasts).forEach(([task, horizons]) => {
    Object.entries(horizons || {}).forEach(([horizon, forecast]) => {
      const value = task === "active_probability"
        ? `${formatNumber(Number(forecast.prediction) * 100, 1)}%`
        : formatNumber(forecast.prediction);
      rows.push({
        task: titleCase(task),
        horizon: `H${horizon}`,
        value,
        model: forecast.model_family || "model",
      });
    });
  });

  longTermMetrics.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Task</th>
          <th>Horizon</th>
          <th>Prediction</th>
          <th>Model</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map((row) => `
          <tr>
            <td>${row.task}</td>
            <td>${row.horizon}</td>
            <td>${row.value}</td>
            <td>${row.model}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function renderCompensation(compensation) {
  compensationBlock.innerHTML = "";
  const latest = compensation?.latest_salary || {};
  compensationBlock.appendChild(metric("Latest Salary", formatCurrency(latest.salary_usd)));
  compensationBlock.appendChild(metric("Salary Cap Share", latest.salary_cap_share === undefined ? "N/A" : `${formatNumber(Number(latest.salary_cap_share) * 100, 2)}%`));

  const history = compensation?.salary_history || [];
  if (history.length) {
    const table = document.createElement("div");
    table.className = "compact-table";
    table.innerHTML = `
      <table>
        <thead>
          <tr>
            <th>Season</th>
            <th>Salary</th>
          </tr>
        </thead>
        <tbody>
          ${history.slice(-5).reverse().map((row) => `
            <tr>
              <td>${row.season_label || "N/A"}</td>
              <td>${formatCurrency(row.salary_usd)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
    compensationBlock.appendChild(table);
  }
}

function renderRecentPerformance(rows) {
  if (!rows || !rows.length) {
    recentPerformance.innerHTML = '<div class="empty">No recent game rows available.</div>';
    return;
  }

  recentPerformance.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Date</th>
          <th>Season</th>
          <th>PTS L5</th>
          <th>AST L5</th>
          <th>REB L5</th>
          <th>MIN L5</th>
        </tr>
      </thead>
      <tbody>
        ${rows.slice().reverse().map((row) => `
          <tr>
            <td>${String(row.as_of_date || "").slice(0, 10)}</td>
            <td>${row.season || "N/A"}</td>
            <td>${formatNumber(row.pts_last_5)}</td>
            <td>${formatNumber(row.ast_last_5)}</td>
            <td>${formatNumber(row.reb_last_5)}</td>
            <td>${formatNumber(row.min_last_5)}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

async function loadReport(candidate, key) {
  activeCandidateKey = key;
  document.querySelectorAll(".candidate-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.key === activeCandidateKey);
  });

  const season = candidate.season || document.querySelector("#season").value || null;
  const playerId = candidate.player_id ?? candidate.candidate_player_id;
  const playerName = candidate.player_name ?? candidate.candidate_player_name;
  selectedPlayer.textContent = playerName || "Loading report";
  setMessage(`Loading report for ${playerName || playerId}...`);

  try {
    const report = await postJson("/players/scouting-report", {
      player_id: playerId ?? null,
      player_name: playerId ? null : playerName,
      season,
      anchor_season: season,
      include_forecasts: true,
      short_term_tasks: ["points", "assists", "rebounds"],
      long_term_tasks: ["active_probability", "pts_per_36", "ast_per_36", "reb_per_36"],
      long_term_horizons: [1, 2, 3],
    });

    selectedPlayer.textContent = report.player?.player_name || playerName || "Player report";
    renderProfile(report.player || {});
    renderShortTerm(report.short_term_forecast);
    renderLongTerm(report.long_term_forecast);
    renderCompensation(report.compensation);
    renderRecentPerformance(report.recent_performance);
    setMessage("Report loaded.");
  } catch (error) {
    setMessage(error.message, true);
  }
}

async function submitRecommendation(event) {
  event.preventDefault();
  const submitButton = form.querySelector("button[type='submit']");
  submitButton.disabled = true;
  setMessage("Finding similar players...");
  clearReport();

  const minutesValue = document.querySelector("#minutesMin").value;
  const payload = {
    player_name: document.querySelector("#playerName").value.trim(),
    season: document.querySelector("#season").value.trim() || null,
    top_n: Number(document.querySelector("#topN").value || 5),
    preset: document.querySelector("#preset").value,
    same_season: document.querySelector("#sameSeason").checked,
    same_position_group: document.querySelector("#samePositionGroup").checked,
    minutes_min: minutesValue === "" ? null : Number(minutesValue),
  };

  try {
    const data = await postJson("/recommendations", payload);
    const candidates = data.recommendations || [];
    renderCandidates(candidates);
    setMessage(candidates.length ? "Candidates loaded. Select one for the full report." : "No candidates found.");
    if (candidates.length) {
      const first = candidateList.querySelector(".candidate-button");
      first?.click();
    }
  } catch (error) {
    renderCandidates([]);
    setMessage(error.message, true);
  } finally {
    submitButton.disabled = false;
  }
}

async function checkService() {
  try {
    const response = await fetch("/health");
    if (!response.ok) {
      throw new Error("Service unavailable");
    }
    serviceStatus.textContent = "API online";
    serviceStatus.classList.add("ok");
    serviceStatus.classList.remove("error");
  } catch {
    serviceStatus.textContent = "API offline";
    serviceStatus.classList.add("error");
    serviceStatus.classList.remove("ok");
  }
}

form.addEventListener("submit", submitRecommendation);
clearReport();
checkService();
