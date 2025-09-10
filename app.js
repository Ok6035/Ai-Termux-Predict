// Color Prediction Game — browser version
// Persists to localStorage. Converted from Termux bash script.

(() => {
  // Config
  const SERVER_URL = "https://www.tashanwin27.com/#/main";
  const GAME_UID = "1969524";
  const VERSION = "2.0";
  const STORAGE_KEY = "color_prediction_game:data";
  const STATS_KEY = "color_prediction_game:stats";

  // DOM
  const panel = document.getElementById("panel");
  const logArea = document.getElementById("logArea");
  document.getElementById("serverUrl").href = SERVER_URL;
  document.getElementById("gameUid").textContent = GAME_UID;

  // Game state
  let results = []; // {period: string, result: number, type: "big"|"small"}
  let trends = [];
  let currentPrediction = null; // {type: "big"|"small", suggestedColor: "red"|"green", inputDigits}
  let stats = { total_predictions: 0, correct_predictions: 0 };

  // Utilities
  const log = (text, type = "info") => {
    const time = new Date().toLocaleTimeString();
    const p = document.createElement("div");
    p.textContent = `[${time}] ${text}`;
    p.style.marginBottom = "6px";
    if (type === "error") p.style.color = "#fca5a5";
    if (type === "success") p.style.color = "#86efac";
    logArea.prepend(p);
  };

  const saveData = () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(results));
    localStorage.setItem(STATS_KEY, JSON.stringify(stats));
  };

  const loadData = () => {
    const raw = localStorage.getItem(STORAGE_KEY);
    const rawStats = localStorage.getItem(STATS_KEY);
    if (raw) {
      try {
        results = JSON.parse(raw);
      } catch (e) {
        results = [];
        log("Corrupt saved data — resetting.", "error");
      }
    } else {
      // Seed sample data (like the bash script)
      results = [
        { period: "100", result: 5, type: "big" },
        { period: "101", result: 3, type: "small" },
        { period: "102", result: 8, type: "big" },
        { period: "103", result: 2, type: "small" },
        { period: "104", result: 7, type: "big" },
      ];
      saveData();
      log("Created sample data set.");
    }
    if (rawStats) {
      try { stats = JSON.parse(rawStats); } catch {}
    }
  };

  // Trend analysis similar to original script
  const analyzeTrends = () => {
    trends = [];
    for (let i = 0; i < results.length; i++) {
      const r = results[i].type;
      if (trends.length >= 1) {
        const last = trends[trends.length - 1];
        if (last === r) trends.push(r);
        else trends.push("change");
      } else {
        trends.push(r);
      }
    }
  };

  // Prediction algorithm translated
  // lastDigits must be exactly 3 digits string
  const predictResult = (lastDigits) => {
    if (!lastDigits || !/^\d{3}$/.test(lastDigits)) {
      throw new Error("Please enter exactly 3 digits");
    }
    const num = parseInt(lastDigits, 10);

    // Count big/small in the last 10 (or fewer)
    const lastTen = results.slice(-10);
    let big_count = 0, small_count = 0;
    lastTen.forEach(r => (r.type === "big" ? big_count++ : small_count++));

    let predictionType = "";
    if (results.length >= 5) {
      const lastFive = results.slice(-5);
      let same_count = 0;
      for (let i = 0; i < 4; i++) {
        if (lastFive[i].type === lastFive[i + 1].type) same_count++;
      }
      if (same_count >= 3) {
        // High chance of change — flip the last type
        const lastType = lastFive[4].type;
        predictionType = lastType === "big" ? "small" : "big";
      } else {
        predictionType = big_count > small_count ? "big" : "small";
      }
    } else {
      // small fallback rule from original: parity
      predictionType = num % 2 === 0 ? "small" : "big";
    }

    // Color suggestion
    const suggestedColor = num % 3 === 0 ? "red" : "green";

    currentPrediction = { type: predictionType, suggestedColor, inputDigits: lastDigits };
    log(`Prediction: ${predictionType} (color: ${suggestedColor})`, "success");
    return currentPrediction;
  };

  const addNewResult = (period, resultValue) => {
    if (!period || period.toString().trim() === "") throw new Error("Period is required");
    if (!/^[0-9]$/.test(String(resultValue))) throw new Error("Result must be single digit 0-9");

    const resultNum = parseInt(resultValue, 10);
    const type = resultNum >= 5 ? "big" : "small";
    results.push({ period: String(period), result: resultNum, type });
    saveData();
    analyzeTrends();

    // Update prediction accuracy if we had a pending prediction
    if (currentPrediction && currentPrediction.type) {
      stats.total_predictions = (stats.total_predictions || 0) + 1;
      if (type === currentPrediction.type) {
        stats.correct_predictions = (stats.correct_predictions || 0) + 1;
        log("Last prediction was correct!", "success");
      } else {
        log("Last prediction was incorrect.", "error");
      }
      saveData();
    }
    log(`Added result: Period ${period}, result ${resultNum}, type ${type}`, "success");
  };

  // Simulated server connection
  const connectToServer = async () => {
    log("Connecting to server...", "info");
    log(`Server: ${SERVER_URL}`, "info");
    log(`Game UID: ${GAME_UID}`, "info");
    // Simulate delay
    await new Promise(r => setTimeout(r, 800));
    log("Connected successfully. Fetched latest results (simulated).", "success");
  };

  // UI rendering functions
  const renderHome = () => {
    panel.innerHTML = `
      <h2>Welcome</h2>
      <p class="hint">Use the buttons on the left to predict, add results, connect to server, or view stats and history.</p>
      <div style="margin-top:12px">
        <span class="pill">Total results: ${results.length}</span>
        <span class="pill">Accuracy: ${getAccuracy()}%</span>
        <span class="pill">Total predictions: ${stats.total_predictions || 0}</span>
      </div>
    `;
  };

  const renderPredict = () => {
    panel.innerHTML = `
      <h2>Predict next result</h2>
      <div class="form-row">
        <label>Enter last 3 digits:</label>
        <input id="inputDigits" type="text" maxlength="3" placeholder="e.g. 123" />
      </div>
      <div style="display:flex;gap:8px;">
        <button id="doPredict" class="btn">Predict</button>
        <button id="clearPrediction" class="btn secondary">Clear</button>
      </div>
      <div id="predictionResult" style="margin-top:12px"></div>
      <div class="hint">Algorithm uses recent history and simple heuristics similar to the original script.</div>
    `;

    document.getElementById("doPredict").addEventListener("click", () => {
      const val = document.getElementById("inputDigits").value.trim();
      const out = document.getElementById("predictionResult");
      out.textContent = "";
      try {
        const p = predictResult(val);
        out.innerHTML = `<div class="stat-card">Prediction: <strong>${p.type}</strong> &nbsp;&nbsp; Color suggestion: <strong style="color:${p.suggestedColor==='red'?'#f87171':'#86efac'}">${p.suggestedColor}</strong></div>`;
      } catch (e) {
        out.innerHTML = `<div class="stat-card" style="color:#fca5a5">${e.message}</div>`;
        log(e.message, "error");
      }
    });

    document.getElementById("clearPrediction").addEventListener("click", () => {
      currentPrediction = null;
      document.getElementById("predictionResult").textContent = "Prediction cleared.";
      log("Cleared current prediction.");
    });
  };

  const renderAddResult = () => {
    panel.innerHTML = `
      <h2>Add new result</h2>
      <div class="form-row">
        <label>Period number:</label>
        <input id="inputPeriod" type="text" placeholder="e.g. 105" />
      </div>
      <div class="form-row">
        <label>Result (0-9):</label>
        <input id="inputResult" type="number" min="0" max="9" placeholder="e.g. 7" />
      </div>
      <div style="display:flex;gap:8px;">
        <button id="doAdd" class="btn">Add</button>
        <button id="showLast" class="btn secondary">Show last 10</button>
      </div>
      <div id="addStatus" style="margin-top:12px"></div>
    `;

    document.getElementById("doAdd").addEventListener("click", () => {
      const period = document.getElementById("inputPeriod").value.trim();
      const resultVal = document.getElementById("inputResult").value.trim();
      const status = document.getElementById("addStatus");
      status.textContent = "";
      try {
        addNewResult(period, resultVal);
        saveData();
        status.innerHTML = `<div class="stat-card">Saved period ${period} result ${resultVal}</div>`;
      } catch (e) {
        status.innerHTML = `<div class="stat-card" style="color:#fca5a5">${e.message}</div>`;
        log(e.message, "error");
      }
    });

    document.getElementById("showLast").addEventListener("click", () => {
      const last = results.slice(-10).reverse();
      const rows = last.map(r => `${r.period}: ${r.result} (${r.type})`).join("\n");
      const status = document.getElementById("addStatus");
      status.innerHTML = `<pre class="hint">${rows || "No data"}</pre>`;
    });
  };

  const renderStats = () => {
    analyzeTrends();
    const recentTrends = trends.slice(-5).map((t, i) => {
      const cls = t === "big" ? "type-big" : (t === "small" ? "type-small" : "type-change");
      return `<div class="${cls}">Period ${results.length - 5 + i + 1 || i+1}: ${t}</div>`;
    }).join("");
    panel.innerHTML = `
      <h2>Game Statistics</h2>
      <div class="stats-grid">
        <div class="stat-card">Total results<br/><strong style="font-size:18px">${results.length}</strong></div>
        <div class="stat-card">Prediction accuracy<br/><strong style="font-size:18px">${getAccuracy()}%</strong></div>
        <div class="stat-card">Total predictions<br/><strong style="font-size:18px">${stats.total_predictions || 0}</strong></div>
        <div class="stat-card">Correct predictions<br/><strong style="font-size:18px">${stats.correct_predictions || 0}</strong></div>
      </div>
      <h3 style="margin-top:12px">Recent trends</h3>
      <div>${recentTrends || '<div class="hint">No trends yet</div>'}</div>
    `;
  };

  const renderHistory = () => {
    panel.innerHTML = `
      <h2>Historical Data (last 50)</h2>
      <table class="table" id="historyTable">
        <thead><tr><th>Period</th><th>Result</th><th>Type</th></tr></thead>
        <tbody></tbody>
      </table>
      <div style="margin-top:8px">
        <button id="exportJson" class="btn">Export JSON</button>
        <button id="importJson" class="btn secondary">Import JSON</button>
        <input type="file" id="fileInput" style="display:none" accept=".json" />
      </div>
    `;

    const tbody = panel.querySelector("tbody");
    const list = results.slice(-50).reverse();
    list.forEach(r => {
      const tr = document.createElement("tr");
      const td1 = document.createElement("td"); td1.textContent = r.period;
      const td2 = document.createElement("td"); td2.textContent = r.result;
      const td3 = document.createElement("td"); td3.textContent = r.type;
      td3.className = r.type === "big" ? "type-big" : "type-small";
      tr.appendChild(td1); tr.appendChild(td2); tr.appendChild(td3);
      tbody.appendChild(tr);
    });

    document.getElementById("exportJson").addEventListener("click", () => {
      const blob = new Blob([JSON.stringify(results, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = `color_prediction_data_${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
      log("Exported data JSON.");
    });

    document.getElementById("importJson").addEventListener("click", () => {
      document.getElementById("fileInput").click();
    });

    document.getElementById("fileInput").addEventListener("change", (ev) => {
      const f = ev.target.files[0];
      if (!f) return;
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const imported = JSON.parse(reader.result);
          if (!Array.isArray(imported)) throw new Error("Invalid format");
          results = imported.map(r => ({ period: String(r.period), result: Number(r.result), type: r.type }));
          saveData();
          analyzeTrends();
          renderHistory();
          log("Imported JSON data.");
        } catch (e) {
          log("Failed to import: " + e.message, "error");
        }
      };
      reader.readAsText(f);
    });
  };

  // Helpers
  const getAccuracy = () => {
    if (!stats.total_predictions || stats.total_predictions === 0) return 0;
    return Math.round((stats.correct_predictions || 0) * 100 / stats.total_predictions);
  };

  // Hook up menu buttons
  const mountButtons = () => {
    document.getElementById("btnPredict").addEventListener("click", renderPredict);
    document.getElementById("btnAddResult").addEventListener("click", renderAddResult);
    document.getElementById("btnConnect").addEventListener("click", async () => {
      await connectToServer();
    });
    document.getElementById("btnStats").addEventListener("click", renderStats);
    document.getElementById("btnHistory").addEventListener("click", renderHistory);
    document.getElementById("btnReset").addEventListener("click", () => {
      if (!confirm("Reset all stored game data? This cannot be undone.")) return;
      localStorage.removeItem(STORAGE_KEY);
      localStorage.removeItem(STATS_KEY);
      loadData();
      analyzeTrends();
      renderHome();
      log("Storage reset.", "info");
    });
  };

  // Initialize app
  const init = () => {
    log(`Initializing Color Prediction Game v${VERSION}...`);
    loadData();
    analyzeTrends();
    mountButtons();
    renderHome();
  };

  // Start
  init();

  // Expose for debugging in console (optional)
  window.ColorPredictionGame = {
    get results() { return results; },
    get stats() { return stats; },
    predict: (digits) => {
      try {
        const p = predictResult(digits);
        return p;
      } catch (e) {
        return { error: e.message };
      }
    },
    add: (period, value) => {
      try {
        addNewResult(period, value);
        return { ok: true };
      } catch (e) {
        return { error: e.message };
      }
    },
  };
})();
