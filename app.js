// Color Prediction Game — browser version with AI learning feature
// Persists to localStorage. Converted from Termux bash script and extended with a lightweight logistic regression model.

(() => {
  // Config
  const SERVER_URL = "https://www.tashanwin27.com/#/main";
  const GAME_UID = "1969524";
  const VERSION = "2.0";
  const STORAGE_KEY = "color_prediction_game:data";
  const STATS_KEY = "color_prediction_game:stats";
  const MODEL_KEY = "color_prediction_game:model";

  // DOM
  const panel = document.getElementById("panel");
  const logArea = document.getElementById("logArea");
  document.getElementById("serverUrl").href = SERVER_URL;
  document.getElementById("gameUid").textContent = GAME_UID;

  // Game state
  let results = []; // {period: string, result: number, type: "big"|"small"}
  let trends = [];
  let currentPrediction = null;
  let stats = { total_predictions: 0, correct_predictions: 0 };
  let model = null; // {weights: [], bias: number, featureCount: n, trainedAt: timestamp, trainAcc}

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
    if (model) localStorage.setItem(MODEL_KEY, JSON.stringify(model));
  };

  const loadData = () => {
    const raw = localStorage.getItem(STORAGE_KEY);
    const rawStats = localStorage.getItem(STATS_KEY);
    const rawModel = localStorage.getItem(MODEL_KEY);
    if (raw) {
      try { results = JSON.parse(raw); } catch (e) { results = []; log("Corrupt saved data — resetting.", "error"); }
    } else {
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
    if (rawModel) {
      try { model = JSON.parse(rawModel); } catch { model = null; }
      if (model) log("Loaded AI model from storage. Train accuracy: " + (model.trainAcc || 0) + "%");
    }
  };

  // Trend analysis
  const analyzeTrends = () => {
    trends = [];
    for (let i = 0; i < results.length; i++) {
      const r = results[i].type;
      if (trends.length >= 1) {
        const last = trends[trends.length - 1];
        trends.push(last === r ? r : "change");
      } else {
        trends.push(r);
      }
    }
  };

  // Feature extraction for ML
  // For training we build examples where label is current.type and features are derived from previous 3 results:
  // features: normalized prev3_result, prev2_result, prev1_result, is_big(prev1), is_big(prev2), is_big(prev3), avg_prev3, count_big_prev3
  const buildTrainingData = () => {
    const X = [];
    const Y = [];
    for (let i = 3; i < results.length; i++) {
      const prev3 = results[i - 3].result;
      const prev2 = results[i - 2].result;
      const prev1 = results[i - 1].result;
      const cur = results[i].type === "big" ? 1 : 0;
      const isBig1 = results[i - 1].type === "big" ? 1 : 0;
      const isBig2 = results[i - 2].type === "big" ? 1 : 0;
      const isBig3 = results[i - 3].type === "big" ? 1 : 0;
      const avg = (prev1 + prev2 + prev3) / 27; // normalize by max 9*3=27
      const countBig = (isBig1 + isBig2 + isBig3) / 3;
      const feat = [
        prev3 / 9,
        prev2 / 9,
        prev1 / 9,
        isBig3,
        isBig2,
        isBig1,
        avg,
        countBig
      ];
      X.push(feat);
      Y.push(cur);
    }
    return { X, Y };
  };

  // Sigmoid
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));

  // Logistic regression trainer (SGD)
  const trainLogisticRegression = (X, Y, options = {}) => {
    const lr = options.lr || 0.2;
    const epochs = options.epochs || 300;
    const batchSize = options.batchSize || 16;
    const featureCount = X[0].length;
    let weights = new Array(featureCount).fill(0).map(() => (Math.random() - 0.5) * 0.1);
    let bias = 0;
    for (let epoch = 0; epoch < epochs; epoch++) {
      // shuffle indices
      const idx = Array.from({ length: X.length }, (_, i) => i);
      for (let i = idx.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [idx[i], idx[j]] = [idx[j], idx[i]];
      }
      for (let start = 0; start < X.length; start += batchSize) {
        const end = Math.min(start + batchSize, X.length);
        const gradW = new Array(featureCount).fill(0);
        let gradB = 0;
        for (let k = start; k < end; k++) {
          const i = idx[k];
          const xi = X[i];
          const yi = Y[i];
          const z = weights.reduce((s, w, j) => s + w * xi[j], bias);
          const pred = sigmoid(z);
          const err = pred - yi;
          for (let j = 0; j < featureCount; j++) gradW[j] += err * xi[j];
          gradB += err;
        }
        // update
        const scale = 1 / (end - start);
        for (let j = 0; j < featureCount; j++) weights[j] -= lr * gradW[j] * scale;
        bias -= lr * gradB * scale;
      }
    }
    return { weights, bias, featureCount };
  };

  const evaluateModel = (modelObj, X, Y) => {
    if (!modelObj || !X || X.length === 0) return { accuracy: 0 };
    let correct = 0;
    for (let i = 0; i < X.length; i++) {
      const z = modelObj.weights.reduce((s, w, j) => s + w * X[i][j], modelObj.bias);
      const p = sigmoid(z);
      const pred = p >= 0.5 ? 1 : 0;
      if (pred === Y[i]) correct++;
    }
    return { accuracy: Math.round((correct / X.length) * 100) };
  };

  // Train model with train/test split and save
  const trainModel = (opts = {}) => {
    const { X, Y } = buildTrainingData();
    if (X.length < 6) {
      log("Not enough data to train AI model. Add more historical results (recommended >= 20).", "error");
      return null;
    }
    // Train/test split 80/20
    const total = X.length;
    const cutoff = Math.max(1, Math.floor(total * 0.8));
    // shuffle paired arrays
    const pairs = X.map((x, i) => ({ x, y: Y[i] }));
    for (let i = pairs.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [pairs[i], pairs[j]] = [pairs[j], pairs[i]];
    }
    const train = pairs.slice(0, cutoff);
    const test = pairs.slice(cutoff);
    const trainX = train.map(p => p.x);
    const trainY = train.map(p => p.y);
    const testX = test.map(p => p.x);
    const testY = test.map(p => p.y);

    const modelObj = trainLogisticRegression(trainX, trainY, opts);
    const trainEval = evaluateModel(modelObj, trainX, trainY);
    const testEval = evaluateModel(modelObj, testX, testY);

    model = {
      weights: modelObj.weights,
      bias: modelObj.bias,
      featureCount: modelObj.featureCount,
      trainedAt: Date.now(),
      trainAcc: trainEval.accuracy,
      testAcc: testEval.accuracy
    };
    saveData();
    log(`Model trained. Train acc: ${trainEval.accuracy}%, Test acc: ${testEval.accuracy}%`, "success");
    return model;
  };

  // Predict using model: build features using last 3 historical results and lastDigits (if provided)
  const predictWithModel = (lastDigits = null) => {
    if (!model) throw new Error("No trained model available. Train the model first.");
    if (results.length < 3) throw new Error("Need at least 3 historical results to build features.");
    const prev3 = results[results.length - 3].result;
    const prev2 = results[results.length - 2].result;
    const prev1 = results[results.length - 1].result;
    const isBig1 = results[results.length - 1].type === "big" ? 1 : 0;
    const isBig2 = results[results.length - 2].type === "big" ? 1 : 0;
    const isBig3 = results[results.length - 3].type === "big" ? 1 : 0;
    const avg = (prev1 + prev2 + prev3) / 27;
    const countBig = (isBig1 + isBig2 + isBig3) / 3;

    // base features as in training
    const feat = [
      prev3 / 9,
      prev2 / 9,
      prev1 / 9,
      isBig3,
      isBig2,
      isBig1,
      avg,
      countBig
    ];

    // predict
    const z = model.weights.reduce((s, w, j) => s + w * feat[j], model.bias);
    const p = sigmoid(z);
    const predType = p >= 0.5 ? "big" : "small";
    // color suggestion fallback: follow previous simple rule using lastDigits if provided, else use parity of prev1
    let suggestedColor = "green";
    if (lastDigits && /^\d{3}$/.test(lastDigits)) {
      const num = parseInt(lastDigits, 10);
      suggestedColor = num % 3 === 0 ? "red" : "green";
    } else {
      suggestedColor = (prev1 % 3 === 0) ? "red" : "green";
    }

    currentPrediction = { type: predType, suggestedColor, confidence: Math.round(p * 100) };
    return currentPrediction;
  };

  // Predict without model (existing heuristic)
  const predictHeuristic = (lastDigits) => {
    if (!lastDigits || !/^\d{3}$/.test(lastDigits)) throw new Error("Please enter exactly 3 digits");
    const num = parseInt(lastDigits, 10);
    const lastTen = results.slice(-10);
    let big_count = 0, small_count = 0;
    lastTen.forEach(r => (r.type === "big" ? big_count++ : small_count++));
    let predictionType = "";
    if (results.length >= 5) {
      const lastFive = results.slice(-5);
      let same_count = 0;
      for (let i = 0; i < 4; i++) if (lastFive[i].type === lastFive[i + 1].type) same_count++;
      if (same_count >= 3) {
        const lastType = lastFive[4].type;
        predictionType = lastType === "big" ? "small" : "big";
      } else {
        predictionType = big_count > small_count ? "big" : "small";
      }
    } else {
      predictionType = num % 2 === 0 ? "small" : "big";
    }
    const suggestedColor = num % 3 === 0 ? "red" : "green";
    currentPrediction = { type: predictionType, suggestedColor, confidence: null };
    return currentPrediction;
  };

  // Add new result and optionally auto-update stats and model
  const addNewResult = (period, resultValue, autoTrain = false) => {
    if (!period || period.toString().trim() === "") throw new Error("Period is required");
    if (!/^[0-9]$/.test(String(resultValue))) throw new Error("Result must be single digit 0-9");
    const resultNum = parseInt(resultValue, 10);
    const type = resultNum >= 5 ? "big" : "small";
    results.push({ period: String(period), result: resultNum, type });
    saveData();
    analyzeTrends();

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

    if (autoTrain) {
      try {
        trainModel({ epochs: 250, lr: 0.15 });
      } catch (e) {
        log("Auto-train failed: " + e.message, "error");
      }
    }
  };

  // Simulated server connection
  const connectToServer = async () => {
    log("Connecting to server...", "info");
    log(`Server: ${SERVER_URL}`, "info");
    log(`Game UID: ${GAME_UID}`, "info");
    await new Promise(r => setTimeout(r, 800));
    log("Connected successfully. Fetched latest results (simulated).", "success");
  };

  // UI rendering
  const renderHome = () => {
    panel.innerHTML = `
      <h2>Welcome</h2>
      <p class="hint">Use the buttons on the left to predict, add results, train the AI model, or view stats and history.</p>
      <div style="margin-top:12px">
        <span class="pill">Total results: ${results.length}</span>
        <span class="pill">Accuracy: ${getAccuracy()}%</span>
        <span class="pill">Total predictions: ${stats.total_predictions || 0}</span>
      </div>
      <div style="margin-top:12px" class="hint">AI model: ${model ? `trained (test acc ${model.testAcc || model.testAcc || model.testAcc || model.testAcc || model.testAcc || model.trainAcc || 0}%)` : 'not trained'}</div>
    `;
  };

  const renderPredict = () => {
    panel.innerHTML = `
      <h2>Predict next result</h2>
      <div class="form-row">
        <label>Enter last 3 digits:</label>
        <input id="inputDigits" type="text" maxlength="3" placeholder="e.g. 123" />
      </div>
      <div class="form-row">
        <label>Prediction method:</label>
        <select id="predMethod">
          <option value="heuristic">Heuristic (original)</option>
          <option value="ai">AI Model (if trained)</option>
        </select>
      </div>
      <div style="display:flex;gap:8px;">
        <button id="doPredict" class="btn">Predict</button>
        <button id="clearPrediction" class="btn secondary">Clear</button>
      </div>
      <div id="predictionResult" style="margin-top:12px"></div>
      <div class="hint">Tip: Train the AI model after adding historical data to improve predictions. AI model uses previous 3 results as features.</div>
    `;

    document.getElementById("doPredict").addEventListener("click", () => {
      const val = document.getElementById("inputDigits").value.trim();
      const method = document.getElementById("predMethod").value;
      const out = document.getElementById("predictionResult");
      out.textContent = "";
      try {
        if (method === "ai") {
          if (!model) {
            out.innerHTML = `<div class="stat-card" style="color:#fca5a5">No trained AI model found. Train the model first.</div>`;
            return;
          }
          const p = predictWithModel(val);
          out.innerHTML = `<div class="stat-card">AI Prediction: <strong>${p.type}</strong> &nbsp;&nbsp; Color: <strong style="color:${p.suggestedColor==='red'?'#f87171':'#86efac'}">${p.suggestedColor}</strong> &nbsp;&nbsp; Confidence: <strong>${p.confidence}%</strong></div>`;
        } else {
          const p = predictHeuristic(val);
          out.innerHTML = `<div class="stat-card">Heuristic Prediction: <strong>${p.type}</strong> &nbsp;&nbsp; Color suggestion: <strong style="color:${p.suggestedColor==='red'?'#f87171':'#86efac'}">${p.suggestedColor}</strong></div>`;
        }
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
      <div class="form-row">
        <label>Auto-train after add:</label>
        <select id="autoTrain"><option value="no">No</option><option value="yes">Yes</option></select>
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
      const auto = document.getElementById("autoTrain").value === "yes";
      const status = document.getElementById("addStatus");
      status.textContent = "";
      try {
        addNewResult(period, resultVal, auto);
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

  const renderTrainModel = () => {
    panel.innerHTML = `
      <h2>Train AI Model</h2>
      <div class="form-row">
        <label>Epochs:</label>
        <input id="inputEpochs" type="number" min="10" max="2000" value="300" />
      </div>
      <div class="form-row">
        <label>Learning rate:</label>
        <input id="inputLR" type="text" value="0.15" />
      </div>
      <div style="display:flex;gap:8px;">
        <button id="doTrain" class="btn">Train</button>
        <button id="btnShowModel" class="btn secondary">Show model</button>
      </div>
      <div id="trainResult" style="margin-top:12px"></div>
      <div class="hint">Training uses previous 3 results as features. More historical data -> better model. Recommended >= 20 entries.</div>
    `;

    document.getElementById("doTrain").addEventListener("click", () => {
      const epochs = parseInt(document.getElementById("inputEpochs").value, 10) || 300;
      const lr = parseFloat(document.getElementById("inputLR").value) || 0.15;
      try {
        const m = trainModel({ epochs, lr, batchSize: 16 });
        if (m) {
          document.getElementById("trainResult").innerHTML = `<div class="stat-card">Trained model. Train acc: ${m.trainAcc}%, Test acc: ${m.testAcc}%</div>`;
        } else {
          document.getElementById("trainResult").innerHTML = `<div class="stat-card" style="color:#fca5a5">Training failed.</div>`;
        }
      } catch (e) {
        document.getElementById("trainResult").innerHTML = `<div class="stat-card" style="color:#fca5a5">${e.message}</div>`;
        log(e.message, "error");
      }
    });

    document.getElementById("btnShowModel").addEventListener("click", () => {
      if (!model) {
        document.getElementById("trainResult").innerHTML = `<div class="hint">No model available.</div>`;
        return;
      }
      const w = model.weights.map((v, i) => `w[${i}]=${v.toFixed(4)}`).join(", ");
      document.getElementById("trainResult").innerHTML = `<div class="stat-card">Model weights: <pre style="white-space:pre-wrap">${w}\nbias=${model.bias.toFixed(4)}\ntrainAcc=${model.trainAcc}%, testAcc=${model.testAcc || 0}%</pre></div>`;
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
      <h3 style="margin-top:12px">AI Model</h3>
      <div class="hint">${model ? `Trained at ${new Date(model.trainedAt).toLocaleString()} — Train ${model.trainAcc}%, Test ${model.testAcc || 0}%` : 'No model trained yet'}</div>
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
    document.getElementById("btnTrainModel").addEventListener("click", renderTrainModel);
    document.getElementById("btnConnect").addEventListener("click", async () => { await connectToServer(); });
    document.getElementById("btnStats").addEventListener("click", renderStats);
    document.getElementById("btnHistory").addEventListener("click", renderHistory);
    document.getElementById("btnReset").addEventListener("click", () => {
      if (!confirm("Reset all stored game data? This cannot be undone.")) return;
      localStorage.removeItem(STORAGE_KEY);
      localStorage.removeItem(STATS_KEY);
      localStorage.removeItem(MODEL_KEY);
      loadData();
      analyzeTrends();
      renderHome();
      log("Storage reset.", "info");
    });
  };

  // Initialize app
  const init = () => {
    log(`Initializing Color Prediction Game v${VERSION} (with AI)...`);
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
    get model() { return model; },
    predictHeuristic: (digits) => {
      try { return predictHeuristic(digits); } catch (e) { return { error: e.message }; }
    },
    predictAI: (digits) => {
      try { return predictWithModel(digits); } catch (e) { return { error: e.message }; }
    },
    trainModel: (opts) => {
      try { return trainModel(opts || {}); } catch (e) { return { error: e.message }; }
    },
    add: (period, value) => {
      try { addNewResult(period, value); return { ok: true }; } catch (e) { return { error: e.message }; }
    },
  };
})();
