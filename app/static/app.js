/* ══════════════════════════════
   ChurnML — app.js
══════════════════════════════ */

const riskColors = {
  Low:    { bg: "#f0fdf4", border: "#86efac", text: "#166534", badge: "#22c55e" },
  Medium: { bg: "#fffbeb", border: "#fcd34d", text: "#92400e", badge: "#f59e0b" },
  High:   { bg: "#fef2f2", border: "#fca5a5", text: "#991b1b", badge: "#ef4444" },
};

const $ = id => document.getElementById(id);

/* ── Tab Switching ── */
function switchTab(name, el) {
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));

  $("panel-" + name).classList.add("active");
  $("tab-" + name).classList.add("active");

  if (el && el.classList.contains("nav-item")) el.classList.add("active");
}

/* ── Error Helpers ── */
function showError(msg) {
  const b = $("errorBox");
  b.textContent = "⚠  " + msg;
  b.style.display = "block";
}

function hideError() {
  $("errorBox").style.display = "none";
}

/* ── Loading State ── */
function setLoading(on) {
  const btn = $("predictBtn");
  btn.disabled = on;
  btn.innerHTML = on
    ? '<div class="spinner"></div> Analyzing...'
    : `<svg width="13" height="13" fill="none" viewBox="0 0 24 24">
         <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" stroke="white" stroke-width="2" stroke-linecap="round"/>
       </svg> Predict Churn`;
}

/* ── Predict ── */
async function predict() {
  hideError();
  $("resultCard").style.display = "none";
  $("infoCard").style.display   = "none";

  const get = id => $(id).value.trim();

  // Validate required numeric fields
  for (const f of ["CreditScore", "Age", "Balance", "EstimatedSalary"]) {
    if (!get(f)) {
      showError("Please fill in " + f + ".");
      return;
    }
  }

  const payload = {
    CreditScore:     parseInt(get("CreditScore")),
    Age:             parseInt(get("Age")),
    Balance:         parseFloat(get("Balance")),
    EstimatedSalary: parseFloat(get("EstimatedSalary")),
    NumOfProducts:   parseInt(get("NumOfProducts")),
    IsActiveMember:  parseInt(get("IsActiveMember")),
    Geography:       get("Geography"),
    Gender:          get("Gender"),
  };

  setLoading(true);
  try {
    const res = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Server error");
    }

    showResult(await res.json(), payload);
  } catch (e) {
    showError(e.message || "Could not connect to API.");
  } finally {
    setLoading(false);
  }
}

/* ── Show Result ── */
function showResult(data, p) {
  const pct  = Math.round(data.churn_probability * 100);
  const r    = riskColors[data.risk_level];
  const card = $("resultCard");

  // Style result card
  card.style.background  = r.bg;
  card.style.borderColor = r.border;

  // Text content
  ["resultLabel", "resultVerdict", "resultMsg", "probLabel", "probValue"].forEach(
    id => $(id).style.color = r.text
  );
  $("resultLabel").textContent   = "Prediction Result";
  $("resultVerdict").textContent = data.churn ? "Will Churn" : "Will Stay";
  $("resultMsg").textContent     = data.message;

  // Badge
  $("riskBadge").style.background = r.badge;
  $("riskBadge").textContent       = data.risk_level + " Risk";

  // Probability
  $("probValue").textContent = pct + "%";

  // Animate bar
  const bar = $("barFill");
  bar.style.background = r.badge;
  bar.style.width      = "0%";
  card.style.display   = "block";
  setTimeout(() => { bar.style.width = pct + "%"; }, 50);

  // Populate submitted values
  const fmt = n => Number(n).toLocaleString();
  $("iCS").textContent     = p.CreditScore;
  $("iAge").textContent    = p.Age;
  $("iBal").textContent    = "$" + fmt(p.Balance);
  $("iSal").textContent    = "$" + fmt(p.EstimatedSalary);
  $("iProd").textContent   = p.NumOfProducts;
  $("iActive").textContent = p.IsActiveMember ? "Yes" : "No";
  $("iGeo").textContent    = p.Geography;
  $("iGen").textContent    = p.Gender;
  $("infoCard").style.display = "block";
}

/* ── Reset Form ── */
function resetForm() {
  ["CreditScore", "Age", "Balance", "EstimatedSalary"].forEach(id => $(id).value = "");
  $("NumOfProducts").value  = "1";
  $("IsActiveMember").value = "1";
  $("Geography").value      = "France";
  $("Gender").value         = "Male";
  $("resultCard").style.display = "none";
  $("infoCard").style.display   = "none";
  hideError();
}
