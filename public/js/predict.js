/**
 * LoanLens — Assessment Form & Result Controller
 * Manages form validation, preset loading, honest model selector, and dynamic result rendering.
 */

(function () {
  'use strict';

  const form = document.getElementById("assessmentForm");
  const runBtn = document.getElementById("runAssessmentBtn");
  const resetBtn = document.getElementById("resetBtn");
  const loadingView = document.getElementById("loadingView");
  const resultView = document.getElementById("resultView");
  const benchmarkNotice = document.getElementById("benchmarkNotice");
  const benchmarkNoticeText = document.getElementById("benchmarkNoticeText");

  // Verified 16-feature Presets (No decorative Name field)
  const presets = {
    prime: {
      Age: 42,
      Education: "Master's",
      MaritalStatus: "Married",
      HasDependents: "No",
      EmploymentType: "Full-time",
      MonthsEmployed: 64,
      Income: 115000,
      CreditScore: 780,
      NumCreditLines: 3,
      DTIRatio: 0.22,
      LoanAmount: 25000,
      InterestRate: 6.8,
      LoanTerm: 36,
      LoanPurpose: "Home",
      HasMortgage: "Yes",
      HasCoSigner: "No",
    },
    moderate: {
      Age: 32,
      Education: "Bachelor's",
      MaritalStatus: "Single",
      HasDependents: "No",
      EmploymentType: "Full-time",
      MonthsEmployed: 24,
      Income: 62000,
      CreditScore: 650,
      NumCreditLines: 6,
      DTIRatio: 0.34,
      LoanAmount: 35000,
      InterestRate: 11.5,
      LoanTerm: 48,
      LoanPurpose: "Auto",
      HasMortgage: "No",
      HasCoSigner: "No",
    },
    subprime: {
      Age: 26,
      Education: "High School",
      MaritalStatus: "Single",
      HasDependents: "Yes",
      EmploymentType: "Part-time",
      MonthsEmployed: 8,
      Income: 35000,
      CreditScore: 540,
      NumCreditLines: 9,
      DTIRatio: 0.46,
      LoanAmount: 48000,
      InterestRate: 18.2,
      LoanTerm: 60,
      LoanPurpose: "Other",
      HasMortgage: "No",
      HasCoSigner: "No",
    }
  };

  // Quick Preset Loader
  document.querySelectorAll("[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const type = btn.getAttribute("data-preset");
      const data = presets[type];
      if (!data) return;

      Object.entries(data).forEach(([key, val]) => {
        const el = document.getElementById(key);
        if (el) el.value = val;
      });
    });
  });

  // Honest Model Selector Interaction
  const modelCards = document.querySelectorAll(".model-option-card");
  modelCards.forEach((card) => {
    card.addEventListener("click", () => {
      const modelId = card.getAttribute("data-model");

      if (modelId === "logistic_regression") {
        if (benchmarkNotice) benchmarkNotice.style.display = "none";
      } else {
        const names = {
          knn: "K-Nearest Neighbors (k=25)",
          naive_bayes: "Gaussian Naive Bayes",
          decision_tree: "Decision Tree (max_depth=6)"
        };
        const mName = names[modelId] || modelId;
        if (benchmarkNotice && benchmarkNoticeText) {
          benchmarkNoticeText.innerHTML = `<strong>${mName}</strong> is an evaluated benchmark baseline. Live scoring is designated to <strong>Logistic Regression</strong> for optimal recall (69.95%) and factor transparency.`;
          benchmarkNotice.style.display = "block";
        }
      }
    });
  });

  // Reset Button
  if (resetBtn && form) {
    resetBtn.addEventListener("click", () => {
      form.reset();
      if (resultView) resultView.style.display = "none";
      if (benchmarkNotice) benchmarkNotice.style.display = "none";
    });
  }

  // Submit Assessment Form
  if (runBtn && form) {
    runBtn.addEventListener("click", async () => {
      const formData = new FormData(form);
      const payload = {};

      for (const [key, val] of formData.entries()) {
        payload[key] = val;
      }

      // Convert numeric fields
      const numKeys = ["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio"];
      numKeys.forEach((k) => {
        if (payload[k] !== undefined && payload[k] !== "") {
          payload[k] = parseFloat(payload[k]);
        }
      });

      // Show Loading State
      if (loadingView) loadingView.style.display = "block";
      if (resultView) resultView.style.display = "none";
      loadingView.scrollIntoView({ behavior: "smooth", block: "center" });

      try {
        const response = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.message || `Assessment error: ${response.statusText}`);
        }

        const result = await response.json();
        renderResult(result);
      } catch (err) {
        if (loadingView) loadingView.style.display = "none";
        alert(`Prediction Error: ${err.message}`);
      }
    });
  }

  // Dynamic Result Renderer
  function renderResult(res) {
    if (loadingView) loadingView.style.display = "none";
    if (!resultView) return;

    resultView.style.display = "block";
    resultView.scrollIntoView({ behavior: "smooth", block: "start" });

    const probVal = typeof res.probability === "number" 
      ? (res.probability <= 1 ? res.probability * 100 : res.probability) 
      : 15;
    const probDisplay = probVal.toFixed(2);
    const riskScore = typeof res.risk_score === "number" ? res.risk_score : 85;
    const confidencePct = typeof res.confidence === "number" ? (res.confidence * 100).toFixed(1) : "85";

    const isLow = res.risk_level === "Low";
    const isMed = res.risk_level === "Medium";
    const badgeClass = isLow ? "badge-low" : isMed ? "badge-med" : "badge-high";
    const gaugeColor = isLow ? "var(--low)" : isMed ? "var(--med)" : "var(--high)";

    const circ = 2 * Math.PI * 54;
    const offset = circ * (1 - Math.min(100, Math.max(0, probVal)) / 100);

    const protective = (res.factors || []).filter(f => f.direction === "down" || (typeof f.raw_impact === "number" && f.raw_impact <= 0));
    const amplifiers = (res.factors || []).filter(f => f.direction === "up" || (typeof f.raw_impact === "number" && f.raw_impact > 0));

    resultView.innerHTML = `
      <div class="result-container">
        <!-- Top Score & Gauge Card -->
        <div class="card gauge-card">
          <div>
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
              <span class="badge ${badgeClass}" style="font-size:12px; font-weight:700; padding:6px 14px;">
                PREDICTED RISK LEVEL: ${(res.risk_level || "UNKNOWN").toUpperCase()}
              </span>
              <span style="font-size:12px; color:var(--muted);">
                Model: <b>${res.model_used || "Logistic Regression"}</b>
              </span>
            </div>

            <h2 style="font-size:26px; font-weight:700; margin:0 0 6px; color:var(--text);">
              ${res.prediction === 1 ? "Predicted Risk: Default" : "Predicted Risk: No Default"}
            </h2>

            <p style="font-size:13px; color:var(--muted); margin:0;">
              Model Classification: <strong style="color:var(--text);">${res.label}</strong> &middot; Default Probability: <strong>${probDisplay}%</strong> &middot; Confidence: <strong>${confidencePct}%</strong>
            </p>
          </div>

          <div class="gauge-visual">
            <svg class="gauge-svg" viewBox="0 0 130 130">
              <circle class="gauge-bg-circle" cx="65" cy="65" r="54" />
              <circle class="gauge-val-circle" cx="65" cy="65" r="54" 
                      stroke="${gaugeColor}"
                      stroke-dasharray="${circ}" 
                      stroke-dashoffset="${offset}" />
            </svg>
            <div class="gauge-center-text">
              <div class="gauge-score-number">${riskScore}</div>
              <div class="gauge-score-label">Risk Score</div>
            </div>
          </div>
        </div>

        <!-- Explainable Model Factors (Partitioned) -->
        <div class="card card-padded">
          <div style="margin-bottom:16px;">
            <h3 style="font-size:16px; font-weight:700; margin:0 0 4px;">Explainable Model Factors</h3>
            <p style="font-size:12px; color:var(--muted); margin:0;">
              Individual log-odds coefficient impact assessed by the Logistic Regression scoring engine.
            </p>
          </div>

          <div class="factors-layout">
            <!-- Protective Mitigants -->
            <div class="factor-column-card">
              <div class="factor-col-title" style="color:var(--low);">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                <span>Protective Mitigants (Decreases Default Risk)</span>
              </div>

              ${protective.length > 0 ? protective.map(f => `
                <div class="factor-item">
                  <div class="factor-top-row">
                    <span class="factor-name">${f.label || f.feature}</span>
                    <span class="factor-impact-badge" style="background:var(--low-soft); color:var(--low);">
                      ${f.raw_impact ? f.raw_impact.toFixed(3) : f.impact}% log-odds
                    </span>
                  </div>
                  <div class="impact-bar-bg">
                    <div class="impact-bar-fill" style="width:${Math.min(100, Math.abs(f.impact || 40))}%; background:var(--low);"></div>
                  </div>
                  <div style="font-size:11px; color:var(--muted);">${f.desc}</div>
                </div>
              `).join("") : '<div style="font-size:12px; color:var(--muted); padding:10px 0;">No strong protective factors identified.</div>'}
            </div>

            <!-- Risk Amplifiers -->
            <div class="factor-column-card">
              <div class="factor-col-title" style="color:var(--high);">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>
                <span>Risk Amplifiers (Increases Default Risk)</span>
              </div>

              ${amplifiers.length > 0 ? amplifiers.map(f => `
                <div class="factor-item">
                  <div class="factor-top-row">
                    <span class="factor-name">${f.label || f.feature}</span>
                    <span class="factor-impact-badge" style="background:var(--high-soft); color:var(--high);">
                      ${f.raw_impact ? '+' + f.raw_impact.toFixed(3) : '+' + f.impact}% log-odds
                    </span>
                  </div>
                  <div class="impact-bar-bg">
                    <div class="impact-bar-fill" style="width:${Math.min(100, Math.abs(f.impact || 40))}%; background:var(--high);"></div>
                  </div>
                  <div style="font-size:11px; color:var(--muted);">${f.desc}</div>
                </div>
              `).join("") : '<div style="font-size:12px; color:var(--muted); padding:10px 0;">No strong risk-amplifying drivers identified.</div>'}
            </div>
          </div>
        </div>

        <!-- Model Result Interpretation -->
        ${res.recommendation ? `
          <div class="card card-padded" style="background:var(--brand-soft); border-color:color-mix(in srgb, var(--brand) 25%, transparent);">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--brand)" stroke-width="2.2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
              <span style="font-size:14px; font-weight:700; color:var(--brand);">${res.recommendation.action}</span>
            </div>
            <ul style="margin:0; padding-left:20px; font-size:13px; color:var(--text); line-height:1.7;">
              ${(res.recommendation.points || []).map(p => `<li>${p}</li>`).join("")}
            </ul>
            <div style="margin-top:12px; padding-top:10px; border-top:1px dashed color-mix(in srgb, var(--brand) 20%, transparent); font-size:11px; color:var(--muted);">
              <em>Notice: This evaluation is an educational machine learning model simulation based on public benchmark data and is not an actual commercial lending or underwriting decision.</em>
            </div>
          </div>
        ` : ''}
      </div>
    `;
  }
})();
