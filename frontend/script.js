const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const btnSend = document.getElementById("btn-send");
// --- UPLOAD STATE TRACKERS ---
const fileInput = document.getElementById("file-upload");
const btnAttach = document.getElementById("btn-attach");
const previewContainer = document.getElementById("file-preview-container");
const previewName = document.getElementById("file-preview-name");
const btnRemoveFile = document.getElementById("btn-remove-file");
// Replace this with your actual URL from Hugging Face
const BACKEND_URL = "https://ankit1831-healbridge.hf.space";

let currentUploadedFile = null;

// 🟢 QUALITY GATE LOGIC
const qualityModal = document.getElementById("quality-gate-modal");
const btnCancelUpload = document.getElementById("btn-cancel-upload");
const btnAcceptUpload = document.getElementById("btn-accept-upload");

btnAttach.addEventListener("click", () => {
  qualityModal.style.display = "flex";
});

btnCancelUpload.addEventListener("click", () => {
  qualityModal.style.display = "none";
});

btnAcceptUpload.addEventListener("click", () => {
  qualityModal.style.display = "none";
  fileInput.click(); // Only open file picker AFTER they agree
});

// Handle file selection
fileInput.addEventListener("change", function () {
  if (this.files && this.files[0]) {
    const file = this.files[0];

    // Safety check: Limit size to 5MB
    if (file.size > 5 * 1024 * 1024) {
      alert("File is too large. Please upload a file smaller than 5MB.");
      this.value = "";
      return;
    }

    currentUploadedFile = file;
    previewName.innerText = file.name;

    // Change icon based on type
    const icon = document.getElementById("file-preview-icon");
    if (file.type.startsWith("image/")) icon.innerText = "🖼️";
    else icon.innerText = "📄";

    previewContainer.style.display = "flex";
  }
});

// Handle file removal
btnRemoveFile.addEventListener("click", () => {
  fileInput.value = "";
  currentUploadedFile = null;
  previewContainer.style.display = "none";
});

// --- STATE TRACKERS ---
let diagnosticQuestionCount = 0;
let diagnosticQuestionLimit = 6;
let fullTranscript = "";
let apiChatHistory = [];
let patientAge = null;
let patientSex = null;
let isTriageComplete = false;
let currentDiagnosis = "";
let followUpHistory = "";
let isReceptionistPassed = false;
let currentReportData = null; // 🟢 NEW: Stores data for the PDF Export

document.getElementById("user-input").placeholder =
  "Type your symptoms here (Type 'predict' when ready)...";

// --- MODAL LOGIC ---
const modal = document.getElementById("triage-modal");
document.querySelectorAll(".launch-modal-btn").forEach((btn) => {
  btn.addEventListener("click", (e) => {
    e.preventDefault();
    modal.classList.remove("hidden");
  });
});
document
  .getElementById("close-modal")
  .addEventListener("click", () => modal.classList.add("hidden"));

// ==========================================
// 🟢 NEW: SESSION PERSISTENCE LOGIC 🟢
// ==========================================
function saveSession() {
  const sessionData = {
    diagnosticQuestionCount,
    fullTranscript,
    apiChatHistory,
    patientAge,
    patientSex,
    isTriageComplete,
    currentDiagnosis,
    followUpHistory,
    isReceptionistPassed,
    currentReportData,
    chatHTML: chatBox.innerHTML,
  };
  localStorage.setItem("healBridgeSession", JSON.stringify(sessionData));
}

function loadSession() {
  const saved = localStorage.getItem("healBridgeSession");
  if (saved) {
    const data = JSON.parse(saved);

    // 1. Restore the Javascript Memory
    diagnosticQuestionCount = data.diagnosticQuestionCount || 0;
    fullTranscript = data.fullTranscript || "";
    apiChatHistory = data.apiChatHistory || [];
    patientAge = data.patientAge || null;
    patientSex = data.patientSex || null;
    isTriageComplete = data.isTriageComplete || false;
    currentDiagnosis = data.currentDiagnosis || "";
    followUpHistory = data.followUpHistory || "";
    isReceptionistPassed = data.isReceptionistPassed || false;
    currentReportData = data.currentReportData || null;
    // 2. Restore the Chat HTML
    if (data.chatHTML) {
      chatBox.innerHTML = data.chatHTML;
      chatBox.scrollTop = chatBox.scrollHeight;
    }

    // 3. Restore the Sidebar UI
    if (patientAge && patientSex) {
      document.getElementById("ui-age").innerText = patientAge;
      document.getElementById("ui-sex").innerText = patientSex;
    }

    // 4. Restore Button & Input States
    if (isReceptionistPassed && !isTriageComplete) {
      document
        .getElementById("btn-predict")
        .style.setProperty("display", "inline-block", "important");
      document.getElementById("user-input").placeholder =
        "Describe your symptoms...";
    } else if (isTriageComplete) {
      document.getElementById("user-input").placeholder =
        "Ask the Chief Medical Officer a question...";
    }
  }
}

// 🟢 Instantly load the session the moment the script runs
loadSession();
// ==========================================
// --- UI FUNCTIONS ---
function addMessage(text, sender) {
  const msgDiv = document.createElement("div");
  msgDiv.classList.add(
    "message",
    sender === "user" ? "user-message" : "ai-message",
  );
  const contentDiv = document.createElement("div");
  contentDiv.classList.add("message-content");

  // 🟢 FIX 2: MARKDOWN PARSER (Handles Bold, Italic, and Newlines)
  let formattedText = text.replace(/\n/g, "<br>");
  formattedText = formattedText.replace(
    /\*\*(.*?)\*\*/g,
    "<strong>$1</strong>",
  ); // Bold
  formattedText = formattedText.replace(/\*(.*?)\*/g, "<em>$1</em>"); // Italic

  contentDiv.innerHTML = formattedText;
  if (sender === "ai") {
    const speakerBtn = document.createElement("button");
    speakerBtn.className = "speaker-btn";
    speakerBtn.title = "Listen to answer";
    speakerBtn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path><path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path></svg>`;

    // 👇 ADD THIS ONE LINE RIGHT HERE 👇
    speakerBtn.style.cssText =
      "display: inline-block; margin-top: 10px; margin-left: 8px; cursor: pointer; color: #94a3b8;";

    speakerBtn.onclick = function () {
      window.speakText(text, this);
    };
    contentDiv.appendChild(speakerBtn); // <-- Notice this is contentDiv, perfectly correct.
  }

  msgDiv.appendChild(contentDiv);
  chatBox.appendChild(msgDiv);
  chatBox.scrollTop = chatBox.scrollHeight;

  saveSession();
}

function addHTMLMessage(htmlContent) {
  const msgDiv = document.createElement("div");
  msgDiv.classList.add("message", "ai-message");
  const contentDiv = document.createElement("div");
  contentDiv.classList.add("message-content");
  contentDiv.style.width = "100%";
  contentDiv.innerHTML = htmlContent;
  msgDiv.appendChild(contentDiv);
  chatBox.appendChild(msgDiv);
  chatBox.scrollTop = chatBox.scrollHeight;

  saveSession(); // 🟢 NEW: Auto-save on dynamic UI elements (like the PDF button)
}

// --- UPDATED LOADING & UI LOCKING ---
function addLoadingBubble(isPredicting = false) {
  const id = "loading-" + Date.now();
  const msgDiv = document.createElement("div");
  msgDiv.id = id;
  msgDiv.classList.add("message", "ai-message");

  // 1. HARD LOCK THE INPUT AREA
  const chatInput = document.querySelector("input[type='text'], #chat-input");
  const bottomBtns = document.querySelectorAll("button"); // Grabs all buttons (Send, Predict, etc.)
  if (chatInput) chatInput.disabled = true;
  bottomBtns.forEach((btn) => (btn.disabled = true)); // Lock everything down

  // 2. STANDARD CHAT: Fast, simple loading
  if (!isPredicting) {
    msgDiv.innerHTML = `
      <div class="message-content" style="background: transparent; color: #94a3b8; font-style: italic; padding: 8px 16px;">
        Analyzing...
      </div>
    `;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return id;
  }

  // 3. PREDICT MODE: Sleek Fading Text (Gemini/ChatGPT Style)
  msgDiv.innerHTML = `
    <div class="message-content" style="background: transparent; padding: 8px 16px;">
      <span id="thinking-text-${id}" style="color: #0ea5e9; font-weight: 500; font-style: italic; transition: opacity 0.4s ease-in-out; opacity: 1;">
        Reviewing patient transcript...
      </span>
    </div>
  `;
  chatBox.appendChild(msgDiv);
  chatBox.scrollTop = chatBox.scrollHeight;

  const steps = [
    "Reviewing patient transcript...",
    "Extracting clinical entities...",
    "Querying medical vector database...",
    "Generating differential diagnosis...",
  ];
  let stepIdx = 0;

  // The fading loop
  const interval = setInterval(() => {
    const textElement = document.getElementById(`thinking-text-${id}`);
    if (textElement) {
      // Fade out
      textElement.style.opacity = 0;

      // Wait 400ms for the fade to finish, change text, and fade back in
      setTimeout(() => {
        stepIdx = (stepIdx + 1) % steps.length;
        if (textElement) {
          textElement.innerText = steps[stepIdx];
          textElement.style.opacity = 1;
        }
      }, 400);
    } else {
      clearInterval(interval);
    }
  }, 1800); // Stays on screen for 1.8 seconds before swapping

  msgDiv.dataset.intervalId = interval;
  return id;
}

function removeLoadingBubble(id) {
  const el = document.getElementById(id);
  if (el) {
    if (el.dataset.intervalId) clearInterval(el.dataset.intervalId);
    el.remove();
  }

  // RE-ENABLE THE INPUT AREA
  const chatInput = document.querySelector("input[type='text'], #chat-input");
  const bottomBtns = document.querySelectorAll("button");
  if (chatInput) {
    chatInput.disabled = false;
    chatInput.focus();
  }
  bottomBtns.forEach((btn) => (btn.disabled = false));
}

// --- RESET LOGIC ---
document.getElementById("btn-restart").addEventListener("click", () => {
  chatBox.innerHTML = `
        <div class="message ai-message">
            <div class="message-content">
                Hello. I am the Heal Bridge Triage Assistant. Could you please tell me your age, biological sex, and what brings you in today?
            </div>
        </div>
    `;

  diagnosticQuestionCount = 0;
  diagnosticQuestionLimit = 6;
  fullTranscript = "";
  apiChatHistory = [];
  isTriageComplete = false;
  currentDiagnosis = "";
  followUpHistory = "";
  patientAge = null;
  patientSex = null;
  isReceptionistPassed = false;
  document.getElementById("ui-age").innerText = "Pending";
  document.getElementById("ui-sex").innerText = "Pending";
  document.getElementById("user-input").value = "";
  document.getElementById("btn-predict").style.display = "none";
  document.getElementById("user-input").placeholder =
    "Type your message here...";
  localStorage.removeItem("healBridgeSession"); // 🟢 NEW: Wipe the memory clean!
});

// --- PREDICT LOGIC ---
window.runDiagnosticPrediction = async function () {
  if (!isReceptionistPassed) {
    addMessage("Please complete the initial details first(Age,Sex).", "ai");
    return;
  }
  // 🟢 DISABLE BUTTONS TO PREVENT DOUBLE-CLICK CRASHES 🟢
  document.getElementById("btn-predict").disabled = true;
  document.getElementById("btn-send").disabled = true;
  userInput.disabled = true;

  const loadingId = addLoadingBubble(true);
  document.getElementById("btn-predict").style.display = "none";

  try {
    const response = await fetch(`${BACKEND_URL}/api/triage`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        age: patientAge,
        sex: patientSex,
        transcript: fullTranscript,
      }),
    });

    if (!response.ok) {
      const errData = await response.json();
      throw new Error(errData.detail || "Server Error");
    }

    const data = await response.json();
    removeLoadingBubble(loadingId);

    isTriageComplete = true;
    currentDiagnosis = data.final_diagnosis;
    const severityColor = data.severity;

    // 🟢 Save the data so the PDF builder can access it
    currentReportData = {
      diagnosis: data.final_diagnosis,
      patientReport: data.patient_friendly_report,
      clinicalReport: data.clinical_report,
      severity: data.severity,
      symptoms: data.extracted_symptoms || [],
      age: patientAge,
      sex: patientSex,
    };

    let borderStyle = "";
    let alertBanner = "";

    if (severityColor === "RED") {
      borderStyle =
        "border: 3px solid #ff4444; background-color: #fff0f0; box-shadow: 0 0 15px rgba(255, 68, 68, 0.4);";
      alertBanner = `<div style="color: #cc0000; font-weight: bold; margin-bottom: 15px; padding: 10px; background: #ffe5e5; border-radius: 5px;">🚨 EMERGENCY: Seek immediate medical attention. Do not wait.</div>`;

      // Auto-Trigger the GPS Map!
      setTimeout(() => {
        window.findNearestHospitals();
        document
          .getElementById("locate")
          .scrollIntoView({ behavior: "smooth" });
      }, 1500);
    } else if (severityColor === "YELLOW") {
      borderStyle = "border: 3px solid #ffbb33; background-color: #fffcf0;";
      alertBanner = `<div style="color: #b27b00; font-weight: bold; margin-bottom: 15px; padding: 10px; background: #fff8e5; border-radius: 5px;">⚠️ URGENT: Please schedule an appointment with a doctor soon.</div>`;
    } else {
      borderStyle = "border: 3px solid #00C851; background-color: #f0fff4;";
      alertBanner = `<div style="color: #007e33; font-weight: bold; margin-bottom: 15px; padding: 10px; background: #e5ffe5; border-radius: 5px;">✅ MINOR CONDITION: Review the safe home-care guidelines below.</div>`;
    }

    const uniqueId = Date.now();
    const reportHTML = `
        <div style="border-radius: 12px; padding: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); ${borderStyle}">
            ${alertBanner}
            <h3 style="color: var(--brand-dark); font-size: 1.3rem; margin-bottom: 20px; margin-top: 0;">
                🩺 Verified Diagnosis: <span style="color: var(--brand-blue);">${data.final_diagnosis}</span>
            </h3>
            
            <div class="tabs" style="border-bottom: 1px solid var(--border-light); margin-bottom: 20px; padding-bottom: 5px;">
                <button class="tab-btn active" onclick="switchTab(this, 'pat-${uniqueId}')">👤 Patient Report</button>
                <button class="tab-btn" onclick="switchTab(this, 'clin-${uniqueId}')">⚕️ Clinical Report</button>
            </div>
            
            <div id="pat-${uniqueId}" class="tab-content active">${window.highlightMedicalTerms(data.patient_friendly_report, data.extracted_symptoms).replace(/\n/g, "<br>")}</div>
            <div id="clin-${uniqueId}" class="tab-content hidden">${window.highlightMedicalTerms(data.clinical_report, data.extracted_symptoms).replace(/\n/g, "<br>")}</div>
            
            <div class="action-chips" style="margin-top: 25px; display: flex; gap: 8px; flex-wrap: wrap;">
                <button style="padding: 8px 14px; border-radius: 6px; font-size: 0.85rem; font-weight: 600; border: 1px solid #cbd5e1; background: #fff; color: #475569; cursor: pointer; transition: 0.2s;" onclick="document.getElementById('user-input').value='What are my treatment options?'; document.getElementById('btn-send').click();">Treatment Options</button>
                <button style="padding: 8px 14px; border-radius: 6px; font-size: 0.85rem; font-weight: 600; border: 1px solid #cbd5e1; background: #fff; color: #475569; cursor: pointer; transition: 0.2s;" onclick="document.getElementById('user-input').value='Are there any home remedies?'; document.getElementById('btn-send').click();">Home Remedies</button>
                <button style="padding: 8px 14px; border-radius: 6px; font-size: 0.85rem; font-weight: 600; border: 1px solid #cbd5e1; background: #fff; color: #475569; cursor: pointer; transition: 0.2s;" onclick="document.getElementById('user-input').value='What are the immediate Dos and Don\\'ts?'; document.getElementById('btn-send').click();">Dos and Don'ts</button>
                
                <button class="speaker-btn" style="padding: 8px 14px; border-radius: 6px; font-size: 0.85rem; font-weight: 700; border: 1px solid #bae6fd; background: #f0f9ff; color: #0ea5e9; cursor: pointer; transition: 0.2s; display: inline-flex; align-items: center; gap: 6px;" onclick="window.speakText(document.getElementById('pat-${uniqueId}').innerText, this)">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path><path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path></svg> Listen
                </button>

                <button style="padding: 8px 14px; border-radius: 6px; font-size: 0.85rem; font-weight: 600; border: none; background: #1e293b; color: white; cursor: pointer; transition: 0.2s;" onclick="generatePDF()">Download Report</button>
                <button style="padding: 8px 14px; border-radius: 6px; font-size: 0.85rem; font-weight: 600; border: none; background: #ef4444; color: white; cursor: pointer; transition: 0.2s;" onclick="document.getElementById('triage-modal').classList.add('hidden'); document.getElementById('locate').scrollIntoView({ behavior: 'smooth' }); window.findNearestHospitals();">View Nearby Hospitals</button>
            </div>
        </div>
    `;

    addHTMLMessage(reportHTML);
    setTimeout(() => {
      addMessage(
        "I have reviewed your results. Do you have any questions about this condition or what to do next?",
        "ai",
      );
      document.getElementById("user-input").placeholder =
        "Ask the Chief Medical Officer a question...";
    }, 500);
  } catch (error) {
    removeLoadingBubble(loadingId);
    document.getElementById("btn-predict").style.display = "inline-block";
    addMessage(`❌ **Diagnostic Engine Error:** ${error.message}`, "ai");
  } finally {
    document.getElementById("btn-predict").disabled = false;
    document.getElementById("btn-send").disabled = false;
    userInput.disabled = false;
    userInput.focus();
  }
};

// 🟢 BIND BOTH TO THE MASTER FUNCTION 🟢
document
  .getElementById("btn-predict")
  .addEventListener("click", window.runDiagnosticPrediction);

/// --- SEND LOGIC ---
btnSend.addEventListener("click", async () => {
  const text = userInput.value.trim();

  // 🟢 NEW: Check if there is text OR a file to send
  if (!text && !currentUploadedFile) return;

  btnSend.disabled = true;
  document.getElementById("btn-predict").disabled = true;
  userInput.disabled = true;

  // 1. Handle File Uploads First
  if (currentUploadedFile) {
    const loadingId = addLoadingBubble();
    const file = currentUploadedFile;
    const fileText = text || "Please analyze this medical document."; // Fallback text

    addMessage(`📎 Uploaded: ${file.name}\n${text}`, "user");
    userInput.value = "";
    userInput.style.height = "auto";

    // Clear the UI attachment
    currentUploadedFile = null;
    previewContainer.style.display = "none";
    fileInput.value = "";

    try {
      // Convert file to Base64
      const base64Data = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result.split(",")[1]);
        reader.onerror = (error) => reject(error);
        reader.readAsDataURL(file);
      });

      // Send to the dedicated Gemini parsing endpoint
      // Send to the dedicated Gemini parsing endpoint
      const response = await fetch(`${BACKEND_URL}/api/analyze-document`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mime_type: file.type,
          data: base64Data,
          prompt: fileText,
          phase: isTriageComplete ? "cmo" : "triage", // 🟢 FIX: Uses your actual state variable
        }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Server rejected the document.");
      }

      const data = await response.json();
      removeLoadingBubble(loadingId);

      // IF WE ARE IN CMO PHASE: Just chat normally about the image
      if (typeof isTriageComplete !== "undefined" && isTriageComplete) {
        addMessage(data.ai_reply, "ai");
        return;
      }

      // Auto-Fill Demographics and synchronize states
      let metadataString = "";
      if (data.age && data.age !== "UNKNOWN") {
        patientAge = data.age;
        const uiAge = document.getElementById("ui-age");
        if (uiAge) uiAge.innerText = patientAge;
      }
      if (data.sex && data.sex !== "UNKNOWN") {
        patientSex = data.sex;
        const uiSex = document.getElementById("ui-sex");
        if (uiSex) uiSex.innerText = patientSex;
      }

      // If both are present from document or manual input, clear the receptionist phase
      if (patientAge && patientSex) {
        isReceptionistPassed = true;
        metadataString = `[System Patient Metadata: Age ${patientAge}, Sex ${patientSex}]\n`;
      }

      // 🟢 THE AMNESIA PROTOCOL (Only run if receptionist phase was already done)
      if (
        typeof isReceptionistPassed !== "undefined" &&
        isReceptionistPassed &&
        apiChatHistory.length > 1
      ) {
        if (apiChatHistory[apiChatHistory.length - 1].role === "assistant") {
          apiChatHistory.pop();
        }
        let lines = fullTranscript
          .split("\n")
          .filter((line) => line.trim() !== "");
        if (lines.length > 0) lines.pop();
        fullTranscript = lines.join("\n") + "\n";
      }

      // Inject data into BOTH the text channel and Llama's memory
      fullTranscript +=
        metadataString +
        `[System Document Analysis: ${data.extracted_symptoms}]\n`;

      apiChatHistory.push({
        role: "assistant",
        content: `[System Image Scan Results]: ${data.extracted_symptoms}. Demographics: ${patientAge} ${patientSex}. I asked the patient if this looks accurate.`,
      });

      // 🟢 FORCE PREDICT BUTTON VISIBILITY
      // Check if your check function exists, or force style layout directly
      const btnPredict = document.getElementById("btn-predict"); // adjust to match your HTML ID
      if (btnPredict) {
        if ((patientAge && patientSex) || data.extracted_symptoms) {
          btnPredict.style.display = "block"; // Or replace with your function name like updateUI()
        }
      }

      addMessage(data.ai_reply, "ai");
      // 🟢 FIX 1: INJECT VISION DATA INTO LLAMA-3's MEMORY
      apiChatHistory.push({
        role: "assistant",
        content: `[System Image Scan Results]: ${data.extracted_symptoms}. ${data.ai_reply}`,
      });
      // ... existing verify button code continues below ...

      // Check valid/healthy status based on your existing logic
      const isInvalid =
        data.extracted_symptoms.includes("Invalid") ||
        data.extracted_symptoms.includes("System rate limit");
      const isHealthy = data.extracted_symptoms.includes(
        "No abnormalities detected",
      );

      // 🟢 Inject the Human-in-the-Loop "Verify" Button
      if (!isInvalid) {
        const verifyBtn = document.createElement("button");
        verifyBtn.innerText = "✅ Yes, data is correct";
        verifyBtn.style.cssText =
          "background: #10b981; color: white; border: none; padding: 8px 16px; border-radius: 8px; margin-top: 10px; cursor: pointer; font-weight: bold; display: block;";

        verifyBtn.onclick = function () {
          this.style.display = "none"; // hide button after click
          addMessage("Yes, data is correct.", "user");

          // Now we unlock the Predict button based on Health status!
          if (!isHealthy) {
            document.getElementById("btn-predict").style.display = "block";
            addMessage(
              "Great. You can click 'Predict' to run the diagnosis, or tell me any additional symptoms.",
              "ai",
            );
          } else {
            addMessage(
              "Since the report is normal, please type out any physical symptoms you are experiencing so we can investigate.",
              "ai",
            );
          }
        };

        // Append button to the last AI message safely
        const messages = document
          .getElementById("chat-box")
          .getElementsByClassName("bot-message");
        if (messages.length > 0) {
          messages[messages.length - 1].appendChild(verifyBtn);
        }
      }
    } catch (error) {
      removeLoadingBubble(loadingId);
      addMessage(`❌ **Upload Error:** ${error.message}`, "ai");
    } finally {
      btnSend.disabled = false;
      document.getElementById("btn-predict").disabled = false;
      userInput.disabled = false;
      userInput.focus();
    }
    return; // Stop the function here so it doesn't trigger standard chat
  }

  // 2. Standard Text Chat (Your existing logic continues here...)
  const wordCount = text.split(/\s+/).filter((word) => word.length > 0).length;
  addMessage(text, "user");
  userInput.value = "";
  userInput.style.height = "auto";

  if (isTriageComplete) {
    const loadingId = addLoadingBubble();
    try {
      const response = await fetch(`${BACKEND_URL}/api/followup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          diagnosis: currentDiagnosis,
          question: text,
          history: followUpHistory,
        }),
      });
      if (!response.ok) throw new Error("Server Error");
      const data = await response.json();
      removeLoadingBubble(loadingId);
      addMessage(data.reply, "ai");
      followUpHistory += `Patient: ${text}\nCMO: ${data.reply}\n`;
    } catch (error) {
      removeLoadingBubble(loadingId);
      addMessage(`❌ **Error:** ${error.message}`, "ai");
    } finally {
      // 🟢 FIX: Re-enable the buttons so you can ask multiple questions!
      btnSend.disabled = false;
      document.getElementById("btn-predict").disabled = false;
      userInput.disabled = false;
      userInput.focus();
    }
    return;
  }

  // ... existing code ...
  fullTranscript += `Patient: ${text}\n`;

  // 🟢 FIX: Ensure Predict button is ALWAYS visible once patient demographics are captured,
  // even if they manually type "yes" instead of clicking the Verify button!
  if (isReceptionistPassed) {
    document
      .getElementById("btn-predict")
      .style.setProperty("display", "inline-block", "important");
  }

  if (
    isReceptionistPassed &&
    diagnosticQuestionCount >= diagnosticQuestionLimit
  ) {
    // ... rest of your existing interceptor code ...
    const uniqueId = Date.now();
    const promptHTML = `
        <div id="intercept-${uniqueId}" style="margin-top: 15px; padding: 20px; background: white; border-radius: 12px; border: 1px solid var(--border-light); box-shadow: 0 4px 15px rgba(0,0,0,0.03);">
            <p style="margin-bottom: 15px; color: var(--brand-dark); font-weight: 600; font-size: 1.05rem;">
                I have gathered enough information to form a clinical picture. What would you like to do?
            </p>
            <div class="action-chips">
                <button class="chip" style="background: var(--brand-gradient); color: white; border: none;" onclick="document.getElementById('btn-predict').click(); document.getElementById('intercept-${uniqueId}').style.display='none';">🩺 Predict Diagnosis</button>
                <button class="chip" onclick="window.extendChat('intercept-${uniqueId}')">💬 Give More Symptoms</button>
            </div>
        </div>
    `;
    setTimeout(() => {
      addHTMLMessage(promptHTML);
      // LOCK THE UI
      document.getElementById("user-input").disabled = true;
      document.getElementById("btn-send").disabled = true;
    }, 400);
    return;
  }

  const loadingId = addLoadingBubble();

  try {
    const response = await fetch(`${BACKEND_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        history: apiChatHistory,
        age: patientAge ? parseInt(patientAge, 10) : null, // 🟢 FIX: Safely forces it to be an Integer
        sex: patientSex,
      }),
    });

    if (!response.ok) throw new Error("Server Error");

    // 🟢 NEW: Setup the Stream Reader
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");

    let fullAiReply = "";
    removeLoadingBubble(loadingId); // Remove bubble immediately

    // 🟢 Create a blank message bubble that we will inject words into
    const msgDiv = document.createElement("div");
    msgDiv.classList.add("message", "ai-message");
    const contentDiv = document.createElement("div");
    contentDiv.classList.add("message-content");
    msgDiv.appendChild(contentDiv);
    chatBox.appendChild(msgDiv);

    // Read the stream chunk by chunk
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunkText = decoder.decode(value, { stream: true });
      const lines = chunkText.split("\n");

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const dataObj = JSON.parse(line.substring(6));

          // Handle Metadata (Updating Profile & Buttons)
          if (dataObj.type === "metadata") {
            const isBypassed = dataObj.skip_demographics === true;
            if (
              (dataObj.extracted_age && dataObj.extracted_sex) ||
              isBypassed
            ) {
              if (!isReceptionistPassed) {
                isReceptionistPassed = true;
                document
                  .getElementById("btn-predict")
                  .style.setProperty("display", "inline-block", "important");
                document.getElementById("user-input").placeholder =
                  "Describe your symptoms...";
              }
              if (dataObj.extracted_age && dataObj.extracted_sex) {
                patientAge = dataObj.extracted_age;
                patientSex = dataObj.extracted_sex;
                document.getElementById("ui-age").innerText = patientAge;
                document.getElementById("ui-sex").innerText = patientSex;
              }
            }
          }

          // Handle Text Chunks (The Typewriter Effect)
          else if (dataObj.type === "chunk") {
            fullAiReply += dataObj.text;
            contentDiv.innerHTML = fullAiReply.replace(/\n/g, "<br>");
            chatBox.scrollTop = chatBox.scrollHeight; // Keep scrolling down as it types
          }
        }
      }
    }

    const speakerBtn = document.createElement("button");
    speakerBtn.className = "speaker-btn";
    speakerBtn.title = "Listen to answer";
    speakerBtn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path><path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path></svg>`;

    // 👇 ADD THIS ONE LINE RIGHT HERE 👇
    speakerBtn.style.cssText =
      "display: inline-block; margin-top: 10px; margin-left: 8px; cursor: pointer; color: #94a3b8;";

    speakerBtn.onclick = function () {
      window.speakText(fullAiReply, this);
    };
    contentDiv.appendChild(speakerBtn); // <-- Notice this is contentDiv, perfectly correct.
    chatBox.scrollTop = chatBox.scrollHeight;

    // --- Post-Stream Processing ---
    // If they typed "predict" or a massive essay, trigger the prediction directly
    if (wordCount >= 75 || text.toLowerCase().includes("predict")) {
      if (isReceptionistPassed) {
        window.runDiagnosticPrediction();
      } else {
        addMessage(
          "Please finish providing your initial details before predicting.",
          "ai",
        );
      }
      return;
    }

    // Save history and count questions
    apiChatHistory.push({ role: "user", content: text });
    apiChatHistory.push({ role: "assistant", content: fullAiReply });
    fullTranscript += `AI: ${fullAiReply}\n`;

    // 🟢 BULLETPROOF QUESTION INTERCEPTOR 🟢
    if (isReceptionistPassed) {
      diagnosticQuestionCount++;
      if (diagnosticQuestionCount === diagnosticQuestionLimit) {
        const uniqueId = Date.now();
        const promptHTML = `
            <div id="intercept-${uniqueId}" style="margin-top: 15px; padding: 20px; background: white; border-radius: 12px; border: 1px solid var(--border-light); box-shadow: 0 4px 15px rgba(0,0,0,0.03);">
                <p style="margin-bottom: 15px; color: var(--brand-dark); font-weight: 600; font-size: 1.05rem;">
                    I have gathered enough information to form a clinical picture. What would you like to do?
                </p>
                <div class="action-chips">
                    <button class="chip" style="background: var(--brand-gradient); color: white; border: none;" onclick="document.getElementById('intercept-${uniqueId}').style.display='none'; window.runDiagnosticPrediction();">🩺 Predict Diagnosis</button>
                    <button class="chip" onclick="window.extendChat('intercept-${uniqueId}')">💬 Give More Symptoms</button>
                </div>
            </div>
        `;
        setTimeout(() => {
          addHTMLMessage(promptHTML);
          // LOCK THE UI
          document.getElementById("user-input").disabled = true;
          document.getElementById("btn-send").disabled = true;
        }, 600);
      }
    }
  } catch (error) {
    removeLoadingBubble(loadingId);
    addMessage(`❌ **API Error:** ${error.message}`, "ai");
  } finally {
    btnSend.disabled = false;
    document.getElementById("btn-predict").disabled = false;
    userInput.disabled = false;
    userInput.focus();
  }
});

// --- UTILITIES ---
window.switchTab = function (btn, tabId) {
  const container = btn.closest(".message-content");
  container
    .querySelectorAll(".tab-content")
    .forEach((tab) => tab.classList.add("hidden"));
  container
    .querySelectorAll(".tab-btn")
    .forEach((b) => b.classList.remove("active"));
  container.querySelector(`#${tabId}`).classList.remove("hidden");
  btn.classList.add("active");
};

// --- NLP HIGHLIGHTER UTILITY ---
window.highlightMedicalTerms = function (text, symptomsList) {
  if (!text || !symptomsList || symptomsList.length === 0) return text;
  let newText = text;
  symptomsList.forEach((symp) => {
    if (symp.trim().length > 2) {
      // Prevents highlighting small words like 'a' or 'is'
      // Finds the exact symptom word and wraps it in a blue highlighted span
      const regex = new RegExp(`\\b(${symp.trim()})\\b`, "gi");
      newText = newText.replace(
        regex,
        `<strong style="color: var(--brand-blue); background: #e0f2fe; padding: 2px 4px; border-radius: 4px;">$1</strong>`,
      );
    }
  });
  return newText;
};
userInput.addEventListener("input", function () {
  this.style.height = "auto";
  this.style.height = this.scrollHeight + "px";
});

userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    btnSend.click();
  }
});

window.extendChat = function (boxId) {
  document.getElementById(boxId).style.display = "none";
  diagnosticQuestionLimit += 4;

  // UNLOCK THE UI
  const userInput = document.getElementById("user-input");
  userInput.disabled = false;
  document.getElementById("btn-send").disabled = false;
  userInput.focus();

  userInput.value = "I have more symptoms I want to discuss.";
  document.getElementById("btn-send").click();
};
// ==========================================
// 🟢 THE NATIVE BROWSER PDF ENGINE 🟢
// ==========================================
window.generatePDF = function () {
  if (!currentReportData) {
    alert("No report data found. Please run a prediction first.");
    return;
  }

  // 1. Data Prep
  const dateStr = new Date().toLocaleDateString();
  const symptomsList =
    currentReportData.symptoms && currentReportData.symptoms.length > 0
      ? currentReportData.symptoms
          .map(
            (s) =>
              `<li style="margin-bottom: 5px;">${s.charAt(0).toUpperCase() + s.slice(1)}</li>`,
          )
          .join("")
      : "<li>As described in transcript</li>";

  let severityLabel = "Standard Care";
  let severityColor = "#00C851";
  let severityBg = "#e5ffe5"; // Light green background for print
  if (currentReportData.severity === "RED") {
    severityLabel = "CRITICAL EMERGENCY";
    severityColor = "#ff4444";
    severityBg = "#ffe5e5";
  } else if (currentReportData.severity === "YELLOW") {
    severityLabel = "URGENT CONDITION";
    severityColor = "#ffbb33";
    severityBg = "#fff8e5";
  }

  const clinicalRep = (
    currentReportData.clinicalReport || "No clinical report generated."
  ).replace(/\n/g, "<br>");
  const patientRep = (
    currentReportData.patientReport || "No patient report generated."
  ).replace(/\n/g, "<br>");
  const diag = currentReportData.diagnosis
    ? String(currentReportData.diagnosis)
    : "Unknown Diagnosis";
  const patAge = currentReportData.age || "Unknown";
  const patSex = currentReportData.sex || "Unknown";

  // 2. Create a dedicated container meant ONLY for printing
  const printContainer = document.createElement("div");
  printContainer.id = "native-print-report";
  printContainer.style.fontFamily =
    "'Helvetica Neue', Helvetica, Arial, sans-serif";
  printContainer.style.color = "#0f172a";
  printContainer.style.backgroundColor = "white";
  printContainer.style.width = "100%";
  printContainer.style.maxWidth = "800px";
  printContainer.style.margin = "0 auto";

  // Note: -webkit-print-color-adjust forces the browser to print background colors perfectly
  printContainer.innerHTML = `
      <div style="border-bottom: 3px solid #0ea5e9; padding-bottom: 20px; margin-bottom: 30px; display: flex; justify-content: space-between; align-items: flex-end;">
          <div>
              <h1 style="color: #0ea5e9; margin: 0; font-size: 32px; letter-spacing: -1px;">Heal Bridge</h1>
              <p style="margin: 5px 0 0 0; font-size: 14px; color: #64748b; font-weight: bold;">Autonomous AI Medical Triage Report</p>
          </div>
          <div style="text-align: right; font-size: 14px; color: #64748b;">
              <p style="margin: 0;"><strong>Date:</strong> ${dateStr}</p>
          </div>
      </div>

      <div style="display: flex; justify-content: space-between; margin-bottom: 30px; background: #f8fafc; padding: 20px; border-radius: 8px; border: 1px solid #e2e8f0; -webkit-print-color-adjust: exact; print-color-adjust: exact;">
          <div>
              <p style="margin: 0 0 5px 0; color: #64748b; text-transform: uppercase; font-size: 12px; font-weight: bold;">Patient Profile</p>
              <p style="margin: 0; font-size: 16px; font-weight: bold;">Age: ${patAge} &nbsp;|&nbsp; Sex: ${patSex}</p>
          </div>
          <div style="text-align: right; background: ${severityBg}; padding: 10px; border-radius: 6px; -webkit-print-color-adjust: exact; print-color-adjust: exact;">
              <p style="margin: 0 0 5px 0; color: #64748b; text-transform: uppercase; font-size: 12px; font-weight: bold;">Triage Severity</p>
              <p style="margin: 0; font-weight: bold; font-size: 16px; color: ${severityColor};">${severityLabel}</p>
          </div>
      </div>

      <div style="margin-bottom: 30px;">
          <h2 style="font-size: 16px; color: #64748b; text-transform: uppercase; border-bottom: 1px solid #e2e8f0; padding-bottom: 5px; margin-bottom: 15px;">Primary Diagnosis</h2>
          <h3 style="font-size: 26px; color: #0ea5e9; margin: 0;">${diag}</h3>
      </div>

      <div style="margin-bottom: 30px;">
          <h2 style="font-size: 16px; color: #64748b; text-transform: uppercase; border-bottom: 1px solid #e2e8f0; padding-bottom: 5px; margin-bottom: 15px;">Detected Symptoms</h2>
          <ul style="font-size: 14px; line-height: 1.6; padding-left: 20px; font-weight: 500;">
              ${symptomsList}
          </ul>
      </div>

      <div style="margin-bottom: 30px;">
          <h2 style="font-size: 16px; color: #64748b; text-transform: uppercase; border-bottom: 1px solid #e2e8f0; padding-bottom: 5px; margin-bottom: 15px;">Clinical Report (For Attending Physician)</h2>
          <p style="font-size: 14px; line-height: 1.6;">${clinicalRep}</p>
      </div>

      <div style="margin-bottom: 30px;">
          <h2 style="font-size: 16px; color: #64748b; text-transform: uppercase; border-bottom: 1px solid #e2e8f0; padding-bottom: 5px; margin-bottom: 15px;">Patient Summary</h2>
          <p style="font-size: 14px; line-height: 1.6;">${patientRep}</p>
      </div>

      <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #e2e8f0; text-align: center; font-size: 11px; color: #94a3b8; line-height: 1.5;">
          <p><strong>Disclaimer:</strong> Generated by the Heal Bridge AI Engine. This is a preliminary triage report generated via algorithmic analysis and does not constitute a definitive medical diagnosis. Please present this document to a licensed medical professional.</p>
      </div>
  `;

  document.body.appendChild(printContainer);

  // 3. 🟢 THE MAGIC CSS 🟢
  // When the print dialog opens, this CSS hides the entire website (navbar, chat, buttons)
  // and ONLY allows the browser to see the beautiful report container we just made.
  const printStyles = document.createElement("style");
  printStyles.id = "native-print-styles";
  printStyles.innerHTML = `
    @media print {
        /* Hide everything on the entire website */
        body > *:not(#native-print-report):not(script):not(style) {
            display: none !important;
        }
        /* Show ONLY the report */
        #native-print-report {
            display: block !important;
            position: absolute !important;
            left: 0 !important;
            top: 0 !important;
            width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        /* Force background colors to print, remove browser margins */
        body {
            background: white !important;
            margin: 0 !important;
        }
        @page {
            margin: 0.5in;
        }
    }
  `;
  document.head.appendChild(printStyles);

  // 4. Trigger the indestructible Native Browser Print Engine
  window.print();

  // 5. Clean up the DOM completely once the user closes the print window
  setTimeout(() => {
    document.body.removeChild(printContainer);
    document.head.removeChild(printStyles);
  }, 1000);
};
// ==========================================
// 🟢 GEOLOCATION & REAL DYNAMIC DIRECTORY 🟢
// ==========================================

let userActualLat = null;
let userActualLon = null;

window.findNearestHospitals = async function (manualLocation = "") {
  const mapContainer = document.getElementById("map-container");
  const directoryPanel = document.querySelector(".directory-panel");

  mapContainer.innerHTML = `<p id="map-status-text" style="font-weight: 600; color: var(--brand-blue);">📍 Locating nearby medical centers...</p>`;
  directoryPanel.innerHTML = `<div style="text-align:center; padding: 40px; color: var(--text-muted);">🔄 Searching global satellite database...</div>`;

  // 🟢 PATH A: Manual Search (User typed a city)
  if (manualLocation.trim() !== "") {
    try {
      const geoResponse = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(manualLocation)}`,
      );
      const geoData = await geoResponse.json();

      if (geoData && geoData.length > 0) {
        const searchLat = geoData[0].lat;
        const searchLon = geoData[0].lon;

        // 🏆 Official Maps Embed URL 🏆
        const mapQuery = encodeURIComponent(`hospitals near ${manualLocation}`);
        const mapUrl = `https://maps.google.com/maps?q=${mapQuery}&t=&z=13&ie=UTF8&iwloc=&output=embed`;

        mapContainer.innerHTML = `<iframe width="100%" height="100%" frameborder="0" style="border:0; border-radius: 12px;" src="${mapUrl}"></iframe>`;

        fetchRealHospitals(searchLat, searchLon, false);
      } else {
        directoryPanel.innerHTML = `<div style="padding: 20px; text-align: center;">City not found. Please try a different name or zip code.</div>`;
        mapContainer.innerHTML = `<p style="color: #ef4444; font-weight: 600;">❌ Map unavailable.</p>`;
      }
    } catch (error) {
      directoryPanel.innerHTML = `<div style="padding: 20px; color: #ef4444;">Error fetching location data.</div>`;
    }
    return;
  }

  // 🟢 PATH B: GPS Auto-Location (Using YOUR Reverse Geocoding Idea!)
  if (userActualLat !== null && userActualLon !== null) {
    try {
      // 1. Ask OpenStreetMap what city/town your GPS coordinates are in
      const reverseGeoResponse = await fetch(
        `https://nominatim.openstreetmap.org/reverse?format=json&lat=${userActualLat}&lon=${userActualLon}`,
      );
      const reverseGeoData = await reverseGeoResponse.json();

      // 2. Extract the name of the city, town, or village
      const address = reverseGeoData.address || {};
      const detectedCity =
        address.city ||
        address.town ||
        address.village ||
        address.suburb ||
        address.county ||
        "my location";

      // 3. Pass that extracted city name to the Google Map iframe!
      const mapQuery = encodeURIComponent(`hospitals near ${detectedCity}`);
      const mapUrl = `https://maps.google.com/maps?q=${mapQuery}&t=&z=13&ie=UTF8&iwloc=&output=embed`;

      mapContainer.innerHTML = `<iframe width="100%" height="100%" frameborder="0" style="border:0; border-radius: 12px;" src="${mapUrl}"></iframe>`;

      fetchRealHospitals(userActualLat, userActualLon, true);
    } catch (error) {
      directoryPanel.innerHTML = `<div style="padding: 20px; color: #ef4444;">Error finding your exact city.</div>`;
    }
  } else {
    mapContainer.innerHTML = `<p style="color: #ef4444; font-weight: 600;">❌ Please click "Near Me" or type your City above.</p>`;
    directoryPanel.innerHTML = `<div style="padding: 20px; text-align:center;">Location access is required to show nearby facilities automatically.<br><br><b>Please click "Near Me" or search above.</b></div>`;
  }
};

// --- FETCH REAL OPEN-SOURCE HOSPITAL DATA (SUPER-QUERY) ---
async function fetchRealHospitals(searchLat, searchLon) {
  const directoryPanel = document.querySelector(".directory-panel");

  const query = `
    [out:json][timeout:25];
    (
      node["amenity"="hospital"](around:25000,${searchLat},${searchLon});
      way["amenity"="hospital"](around:25000,${searchLat},${searchLon});
      relation["amenity"="hospital"](around:25000,${searchLat},${searchLon});
      node["healthcare"="hospital"](around:25000,${searchLat},${searchLon});
      way["healthcare"="hospital"](around:25000,${searchLat},${searchLon});
      relation["healthcare"="hospital"](around:25000,${searchLat},${searchLon});
    );
    out center;
  `;

  try {
    const response = await fetch("https://overpass-api.de/api/interpreter", {
      method: "POST",
      headers: {
        Accept: "*/*",
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: query,
    });

    const data = await response.json();

    if (data.elements && data.elements.length > 0) {
      directoryPanel.innerHTML = "";

      // 🏆 THE REAL FIX 2: ALWAYS CALCULATE DISTANCE FROM GPS IF AVAILABLE 🏆
      const originLat = userActualLat !== null ? userActualLat : searchLat;
      const originLon = userActualLon !== null ? userActualLon : searchLon;

      // 1. Calculate Distances
      let facilities = data.elements.map((place) => {
        const placeLat = place.lat || (place.center && place.center.lat);
        const placeLon = place.lon || (place.center && place.center.lon);

        // KEEP DISTANCE: Calculates how far it is from your physical GPS body
        const displayDistance = calculateDistance(
          originLat,
          originLon,
          placeLat,
          placeLon,
        );

        // 🏆 NEW SORT DISTANCE: Calculates how far it is from the Searched City Center
        const sortDistance = calculateDistance(
          searchLat,
          searchLon,
          placeLat,
          placeLon,
        );

        return {
          place,
          placeLat,
          placeLon,
          distance: displayDistance,
          sortDistance: sortDistance,
        };
      });

      // 2. Filter out bad data
      facilities = facilities.filter((item) => {
        if (!item.place.tags.name) return false;
        const nameLow = item.place.tags.name.toLowerCase();
        if (
          nameLow.includes("clinic") ||
          nameLow.includes("dispensary") ||
          nameLow.includes("pharmacy")
        )
          return false;
        return true;
      });

      // 3. 🏆 THE FIX: Sort by the city center, NOT your GPS distance!
      facilities.sort((a, b) => a.sortDistance - b.sortDistance);

      // 4. Take the top 15 remaining major hospitals
      const topFacilities = facilities.slice(0, 15);

      if (topFacilities.length === 0) {
        directoryPanel.innerHTML = `<div style="padding:20px; text-align:center; font-weight: 600;">No major hospitals found within 25 km. Try a different city.</div>`;
        return;
      }

      topFacilities.forEach((item) => {
        const { place, placeLat, placeLon, distance } = item;
        const name = place.tags.name;

        const isER = place.tags.emergency === "yes";
        let facilityCategory = "general";
        let typeText = "🏥 General Hospital";
        let emergencyTag = `<span class="status-badge open">🟢 General Care</span>`;

        if (isER) {
          facilityCategory = "emergency";
          typeText = "🚨 Trauma & Emergency";
          emergencyTag = `<span class="status-badge emergency">🚨 ER Available</span>`;
        }

        const formattedDistance = distance.toFixed(1);

        // 🏆 THE REAL FIX 3: OFFICIAL GOOGLE MAPS NAVIGATION URL 🏆
        // 🏆 FIX 3: Official Google Maps Directions API
        const directionsUrl = `https://www.google.com/maps/dir/?api=1&destination=${placeLat},${placeLon}`;
        const cardHTML = `
          <div class="facility-card" data-category="${facilityCategory}">
            <div class="card-header">
              <h3 style="font-size: 1.05rem;">${name}</h3>
              ${emergencyTag}
            </div>
            <p class="facility-type">${typeText}</p>
            <div class="card-footer">
              <span class="distance">🚗 ~${formattedDistance} km away</span>
              <a href="${directionsUrl}" target="_blank" class="card-action-btn" style="text-decoration:none; text-align:center; display:inline-block;">Get Directions</a>
            </div>
          </div>
        `;
        directoryPanel.innerHTML += cardHTML;
      });
    } else {
      directoryPanel.innerHTML = `<div style="padding:20px; text-align:center; font-weight: 600;">No major hospitals found within 25 km of this location.</div>`;
    }
  } catch (err) {
    directoryPanel.innerHTML = `<div style="padding:20px; color: #ef4444;">Error syncing with global hospital database.</div>`;
  }
}

// --- HAVERSINE FORMULA WITH ROAD-ROUTING MULTIPLIER ---
function calculateDistance(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const dLat = (lat2 - lat1) * (Math.PI / 180);
  const dLon = (lon2 - lon1) * (Math.PI / 180);
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(lat1 * (Math.PI / 180)) *
      Math.cos(lat2 * (Math.PI / 180)) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c * 1.45;
}

// --- FILTER CHIP LOGIC ---
document.querySelectorAll(".filter-chips .chip").forEach((chip) => {
  chip.addEventListener("click", (e) => {
    document
      .querySelectorAll(".filter-chips .chip")
      .forEach((c) => c.classList.remove("active"));
    e.target.classList.add("active");

    const filterText = e.target.innerText.toLowerCase();
    const allCards = document.querySelectorAll(".facility-card");

    allCards.forEach((card) => {
      const category = card.getAttribute("data-category");
      if (filterText.includes("all")) {
        card.style.display = "block";
      } else if (filterText.includes("emergency") && category === "emergency") {
        card.style.display = "block";
      } else if (filterText.includes("physicians") && category === "general") {
        card.style.display = "block";
      } else if (filterText.includes("pediatrics") && category === "clinic") {
        card.style.display = "block";
      } else {
        card.style.display = "none";
      }
    });
  });
});

// --- NEW BUTTON LOGIC: "NEAR ME" ---
const btnMyLocation = document.getElementById("btn-my-location");
if (btnMyLocation) {
  btnMyLocation.addEventListener("click", () => {
    if (navigator.geolocation) {
      document.getElementById("map-search-input").value = "";
      document.getElementById("map-container").innerHTML =
        `<p style="font-weight: 600; color: var(--brand-blue);">📍 Requesting GPS Access...</p>`;

      navigator.geolocation.getCurrentPosition(
        (position) => {
          userActualLat = position.coords.latitude;
          userActualLon = position.coords.longitude;
          window.findNearestHospitals("");
        },
        (error) => {
          alert(
            "Location access denied. Please type your city in the search bar.",
          );
          window.findNearestHospitals();
        },
      );
    }
  });
}

// --- INITIALIZE GEOLOCATION ON PAGE LOAD ---
window.addEventListener("DOMContentLoaded", () => {
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      (position) => {
        userActualLat = position.coords.latitude;
        userActualLon = position.coords.longitude;
        window.findNearestHospitals("");
      },
      (error) => {
        window.findNearestHospitals();
      },
    );
  } else {
    window.findNearestHospitals();
  }
});

// Hook up the Search Button
const searchBtn = document.getElementById("btn-search-map");
const searchInput = document.getElementById("map-search-input");
if (searchBtn && searchInput) {
  searchBtn.addEventListener("click", () =>
    findNearestHospitals(searchInput.value),
  );
  searchInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      findNearestHospitals(searchInput.value);
    }
  });
}

// ==========================================
// 🟢 HIGH-END UX INTEGRATIONS
// ==========================================

// --- 1. PORTFOLIO TEXT-TO-SPEECH ENGINE ---
let currentUtterance = null;
window.speechSynthesis.getVoices(); // Pre-load voices

const iconPlay = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path><path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path></svg>`;
const iconStop = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect></svg>`;

window.speakText = function (text, buttonElement) {
  // If playing, stop it
  if (buttonElement.classList.contains("playing")) {
    window.speechSynthesis.cancel();
    buttonElement.classList.remove("playing");
    buttonElement.innerHTML = iconPlay;
    return;
  }

  // Cancel any other speaking audio
  window.speechSynthesis.cancel();
  document.querySelectorAll(".speaker-btn").forEach((btn) => {
    btn.classList.remove("playing");
    btn.innerHTML = iconPlay;
  });

  // Clean the markdown text
  let cleanText = text.replace(/[*#_]/g, "").replace(/•/g, ". ");

  currentUtterance = new SpeechSynthesisUtterance(cleanText);
  currentUtterance.rate = 1.0;

  // 🟢 NEW: THE AUTO-DETECTION ENGINE
  const voices = window.speechSynthesis.getVoices();
  let bestVoice;

  // Check if the text contains Hindi (Devanagari) characters
  const isHindi = /[\u0900-\u097F]/.test(cleanText);

  if (isHindi) {
    // Force the native Hindi voice
    bestVoice = voices.find(
      (v) => v.lang.includes("hi-IN") || v.lang.includes("hi"),
    );
  } else {
    // Default back to Indian English or UK English
    bestVoice = voices.find(
      (v) =>
        v.lang.includes("en-IN") ||
        v.name.includes("India") ||
        v.name.includes("UK"),
    );
  }

  if (bestVoice) {
    currentUtterance.voice = bestVoice;
  }

  // Set UI to playing
  buttonElement.classList.add("playing");
  buttonElement.innerHTML = iconStop;

  // Reset UI when done
  currentUtterance.onend = () => {
    buttonElement.classList.remove("playing");
    buttonElement.innerHTML = iconPlay;
  };

  window.speechSynthesis.speak(currentUtterance);
};
/*
// --- 2. GOOGLE PLACES AUTOCOMPLETE ---
window.addEventListener("DOMContentLoaded", () => {
  const input = document.getElementById("map-search-input");

  // Check if Google Maps is loaded in index.html
  if (
    input &&
    window.google &&
    window.google.maps &&
    window.google.maps.places
  ) {
    const autocomplete = new google.maps.places.Autocomplete(input);

    // When the user clicks a dropdown suggestion, auto-trigger the map!
    autocomplete.addListener("place_changed", () => {
      const place = autocomplete.getPlace();
      if (place.formatted_address || place.name) {
        window.findNearestHospitals(place.formatted_address || place.name);
      }
    });
  }
});
*/

// --- 3. SCROLL-TRIGGERED ANIMATIONS ---
window.addEventListener("DOMContentLoaded", () => {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target); // Only animate once
        }
      });
    },
    { threshold: 0.15 },
  ); // Triggers when 15% of the element is visible

  // Grab the elements we want to animate and attach them to the observer
  const animatedElements = document.querySelectorAll(
    ".bento-card-clean, .stream-card, .section-header, .search-bar-unified",
  );

  animatedElements.forEach((el) => {
    el.classList.add("fade-up");
    observer.observe(el);
  });
});
// ==========================================
// 🟢 4. VOICE-TO-TEXT ENGINE (SPEECH RECOGNITION)
// ==========================================
window.addEventListener("DOMContentLoaded", () => {
  const btnMic = document.getElementById("btn-mic");
  const inputArea = document.getElementById("user-input");

  // Check browser support for Speech API
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;

  if (SpeechRecognition) {
    const recognition = new SpeechRecognition();
    recognition.continuous = false; // Stops automatically when they pause speaking
    recognition.interimResults = true; // Shows words in real-time as they talk

    let isRecording = false;

    btnMic.addEventListener("click", () => {
      if (isRecording) {
        recognition.stop();
        return;
      }
      recognition.start();
    });

    recognition.onstart = () => {
      isRecording = true;
      btnMic.classList.add("recording");
      inputArea.placeholder = "Listening... Speak your symptoms clearly.";
    };

    recognition.onresult = (event) => {
      let interimTranscript = "";
      let finalTranscript = "";

      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        } else {
          interimTranscript += event.results[i][0].transcript;
        }
      }

      // Combine existing text with the new voice input
      inputArea.value = finalTranscript || interimTranscript;

      // Auto-expand the textarea so the voice text fits perfectly
      inputArea.style.height = "auto";
      inputArea.style.height = inputArea.scrollHeight + "px";
    };

    recognition.onend = () => {
      isRecording = false;
      btnMic.classList.remove("recording");
      inputArea.placeholder = "Type your message or upload a document...";
    };

    recognition.onerror = (event) => {
      console.error("Speech recognition error", event.error);
      isRecording = false;
      btnMic.classList.remove("recording");
      inputArea.placeholder = "Type your message or upload a document...";

      if (event.error === "not-allowed") {
        alert(
          "Microphone access was denied. Please enable it in your browser settings to use voice features.",
        );
      }
    };
  } else {
    // Hide the mic button if the browser doesn't support the Speech API (e.g., Firefox without flags)
    if (btnMic) btnMic.style.display = "none";
  }
});

// ==========================================
// 🟢 FREE OPEN-SOURCE AUTOCOMPLETE (NOMINATIM)
// ==========================================
window.addEventListener("DOMContentLoaded", () => {
  const input = document.getElementById("map-search-input");
  if (!input) return;

  // 1. Create the dropdown list
  const dropdown = document.createElement("ul");
  dropdown.className = "autocomplete-dropdown";

  // 2. Inject it right below the unified search bar
  const searchBar = document.querySelector(".search-bar-unified");
  if (searchBar) {
    searchBar.parentNode.insertBefore(dropdown, searchBar.nextSibling);
  }

  let timeoutId;

  // 3. Listen to typing
  input.addEventListener("input", function () {
    clearTimeout(timeoutId);
    const query = this.value;

    // Wait until they type at least 3 letters
    if (query.length < 3) {
      dropdown.classList.remove("active");
      return;
    }

    // Wait 300ms after they stop typing to fetch data (Debouncing)
    timeoutId = setTimeout(async () => {
      try {
        // Ping the free OpenStreetMap API for city suggestions
        const res = await fetch(
          `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=5`,
        );
        const data = await res.json();

        dropdown.innerHTML = ""; // Clear old results

        if (data.length > 0) {
          data.forEach((item) => {
            const li = document.createElement("li");
            li.className = "autocomplete-item";

            // Format the text nicely
            const parts = item.display_name.split(",");
            const mainText = parts[0];
            const subText = parts.slice(1, 3).join(","); // Just show state/country to keep it clean

            li.innerHTML = `📍 <span><strong>${mainText}</strong>${subText ? "," + subText : ""}</span>`;

            // When clicked, auto-fill and run the map search!
            li.addEventListener("click", () => {
              input.value = item.display_name;
              dropdown.classList.remove("active");
              window.findNearestHospitals(item.display_name);
            });

            dropdown.appendChild(li);
          });
          dropdown.classList.add("active");
        } else {
          dropdown.classList.remove("active");
        }
      } catch (err) {
        console.error("Autocomplete error:", err);
      }
    }, 300);
  });

  // 4. Hide dropdown if the user clicks anywhere else on the screen
  document.addEventListener("click", (e) => {
    if (e.target !== input && !dropdown.contains(e.target)) {
      dropdown.classList.remove("active");
    }
  });
});
// ==========================================
// 🟢 5. LIVE CAMERA VISION ENGINE
// ==========================================
window.addEventListener("DOMContentLoaded", () => {
  const cameraModal = document.getElementById("camera-modal");
  const cameraVideo = document.getElementById("camera-video");
  const cameraCanvas = document.getElementById("camera-canvas");
  const btnCamera = document.getElementById("btn-camera");
  const btnCloseCamera = document.getElementById("btn-close-camera");
  const btnCapture = document.getElementById("btn-capture");

  let stream = null;

  // 1. Open Camera Viewfinder
  if (btnCamera) {
    btnCamera.addEventListener("click", async () => {
      cameraModal.classList.remove("hidden");
      try {
        // Request camera access (prefer rear camera for medical scans)
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment" },
          audio: false,
        });
        cameraVideo.srcObject = stream;
      } catch (err) {
        console.error("Camera error:", err);
        alert(
          "Camera access denied or unavailable. Please check your browser permissions.",
        );
        cameraModal.classList.add("hidden");
      }
    });
  }

  // 2. Close Camera Viewfinder safely
  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
    }
    cameraModal.classList.add("hidden");
  }

  if (btnCloseCamera) {
    btnCloseCamera.addEventListener("click", stopCamera);
  }

  // 3. Snap the Photo & Inject into Upload System
  if (btnCapture) {
    btnCapture.addEventListener("click", () => {
      // Draw the current video frame onto the invisible canvas
      cameraCanvas.width = cameraVideo.videoWidth;
      cameraCanvas.height = cameraVideo.videoHeight;
      cameraCanvas.getContext("2d").drawImage(cameraVideo, 0, 0);

      // Convert the canvas to a Base64 image
      const dataUrl = cameraCanvas.toDataURL("image/jpeg", 0.9);

      // Convert Base64 into a standard Javascript File object
      fetch(dataUrl)
        .then((res) => res.blob())
        .then((blob) => {
          const file = new File([blob], "live-camera-scan.jpg", {
            type: "image/jpeg",
          });

          // 🟢 INJECT: Hand it over to your existing file upload logic!
          currentUploadedFile = file;

          // Update your UI to show the image is attached and ready to send
          const previewContainer = document.getElementById(
            "file-preview-container",
          );
          const previewName = document.getElementById("file-preview-name");
          const previewIcon = document.getElementById("file-preview-icon");

          if (previewName) previewName.innerText = "Live Camera Scan 📸";
          if (previewIcon) previewIcon.innerText = "🖼️";
          if (previewContainer) previewContainer.style.display = "flex";

          // Shut down the camera
          stopCamera();
        });
    });
  }
});
// ==========================================
// 🟢 6. ADVANCED 3D SYMPTOM MAPPER (THREE.JS)
// ==========================================
window.addEventListener("DOMContentLoaded", () => {
  const mapModal = document.getElementById("bodymap-modal");
  const btnOpenMap = document.getElementById("btn-body-map");
  const btnCloseMap = document.getElementById("btn-close-bodymap");
  const btnConfirm = document.getElementById("btn-confirm-symptoms");
  const container = document.getElementById("three-canvas-container");
  const hoverLabel = document.getElementById("hover-label");
  const tagsContainer = document.getElementById("selected-symptoms-container");
  const btnExternal = document.getElementById("btn-view-external");
  const btnInternal = document.getElementById("btn-view-internal");

  if (!container || !window.THREE) return;

  let scene, camera, renderer, controls, raycaster, mouse;
  let externalGroup, internalGroup;
  let isInternalView = false;
  let selectedParts = new Set();
  let allMeshes = [];

  // --- CORE UI COLORS ---
  const COLOR_SKIN_BASE = 0x334155; // Dark slate for skin
  const COLOR_SKIN_HOVER = 0x38bdf8; // Bright blue for skin hover
  const COLOR_SKIN_SELECT = 0x0ea5e9; // Solid blue for selected skin

  const COLOR_ORGAN_HOVER = 0xffffff; // Pure White for organ hover
  const COLOR_ORGAN_SELECT = 0x00ffff; // Neon Cyan for selected organs (High Visibility)

  // --- ANATOMICAL ORGAN COLORS ---
  const ORG_BRAIN = 0xd8b4e2; // Soft Purple
  const ORG_HEART = 0xff4d4d; // Vibrant Red
  const ORG_LUNGS = 0x87cefa; // Sky Blue
  const ORG_LIVER = 0x8b0000; // Dark Red
  const ORG_STOMACH = 0xffa500; // Orange
  const ORG_KIDNEYS = 0xda70d6; // Orchid Magenta
  const ORG_INTESTINES = 0x98fb98; // Pale Green

  function init3D() {
    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(
      45,
      container.clientWidth / container.clientHeight,
      0.1,
      100,
    );
    camera.position.set(0, 0, 15);

    renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.innerHTML = "";
    container.appendChild(renderer.domElement);
    container.appendChild(hoverLabel);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
    dirLight.position.set(10, 20, 10);
    scene.add(dirLight);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enablePan = false;
    controls.minDistance = 5;
    controls.maxDistance = 25;

    externalGroup = new THREE.Group();
    internalGroup = new THREE.Group();
    scene.add(externalGroup);
    scene.add(internalGroup);

    const sphereGeo = new THREE.SphereGeometry(1, 32, 32);

    // 🟢 UPGRADED: Accepts custom colors and saves them to memory
    function createPart(
      name,
      group,
      isInternal,
      x,
      y,
      z,
      scaleX,
      scaleY,
      scaleZ,
      customColor = null,
    ) {
      const baseColor = customColor || COLOR_SKIN_BASE;

      const material = new THREE.MeshStandardMaterial({
        color: baseColor,
        transparent: true,
        opacity: 1,
        depthWrite: true,
        roughness: 0.3,
        metalness: 0.1,
      });

      const mesh = new THREE.Mesh(sphereGeo, material);
      mesh.position.set(x, y, z);
      mesh.scale.set(scaleX, scaleY, scaleZ);

      // Save the original color so we can revert back to it when un-clicking
      mesh.userData = {
        name: name,
        isInternal: isInternal,
        isSelected: false,
        baseColor: baseColor,
      };

      group.add(mesh);
      allMeshes.push(mesh);
      return mesh;
    }

    // --- BUILD EXTERNAL MANNEQUIN (Dark Theme) ---
    // Head & Torso
    createPart("Face", externalGroup, false, 0, 4.5, 0.3, 0.9, 1.2, 0.8);
    createPart(
      "Back of Head",
      externalGroup,
      false,
      0,
      4.5,
      -0.3,
      0.9,
      1.2,
      0.8,
    );
    createPart("Neck", externalGroup, false, 0, 3.2, 0, 0.5, 0.6, 0.5);
    createPart(
      "Left Chest",
      externalGroup,
      false,
      -0.8,
      1.8,
      0.6,
      0.9,
      1.2,
      0.5,
    );
    createPart(
      "Right Chest",
      externalGroup,
      false,
      0.8,
      1.8,
      0.6,
      0.9,
      1.2,
      0.5,
    );
    createPart(
      "Left Upper Back",
      externalGroup,
      false,
      -0.8,
      1.8,
      -0.6,
      0.9,
      1.2,
      0.5,
    );
    createPart(
      "Right Upper Back",
      externalGroup,
      false,
      0.8,
      1.8,
      -0.6,
      0.9,
      1.2,
      0.5,
    );
    createPart(
      "Left Abdomen",
      externalGroup,
      false,
      -0.8,
      0,
      0.6,
      0.8,
      1.2,
      0.6,
    );
    createPart(
      "Right Abdomen",
      externalGroup,
      false,
      0.8,
      0,
      0.6,
      0.8,
      1.2,
      0.6,
    );
    createPart(
      "Left Lower Back",
      externalGroup,
      false,
      -0.8,
      0,
      -0.6,
      0.8,
      1.2,
      0.5,
    );
    createPart(
      "Right Lower Back",
      externalGroup,
      false,
      0.8,
      0,
      -0.6,
      0.8,
      1.2,
      0.5,
    );
    createPart(
      "Pelvis / Groin",
      externalGroup,
      false,
      0,
      -1.5,
      0.3,
      1.7,
      0.8,
      0.5,
    );
    createPart("Glutes", externalGroup, false, 0, -1.5, -0.3, 1.7, 0.8, 0.5);

    // Arms
    createPart(
      "Left Shoulder",
      externalGroup,
      false,
      -2.2,
      2.2,
      0,
      0.6,
      0.6,
      0.6,
    );
    createPart(
      "Right Shoulder",
      externalGroup,
      false,
      2.2,
      2.2,
      0,
      0.6,
      0.6,
      0.6,
    );
    createPart("Left Bicep", externalGroup, false, -2.5, 0.8, 0, 0.5, 1.2, 0.5);
    createPart("Right Bicep", externalGroup, false, 2.5, 0.8, 0, 0.5, 1.2, 0.5);
    createPart(
      "Left Forearm",
      externalGroup,
      false,
      -2.8,
      -1.0,
      0,
      0.4,
      1.0,
      0.4,
    );
    createPart(
      "Right Forearm",
      externalGroup,
      false,
      2.8,
      -1.0,
      0,
      0.4,
      1.0,
      0.4,
    );
    createPart("Left Hand", externalGroup, false, -2.9, -2.5, 0, 0.3, 0.6, 0.2);
    createPart("Right Hand", externalGroup, false, 2.9, -2.5, 0, 0.3, 0.6, 0.2);

    // Legs
    createPart(
      "Left Front Thigh",
      externalGroup,
      false,
      -0.8,
      -3.5,
      0.2,
      0.7,
      1.6,
      0.7,
    );
    createPart(
      "Right Front Thigh",
      externalGroup,
      false,
      0.8,
      -3.5,
      0.2,
      0.7,
      1.6,
      0.7,
    );
    createPart(
      "Left Hamstring",
      externalGroup,
      false,
      -0.8,
      -3.5,
      -0.2,
      0.7,
      1.6,
      0.7,
    );
    createPart(
      "Right Hamstring",
      externalGroup,
      false,
      0.8,
      -3.5,
      -0.2,
      0.7,
      1.6,
      0.7,
    );
    createPart(
      "Left Knee",
      externalGroup,
      false,
      -0.8,
      -5.2,
      0.3,
      0.6,
      0.4,
      0.6,
    );
    createPart(
      "Right Knee",
      externalGroup,
      false,
      0.8,
      -5.2,
      0.3,
      0.6,
      0.4,
      0.6,
    );
    createPart(
      "Left Calf",
      externalGroup,
      false,
      -0.8,
      -6.8,
      0.1,
      0.5,
      1.5,
      0.5,
    );
    createPart(
      "Right Calf",
      externalGroup,
      false,
      0.8,
      -6.8,
      0.1,
      0.5,
      1.5,
      0.5,
    );
    createPart(
      "Left Foot",
      externalGroup,
      false,
      -0.8,
      -8.5,
      0.5,
      0.4,
      0.3,
      0.8,
    );
    createPart(
      "Right Foot",
      externalGroup,
      false,
      0.8,
      -8.5,
      0.5,
      0.4,
      0.3,
      0.8,
    );

    // --- BUILD INTERNAL ORGANS (With Distinct Colors) ---
    createPart(
      "Brain",
      internalGroup,
      true,
      0,
      4.6,
      0,
      0.6,
      0.7,
      0.6,
      ORG_BRAIN,
    );
    createPart(
      "Heart",
      internalGroup,
      true,
      -0.3,
      1.8,
      0.2,
      0.4,
      0.5,
      0.4,
      ORG_HEART,
    );
    createPart(
      "Left Lung",
      internalGroup,
      true,
      -0.8,
      1.8,
      0,
      0.5,
      0.9,
      0.4,
      ORG_LUNGS,
    );
    createPart(
      "Right Lung",
      internalGroup,
      true,
      0.8,
      1.8,
      0,
      0.5,
      0.9,
      0.4,
      ORG_LUNGS,
    );
    createPart(
      "Liver",
      internalGroup,
      true,
      0.4,
      0.6,
      0.2,
      0.8,
      0.5,
      0.5,
      ORG_LIVER,
    );
    createPart(
      "Stomach",
      internalGroup,
      true,
      -0.5,
      0.4,
      0.2,
      0.5,
      0.4,
      0.4,
      ORG_STOMACH,
    );
    createPart(
      "Kidneys",
      internalGroup,
      true,
      0,
      0.4,
      -0.3,
      0.8,
      0.3,
      0.3,
      ORG_KIDNEYS,
    );
    createPart(
      "Intestines",
      internalGroup,
      true,
      0,
      -0.6,
      0.1,
      1.0,
      0.8,
      0.5,
      ORG_INTESTINES,
    );

    internalGroup.visible = false;

    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    container.addEventListener("mousemove", onMouseMove);
    container.addEventListener("click", onClick);

    animate();
  }

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }

  // --- INTERACTION LOGIC ---
  function onMouseMove(event) {
    const rect = container.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const activeGroup = isInternalView
      ? internalGroup.children
      : externalGroup.children;
    const intersects = raycaster.intersectObjects(activeGroup);

    // Reset hover colors back to their specific base color or selected color
    allMeshes.forEach((mesh) => {
      if (!mesh.userData.isSelected) {
        mesh.material.color.setHex(mesh.userData.baseColor);
      } else {
        mesh.material.color.setHex(
          mesh.userData.isInternal ? COLOR_ORGAN_SELECT : COLOR_SKIN_SELECT,
        );
      }
    });

    if (intersects.length > 0) {
      const hit = intersects[0].object;

      // Apply the hover color without destroying the selection state
      if (!hit.userData.isSelected) {
        hit.material.color.setHex(
          hit.userData.isInternal ? COLOR_ORGAN_HOVER : COLOR_SKIN_HOVER,
        );
      }

      hoverLabel.innerText = hit.userData.name;
      hoverLabel.style.display = "block";
    } else {
      hoverLabel.style.display = "none";
    }
  }

  function onClick(event) {
    const rect = container.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const activeGroup = isInternalView
      ? internalGroup.children
      : externalGroup.children;
    const intersects = raycaster.intersectObjects(activeGroup);

    if (intersects.length > 0) {
      const hit = intersects[0].object;
      hit.userData.isSelected = !hit.userData.isSelected;

      if (hit.userData.isSelected) {
        hit.material.color.setHex(
          hit.userData.isInternal ? COLOR_ORGAN_SELECT : COLOR_SKIN_SELECT,
        );
        selectedParts.add(hit.userData.name);
      } else {
        hit.material.color.setHex(hit.userData.baseColor);
        selectedParts.delete(hit.userData.name);
      }
      updateTagsUI();
    }
  }

  function updateTagsUI() {
    if (selectedParts.size === 0) {
      tagsContainer.innerHTML = `<span style="font-size: 0.85rem; color: #94a3b8; font-style: italic;">No areas selected...</span>`;
      return;
    }
    tagsContainer.innerHTML = "";
    selectedParts.forEach((part) => {
      tagsContainer.innerHTML += `<div class="symptom-tag">🎯 ${part}</div>`;
    });
  }

  // --- TOGGLE LOGIC ---
  function setInternalView(active) {
    isInternalView = active;
    if (active) {
      btnInternal.classList.add("active");
      btnInternal.style.background = "white";
      btnInternal.style.color = "var(--brand-dark)";
      btnInternal.style.boxShadow = "0 2px 4px rgba(0,0,0,0.05)";

      btnExternal.classList.remove("active");
      btnExternal.style.background = "transparent";
      btnExternal.style.color = "#64748b";
      btnExternal.style.boxShadow = "none";

      internalGroup.visible = true;
      externalGroup.children.forEach((mesh) => {
        mesh.material.opacity = 0.1;
        mesh.material.depthWrite = false;
      });
    } else {
      btnExternal.classList.add("active");
      btnExternal.style.background = "white";
      btnExternal.style.color = "var(--brand-dark)";
      btnExternal.style.boxShadow = "0 2px 4px rgba(0,0,0,0.05)";

      btnInternal.classList.remove("active");
      btnInternal.style.background = "transparent";
      btnInternal.style.color = "#64748b";
      btnInternal.style.boxShadow = "none";

      internalGroup.visible = false;
      externalGroup.children.forEach((mesh) => {
        mesh.material.opacity = 1;
        mesh.material.depthWrite = true;
      });
    }
  }

  btnExternal.addEventListener("click", () => setInternalView(false));
  btnInternal.addEventListener("click", () => setInternalView(true));

  // --- MODAL CONTROLS ---
  if (btnOpenMap) {
    btnOpenMap.addEventListener("click", () => {
      mapModal.classList.remove("hidden");
      if (!scene) init3D();
    });
  }

  if (btnCloseMap) {
    btnCloseMap.addEventListener("click", () => {
      mapModal.classList.add("hidden");
    });
  }

  if (btnConfirm) {
    btnConfirm.addEventListener("click", () => {
      if (selectedParts.size > 0) {
        const inputArea = document.getElementById("user-input");
        const symptomsText = Array.from(selectedParts).join(", ");

        if (inputArea.value.trim() === "") {
          inputArea.value =
            "I am experiencing pain/issues in my: " + symptomsText;
        } else {
          inputArea.value += ", and my: " + symptomsText;
        }

        inputArea.style.height = "auto";
        inputArea.style.height = inputArea.scrollHeight + "px";

        // Reset selections
        selectedParts.clear();
        allMeshes.forEach((mesh) => {
          mesh.userData.isSelected = false;
          mesh.material.color.setHex(mesh.userData.baseColor);
        });
        updateTagsUI();
      }
      mapModal.classList.add("hidden");
      document.getElementById("user-input").focus();
    });
  }
});
