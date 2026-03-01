const transcriptScreen = document.getElementById("screen-transcript");
const loadingScreen = document.getElementById("screen-loading");
const finalScreen = document.getElementById("screen-final");

const transcriptList = document.getElementById("transcriptList");
const practiceList = document.getElementById("practiceList");
const summaryText = document.getElementById("summaryText");
const backendStatus = document.getElementById("backendStatus");
const transcriptHint = document.getElementById("transcriptHint");

const startConversationBtn = document.getElementById("startConversationBtn");
const toLoadingBtn = document.getElementById("toLoadingBtn");
const restartBtn = document.getElementById("restartBtn");

const demoTranscript = [
  { role: "User", text: "I really really like playing football." },
  { role: "Coach", text: "Love that! What part of the game is your favorite?" },
  { role: "User", text: "I like midfield because because I can control play." },
  { role: "Coach", text: "Awesome detail — you are explaining this really well!" },
  { role: "User", text: "Sometimes I repeat words when I get excited." }
];

const demoPracticeWords = {
  really: 3,
  because: 3,
  midfield: 2,
  excited: 2
};

const apiBase = "";
let conversationStarted = false;
let transcriptTimer = null;
let currentTranscript = [];

function showScreen(target) {
  [transcriptScreen, loadingScreen, finalScreen].forEach((screen) => {
    screen.classList.remove("active");
  });
  target.classList.add("active");
}

function renderTranscript(lines = []) {
  transcriptList.innerHTML = "";
  lines.forEach((line) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="who">${line.role}</span>${line.text}`;
    transcriptList.append(li);
  });
}

function resetTranscriptState() {
  if (transcriptTimer) {
    clearInterval(transcriptTimer);
    transcriptTimer = null;
  }

  conversationStarted = false;
  currentTranscript = [];
  renderTranscript(currentTranscript);

  transcriptHint.textContent = "Tap Start Conversation to begin live transcript.";
  startConversationBtn.disabled = false;
  toLoadingBtn.disabled = true;
}

function streamDemoTranscript() {
  let idx = 0;

  const pushNextLine = () => {
    if (idx >= demoTranscript.length) {
      if (transcriptTimer) {
        clearInterval(transcriptTimer);
        transcriptTimer = null;
      }
      toLoadingBtn.disabled = false;
      transcriptHint.textContent = "Conversation captured. Tap Finish Conversation to analyze.";
      return;
    }

    currentTranscript.push(demoTranscript[idx]);
    renderTranscript(currentTranscript);
    transcriptList.scrollTop = transcriptList.scrollHeight;
    idx += 1;

    if (idx >= demoTranscript.length) {
      toLoadingBtn.disabled = false;
      transcriptHint.textContent = "Conversation captured. Tap Finish Conversation to analyze.";
      if (transcriptTimer) {
        clearInterval(transcriptTimer);
        transcriptTimer = null;
      }
    }
  };

  pushNextLine();
  transcriptTimer = setInterval(pushNextLine, 700);
}

function renderFinal(result) {
  const practiceWords = result?.practiceWords || demoPracticeWords;
  const summary =
    result?.summary ||
    "Strong effort today. Repeat each highlighted word slowly and clearly for your next round.";

  practiceList.innerHTML = "";

  Object.entries(practiceWords)
    .sort((a, b) => b[1] - a[1])
    .forEach(([word, count]) => {
      const li = document.createElement("li");
      li.textContent = `${word} (x${count})`;
      practiceList.append(li);
    });

  summaryText.textContent = summary;
}

async function checkBackend() {
  try {
    const res = await fetch(`${apiBase}/api/health`, { method: "GET" });
    if (!res.ok) throw new Error("unhealthy");
    const payload = await res.json();
    backendStatus.textContent = payload.mode === "live" ? "Backend connected" : "Demo mode";
  } catch (err) {
    backendStatus.textContent = "Demo mode";
  }
}

async function analyzeTranscriptWithBackend() {
  const payload = { transcript: currentTranscript.length ? currentTranscript : demoTranscript };

  const res = await fetch(`${apiBase}/api/analyze-transcript`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!res.ok) {
    throw new Error(`Backend error: ${res.status}`);
  }

  return res.json();
}

startConversationBtn.addEventListener("click", () => {
  if (conversationStarted) {
    return;
  }

  conversationStarted = true;
  startConversationBtn.disabled = true;
  transcriptHint.textContent = "Conversation started… capturing lines now.";
  streamDemoTranscript();
});

toLoadingBtn.addEventListener("click", async () => {
  showScreen(loadingScreen);

  try {
    const result = await analyzeTranscriptWithBackend();
    renderFinal(result);
  } catch (err) {
    renderFinal();
  }

  setTimeout(() => {
    showScreen(finalScreen);
  }, 900);
});

restartBtn.addEventListener("click", () => {
  resetTranscriptState();
  showScreen(transcriptScreen);
});

resetTranscriptState();
showScreen(transcriptScreen);
checkBackend();
