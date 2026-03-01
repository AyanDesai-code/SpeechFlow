const transcriptScreen = document.getElementById("screen-transcript");
const loadingScreen = document.getElementById("screen-loading");
const finalScreen = document.getElementById("screen-final");

const transcriptList = document.getElementById("transcriptList");
const practiceList = document.getElementById("practiceList");
const summaryText = document.getElementById("summaryText");

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

function showScreen(target) {
  [transcriptScreen, loadingScreen, finalScreen].forEach((screen) => {
    screen.classList.remove("active");
  });
  target.classList.add("active");
}

function renderTranscript() {
  transcriptList.innerHTML = "";
  demoTranscript.forEach((line) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="who">${line.role}</span>${line.text}`;
    transcriptList.append(li);
  });
}

function renderFinal() {
  practiceList.innerHTML = "";

  Object.entries(demoPracticeWords)
    .sort((a, b) => b[1] - a[1])
    .forEach(([word, count]) => {
      const li = document.createElement("li");
      li.textContent = `${word} (x${count})`;
      practiceList.append(li);
    });

  summaryText.textContent =
    "Fantastic effort today! Say each highlighted word slowly and confidently, then celebrate each clean try — small wins add up fast.";
}

toLoadingBtn.addEventListener("click", () => {
  showScreen(loadingScreen);
  setTimeout(() => {
    renderFinal();
    showScreen(finalScreen);
  }, 2200);
});

restartBtn.addEventListener("click", () => {
  renderTranscript();
  showScreen(transcriptScreen);
});

renderTranscript();
showScreen(transcriptScreen);
