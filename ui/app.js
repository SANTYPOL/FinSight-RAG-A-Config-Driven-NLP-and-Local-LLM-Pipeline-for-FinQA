const form = document.getElementById("chat-form");
const questionInput = document.getElementById("question-input");
const messages = document.getElementById("messages");
const sourcesSection = document.getElementById("sources");
const sourceList = document.getElementById("source-list");
const sourceCount = document.getElementById("source-count");
const statusText = document.getElementById("status");
const sendButton = document.getElementById("send-button");
const typingIndicator = document.getElementById("typing-indicator");
const historyList = document.getElementById("history-list");
const clearHistoryButton = document.getElementById("clear-history");
const copyLastAnswerButton = document.getElementById("copy-last-answer");
const companyFilter = document.getElementById("company-filter");
const yearFilter = document.getElementById("year-filter");
const presetButtons = document.querySelectorAll(".preset");
const HISTORY_KEY = "nse_finance_chat_history";
let lastAnswer = "";

function appendMessage(role, text, options = {}) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const wrapper = document.createElement("div");
  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const paragraph = document.createElement("p");
  paragraph.textContent = text;

  bubble.appendChild(paragraph);
  wrapper.appendChild(bubble);

  if (role === "assistant" && options.copyable) {
    const toolbar = document.createElement("div");
    toolbar.className = "message-toolbar";

    const copyButton = document.createElement("button");
    copyButton.className = "copy-button";
    copyButton.type = "button";
    copyButton.textContent = "Copy";
    copyButton.addEventListener("click", async () => {
      await navigator.clipboard.writeText(text);
      copyButton.textContent = "Copied";
      setTimeout(() => {
        copyButton.textContent = "Copy";
      }, 1200);
    });

    toolbar.appendChild(copyButton);
    wrapper.appendChild(toolbar);
  }

  article.appendChild(wrapper);
  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
}

function renderSources(sources = []) {
  sourceList.innerHTML = "";
  if (!sources.length) {
    sourcesSection.classList.add("hidden");
    return;
  }

  sourcesSection.classList.remove("hidden");
  sourceCount.textContent = `${sources.length} item(s)`;

  sources.forEach((source, index) => {
    const card = document.createElement("article");
    card.className = "source-card";

    const topline = document.createElement("div");
    topline.className = "source-topline";

    const title = document.createElement("h4");
    title.textContent = `${index + 1}. ${source.question}`;

    const score = document.createElement("p");
    score.className = "source-score";
    score.textContent = `Score: ${source.score ?? "n/a"}`;

    topline.appendChild(title);
    topline.appendChild(score);

    const meta = document.createElement("p");
    meta.className = "source-meta";
    meta.textContent = `${source.symbol} | ${source.category} | ${source.source_section || "General"} | Pages ${source.source_pages || "n/a"}`;

    const answer = document.createElement("p");
    answer.textContent = `Answer: ${source.answer}`;

    const context = document.createElement("p");
    context.className = "source-meta";
    context.textContent = `Context hint: ${source.context_hint || "None"}`;

    card.appendChild(topline);
    card.appendChild(meta);
    card.appendChild(answer);
    card.appendChild(context);
    sourceList.appendChild(card);
  });
}

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveHistory(history) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 12)));
}

function renderHistory() {
  const history = loadHistory();
  historyList.innerHTML = "";

  if (!history.length) {
    const empty = document.createElement("p");
    empty.className = "source-meta";
    empty.textContent = "No saved conversations yet.";
    historyList.appendChild(empty);
    return;
  }

  history.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "history-item";
    button.innerHTML = `<strong>${item.question}</strong><span>${item.answer}</span>`;
    button.addEventListener("click", () => {
      questionInput.value = item.question;
      questionInput.focus();
    });
    historyList.appendChild(button);
  });
}

function addHistory(question, answer) {
  const history = loadHistory();
  history.unshift({ question, answer });
  saveHistory(history);
  renderHistory();
}

function buildFilteredQuestion(question) {
  const company = companyFilter.value;
  const year = yearFilter.value;
  const parts = [];
  if (company) {
    parts.push(`Company: ${company}`);
  }
  if (year) {
    parts.push(`Financial Year: ${year}`);
  }
  parts.push(`Question: ${question}`);
  return parts.join("\n");
}

function setLoading(isLoading) {
  typingIndicator.classList.toggle("hidden", !isLoading);
  sendButton.disabled = isLoading;
  copyLastAnswerButton.disabled = isLoading || !lastAnswer;
  statusText.textContent = isLoading ? "Thinking..." : "Ready";
}

async function askQuestion(question) {
  appendMessage("user", question);
  setLoading(true);

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        config_path: "config.yaml",
        query: buildFilteredQuestion(question),
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Request failed");
    }

    lastAnswer = payload.answer || "No answer returned.";
    appendMessage("assistant", lastAnswer, { copyable: true });
    renderSources(payload.sources || []);
    addHistory(question, lastAnswer);
  } catch (error) {
    appendMessage("assistant", `There was a problem: ${error.message}`);
    renderSources([]);
    statusText.textContent = "Error";
  } finally {
    setLoading(false);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) {
    return;
  }

  questionInput.value = "";
  await askQuestion(question);
});

presetButtons.forEach((button) => {
  button.addEventListener("click", async () => {
    const question = button.dataset.question;
    questionInput.value = question;
    await askQuestion(question);
  });
});

clearHistoryButton.addEventListener("click", () => {
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
});

copyLastAnswerButton.addEventListener("click", async () => {
  if (!lastAnswer) {
    return;
  }
  await navigator.clipboard.writeText(lastAnswer);
  copyLastAnswerButton.textContent = "Copied";
  setTimeout(() => {
    copyLastAnswerButton.textContent = "Copy Last Answer";
  }, 1200);
});

renderHistory();
copyLastAnswerButton.disabled = true;
