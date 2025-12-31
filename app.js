const promptInput = document.getElementById('promptInput');
const generateBtn = document.getElementById('generateBtn');
const metaChips = document.getElementById('metaChips');
const buildStatus = document.getElementById('buildStatus');
const builtCount = document.getElementById('builtCount');
const targetCount = document.getElementById('targetCount');
const progressBar = document.getElementById('progressBar').firstElementChild;
const buildList = document.getElementById('buildList');
const reviewCard = document.getElementById('reviewCard');
const cardFront = document.getElementById('cardFront');
const cardBack = document.getElementById('cardBack');
const cardOverlay = document.getElementById('cardOverlay');
const reviewMeta = document.getElementById('reviewMeta');
const editInput = document.getElementById('editInput');
const editBtn = document.getElementById('editBtn');

let deck = [];
let currentIndex = 0;
let isFlipped = false;
let buildTimer = null;

const examAliases = [
  { key: 'EM ITE', patterns: ['em ite', 'emergency medicine ite', '2nd year em', 'em inservice'] },
  { key: 'USMLE Step 1', patterns: ['step 1', 'usmle 1', 'step one'] },
  { key: 'USMLE Step 2', patterns: ['step 2', 'usmle 2', 'step two', 'ck'] },
  { key: 'USMLE Step 3', patterns: ['step 3', 'usmle 3', 'step three'] },
  { key: 'NSGY Boards', patterns: ['nsgy', 'neurosurgery', 'skull base'] }
];

const abstractionAliases = [
  { key: 'Super High Yield', patterns: ['super high', 'ultra high', 'highest yield'] },
  { key: 'High Yield', patterns: ['high yield', 'high-yield'] },
  { key: 'Concise', patterns: ['concise', 'overview', 'basic'] },
  { key: 'Deep Dive', patterns: ['deep', 'dive'] }
];

function parsePrompt(text) {
  const lower = text.toLowerCase();
  const countMatch = text.match(/(\d{1,3})\s*cards?/i);
  const count = countMatch ? Number(countMatch[1]) : 40;

  const exam = examAliases.find((e) => e.patterns.some((p) => lower.includes(p)))?.key || 'Custom Exam';
  const abstraction = abstractionAliases.find((a) => a.patterns.some((p) => lower.includes(p)))?.key || 'High Yield';

  return {
    count,
    exam,
    abstraction,
    topics: extractTopics(lower)
  };
}

function extractTopics(text) {
  const cleaned = text
    .replace(/\b(em|emergency|medicine|ite|step|usmle|boards|exam|cards|for|year)\b/gi, '')
    .replace(/\s+/g, ' ')
    .trim();
  if (!cleaned) return [];
  return cleaned.split(',').map((t) => t.trim()).filter(Boolean);
}

function showChips(meta) {
  metaChips.innerHTML = '';
  const entries = [
    `Exam: ${meta.exam}`,
    `Abstraction: ${meta.abstraction}`,
    `Count: ${meta.count}`,
    meta.topics.length ? `Topics: ${meta.topics.join(', ')}` : null
  ].filter(Boolean);
  entries.forEach((text) => {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = text;
    metaChips.appendChild(chip);
  });
}

function buildDeck(meta) {
  clearInterval(buildTimer);
  deck = [];
  currentIndex = 0;
  isFlipped = false;
  updateReviewCard();

  buildStatus.textContent = 'Generating…';
  builtCount.textContent = '0';
  targetCount.textContent = String(meta.count);
  progressBar.style.width = '0%';
  buildList.innerHTML = '';

  const placeholders = Array.from({ length: meta.count }, (_, i) => createCardShell(i + 1, true));
  placeholders.forEach((node) => buildList.appendChild(node));

  let produced = 0;
  buildTimer = setInterval(() => {
    if (produced >= meta.count) {
      clearInterval(buildTimer);
      buildStatus.textContent = 'Deck ready';
      return;
    }
    const card = generateCard(meta, produced);
    deck.push(card);
    resolveCardShell(produced, card);
    produced += 1;
    builtCount.textContent = String(produced);
    progressBar.style.width = `${(produced / meta.count) * 100}%`;
    if (produced === 1) updateReviewCard();
  }, 40);
}

function createCardShell(index, loading = false) {
  const node = document.createElement('div');
  node.className = 'card-shell' + (loading ? ' loading' : '');
  node.dataset.index = String(index - 1);
  node.innerHTML = `
    <h4>Card ${index}</h4>
    <p>${loading ? 'Streaming…' : ''}</p>
  `;
  return node;
}

function resolveCardShell(position, card) {
  const shell = buildList.querySelector(`.card-shell[data-index="${position}"]`);
  if (!shell) return;
  shell.classList.remove('loading');
  shell.innerHTML = `
    <h4>${card.front}</h4>
    <p>${card.back}</p>
  `;
}

function generateCard(meta, i) {
  const topic = meta.topics[i % Math.max(1, meta.topics.length)] || 'Core';
  return {
    id: `${meta.exam}-${i}`,
    front: `${meta.exam} — ${topic} #${i + 1}`,
    back: `${meta.abstraction} takeaway about ${topic}. Expand with nuances and pearls.`,
    yield_score: null,
    tags: meta.topics.length ? [topic] : []
  };
}

function updateReviewCard() {
  const card = deck[currentIndex];
  if (!card) {
    cardFront.textContent = 'No deck yet.';
    cardBack.textContent = '';
    cardOverlay.textContent = 'Generate a deck to start reviewing';
    reviewMeta.textContent = 'No deck loaded.';
    reviewCard.classList.remove('flipped');
    return;
  }
  cardFront.innerHTML = `<h3>${card.front}</h3>`;
  cardBack.innerHTML = `<p>${card.back}</p>`;
  cardOverlay.textContent = 'Flip with space';
  reviewMeta.textContent = `Card ${currentIndex + 1}/${deck.length}`;
  reviewCard.classList.toggle('flipped', isFlipped);
}

function flipCard() {
  if (!deck.length) return;
  isFlipped = !isFlipped;
  reviewCard.classList.toggle('flipped', isFlipped);
}

function gradeCard(grade) {
  if (!deck.length) return;
  // In MVP we just move to next card.
  nextCard();
}

function nextCard() {
  if (!deck.length) return;
  currentIndex = (currentIndex + 1) % deck.length;
  isFlipped = false;
  updateReviewCard();
}

function applyEdit() {
  const prompt = editInput.value.trim();
  if (!prompt || !deck.length) return;
  deck = deck.map((card) => ({
    ...card,
    back: `${card.back} (${prompt})`
  }));
  updateReviewCard();
}

function handleKeys(e) {
  if (e.code === 'Space') {
    e.preventDefault();
    flipCard();
  }
  if (e.code === 'Enter') {
    nextCard();
  }
  if (['Digit1', 'Digit2', 'Digit3', 'Digit4'].includes(e.code)) {
    const gradeMap = { Digit1: 'again', Digit2: 'hard', Digit3: 'good', Digit4: 'easy' };
    gradeCard(gradeMap[e.code]);
  }
  if (['ArrowLeft', 'ArrowRight'].includes(e.code)) {
    const delta = e.code === 'ArrowRight' ? 1 : -1;
    if (deck.length) {
      currentIndex = (currentIndex + delta + deck.length) % deck.length;
      isFlipped = false;
      updateReviewCard();
    }
  }
}

function wireEvents() {
  generateBtn.addEventListener('click', () => {
    const text = promptInput.value.trim();
    if (!text) return;
    const meta = parsePrompt(text);
    showChips(meta);
    buildDeck(meta);
  });

  document.querySelectorAll('.review-controls .btn').forEach((btn) => {
    btn.addEventListener('click', () => gradeCard(btn.dataset.grade));
  });

  reviewCard.addEventListener('click', flipCard);
  editBtn.addEventListener('click', applyEdit);
  document.addEventListener('keydown', handleKeys);
}

wireEvents();
updateReviewCard();
