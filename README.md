# Med Deck Forge — MVP

Minimal, single-page prototype focused on a beautiful “one prompt → deck → Anki-style review” loop. It simulates streaming generation and review without requiring a backend.

## What it does
- Command bar: type “40 cards for EM ITE high yield,” click **Generate**.
- Live build animation: counter, progress bar, placeholders resolving into cards.
- Anki-style reviewer: space to flip; 1/2/3/4 for Again/Hard/Good/Easy; Enter to advance; arrows to move.
- Quick metadata chips: parsed exam, abstraction, count, topics.
- Magic edit (simulated): apply a single edit prompt to all card backs.

## How to run
This is static—no build step required.

```bash
python -m http.server 3000
```

Open http://localhost:3000 to use the prototype.

## Notes for a real backend
- Replace the simulated generator with `POST /api/decks/generate` returning structured JSON (front, back, tags, yield_score?).
- Add `POST /api/decks/:id/edit` to transform all cards with an edit prompt.
- Enforce structured outputs (OpenAI function calling / JSON schema) to keep the UI stable while streaming.

## Safety
Content is placeholder; verify medical details with authoritative sources.
