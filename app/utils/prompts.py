import json
from datetime import datetime

from utils.vectordb import fetch_vocab_from_vector_db

def get_listening_prompt(topic, level="A1", item_id_start=1, prefer_type="MultipleChoice"):
    """Listening-specific generation prompt that returns an ARRAY of 10 items."""
    
    vocab_list = fetch_vocab_from_vector_db(topic, level)

    return f"""
Task:
Generate EXACTLY 10 listening comprehension items for CEFR {level}.
Each item must be of type "{prefer_type}" unless clearly unsuitable, then use "RichtigFalsch".
The output MUST be a SINGLE JSON ARRAY with 10 objects. No text before or after the JSON.

Inputs:
- vocab_list: {json.dumps(vocab_list, ensure_ascii=False)}
- topic: "{topic}"
- start_id: {item_id_start}
- max_audio_length: 12 seconds

JSON ARRAY STRUCTURE (exact):
[
  {{
    "id": integer,
    "type": "MultipleChoice" | "RichtigFalsch",
    "question": string,
    "translation": string,
    "audioText": string,
    "audioText_translation": string,
    "audioDescription": string,
    "ttsPrompt": string,
    "options": [string],
    "options_translations": [string],
    "correctAnswer": string,
    "imagePlaceholder": string,
    "metadata": {{
        "level": "{level}",
        "skill": "LISTENING",
        "topic": "{topic}",
        "source": "generated",
        "timestamp": "{datetime.now().isoformat()}"
    }}
  }},
  ...
]  <-- exactly 10 objects
  

CRITICAL REQUIREMENTS:
- Start IDs at {item_id_start} and increment sequentially.
- Each audioText must include at least ONE word from vocab_list.
- audioText must be SIMPLE A1 German (max 15 words).
- distractors must be realistic (e.g., similar times, similar places).
- options MUST contain 3 items for MultipleChoice, 2 for RichtigFalsch.
- correctAnswer MUST be EXACTLY one of the options.
- No explanations, no prose, no markdown — ONLY the JSON array.

Content Rules:
- Use daily-life contexts: Bahnhof, Bus, Supermarkt, Café, Arbeit, Wetter, Termine.
- Use short, natural, realistic announcements or dialogues.
- Avoid proper nouns except common German cities (Berlin, Hamburg, München).

Return ONLY the JSON array with 10 objects.
"""


if __name__ == "__main__":
    prompt = get_listening_prompt("food", "B2")
    print(prompt)