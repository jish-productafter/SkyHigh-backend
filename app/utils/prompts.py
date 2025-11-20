import json
import logging
from datetime import datetime

from utils.vectordb import fetch_vocab_from_vector_db

# Set up logger
logger = logging.getLogger(__name__)

def get_listening_prompt(topic, level="A1", item_id_start=1, prefer_type="MultipleChoice"):
    """Listening-specific generation prompt that returns an ARRAY of 10 items."""
    logger.info(f"Generating listening prompt: topic='{topic}', level={level}, item_id_start={item_id_start}, prefer_type={prefer_type}")
    
    logger.debug(f"Fetching vocabulary for topic '{topic}' at level {level}")
    vocab_list = fetch_vocab_from_vector_db(topic, level)
    logger.info(f"Retrieved {len(vocab_list)} vocabulary items for prompt generation")
    logger.debug(f"Vocabulary items: {vocab_list}")
    
    level_rules = {
        "A1": "CRITICAL GRAMMAR: Present tense, Imperative, Nominative/Accusative/Dative cases. Simple Sentences with 'und, aber, oder, sondern, denn, dann'.\nTOPICS: Personal info, daily routine, time (official/unofficial), directions, simple announcements (Bahnhof/Bus). MAX AUDIO LENGTH: 12 seconds. Audio must be SIMPLE A1 German (max 15 words). (Source A1 Plan)",
        "A2": "CRITICAL GRAMMAR: Perfekt/Präteritum (Auxiliary/Modal verbs), Reflexive verbs, Passiv Präsens. Basic Subordinate clauses with 'weil, wenn, dass'.\nTOPICS: Giving reasons, reporting travel, discussing home/neighborhood, health, education/career, past life events. MAX AUDIO LENGTH: 15-20 seconds. Audio can contain past tenses and basic subordinate clauses. (Source A2 Plan)",
        "B1": "CRITICAL GRAMMAR: Futur I, Konjunktiv II (Irreale Wunsch/Bedingungssätze), Konjunktiv I (reported speech), Passiv Präsens/Perfekt, Genitiv Relativsätze. Subordinate clauses: Concessive (obwohl), Consecutive (so dass), Final (um...zu, damit).\nTOPICS: Opinions, consensus-finding, advice, job application/interview, statistics, political/historical topics. MAX AUDIO LENGTH: 20-25 seconds. Audio must be clear B1 German with complex structures. (Source B1 Plan)",
        "B2": "CRITICAL GRAMMAR: Direkte/Indirekte Rede with Konjunktiv I/Umformung. Passiv in allen Zeiten. Complex Relativsätze with Präpositional-adverbien. Extensive use of Modalpartikels.\nTOPICS: Non-verbal communication, negative criticism, analyzing CVs, abstract discussions (violence, poverty, internet addiction, science, history). MAX AUDIO LENGTH: 25-30 seconds. Audio must be sophisticated, formal, and complex B2 German. (Source B2 Plan)",
    }.get(level, "Use general CEFR rules for this level.")


    prompt = f"""
Task:
Generate EXACTLY 10 listening comprehension items for CEFR {level}.
Each item must be of type "{prefer_type}" unless clearly unsuitable, then use "RichtigFalsch".
The output MUST be a SINGLE JSON ARRAY with 5 objects. No text before or after the JSON.

Inputs:
- vocab_list: {json.dumps(vocab_list, ensure_ascii=False)}
- topic: "{topic}"
- start_id: {item_id_start}

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
]  <-- exactly 5 objects
  

CRITICAL REQUIREMENTS:
- Start IDs at {item_id_start} and increment sequentially.
- Each audioText must include at least ONE word from vocab_list.
- Distractors must be realistic (e.g., similar times, similar places).
- options MUST contain 3 items for MultipleChoice, 2 for RichtigFalsch.
- correctAnswer MUST be EXACTLY one of the options.
- No explanations, no prose, no markdown — ONLY the JSON array.
- Perfectly follow the JSON array structure.
- The output MUST be valid JSON.
- The JSON MUST be a single array with EXACTLY 5 objects.
- The JSON MUST contain:
    * no trailing commas
    * no comments
    * all keys in double-quotes
    * all strings in double-quotes
- The output MUST be parsable by JSON.parse() without errors.


Content Rules:
- Use short, natural, realistic announcements or dialogues.
- Avoid proper nouns except common German cities (Berlin, Hamburg, München).

CEFR Level Specific Rules:
{level_rules}

Return ONLY the JSON array with 5 objects.
"""
    logger.debug(f"Generated prompt (length: {len(prompt)} characters)")
    logger.info("Listening prompt generated successfully")
    return prompt


def get_reading_prompt(topic, level="A1", item_id_start=1, prefer_type="MultipleChoice"):
    """Reading-specific generation prompt that returns an ARRAY of 8 items."""
    logger.info(f"Generating reading prompt: topic='{topic}', level={level}, item_id_start={item_id_start}, prefer_type={prefer_type}")
    
    logger.debug(f"Fetching vocabulary for topic '{topic}' at level {level}")
    vocab_list = fetch_vocab_from_vector_db(topic, level)
    logger.info(f"Retrieved {len(vocab_list)} vocabulary items for prompt generation")
    logger.debug(f"Vocabulary items: {vocab_list}")

    level_rules = {
        "A1": "CRITICAL GRAMMAR: Present tense, Nominative/Accusative/Dative (articles). Conjunctions: 'und, aber, oder, sondern, denn, dann'.\nTEXT TYPES: Postcards, simple messages, announcements, timetables, short forms (max 80 words).\nQUESTIONS: Test basic facts, location, time, and main idea. (Source A1 Plan)",
        "A2": "CRITICAL GRAMMAR: Perfekt/Präteritum/Plusquamperfekt (Auxiliary/Modal verbs). Reflexive verbs, Passiv Präsens. Subordinate clauses: 'weil, wenn, dass'.\nTEXT TYPES: Emails about travel/weekend, simple articles, small ads, health tips (max 80 words). Questions: Test specific details, events, reasons, and simple inference. (Source A2 Plan)",
        "B1": "CRITICAL GRAMMAR: Futur I, Konjunktiv II, Konjunktiv I (reported speech), Passiv Präsens/Perfekt, Genitiv Relativsätze. Subordinate clauses: Concessive, Consecutive, Final. Two-part conjunctions.\nTEXT TYPES: Letters of complaint/inquiry, job application materials, statistics reports, newspaper articles (max 150 words). Questions: Test main idea, author's suggestion, purpose of text, and moderate inference. (Source B1 Plan)",
        "B2": "CRITICAL GRAMMAR: Direkte/Indirekte Rede with Konjunktiv I/Umformung. Passiv in allen Zeiten. Complex Relativsätze with Präpositional-adverbien. Modalpartikels. Formal/Abstract Discourse.\nTEXT TYPES: Formal reports, complex articles on science/politics/history/sociology, nuanced critiques, analyses of CVs (max 150 words). Questions: Test complex inference, implied meaning, author's stance, and detailed analysis of argument structure. (Source B2 Plan)",
    }.get(level, "Use general CEFR rules for this level.")


    prompt = f"""
Task:
Generate EXACTLY 8 reading comprehension items for CEFR {level}.
Each item must be of type "{prefer_type}" unless clearly unsuitable, then use "RichtigFalsch".
The output MUST be a SINGLE JSON ARRAY with 5 objects. No text before or after the JSON.


Inputs:
- vocab_list: {json.dumps(vocab_list, ensure_ascii=False)}
- topic: "{topic}"
- start_id: {item_id_start}
- max_passage_length: 80 words for A1/A2, 150 words for B1/B2


JSON ARRAY STRUCTURE (exact):
[
  {{
    "id": integer,
    "type": "MultipleChoice" | "RichtigFalsch",
    "question": string,
    "question_translation": string,
    "passage": string,
    "passage_translation": string,
    "options": [string],
    "options_translations": [string],
    "correctAnswer": string,
    "imagePlaceholder": string,
    "metadata": {{
        "level": "{level}",
        "skill": "READING",
        "topic": "{topic}",
        "source": "generated",
        "timestamp": "{datetime.now().isoformat()}",
        "passage_word_count": integer
    }}
  }},
  ...
]  <-- exactly 5 objects


CRITICAL REQUIREMENTS:
- Start IDs at {item_id_start} and increment sequentially.
- Passage MUST include at least TWO words from vocab_list.
- Passage must use simple, natural German appropriate for level {level}.
- options MUST contain 3 items for MultipleChoice, 2 for RichtigFalsch.
- correctAnswer MUST be EXACTLY one of the options.
- Distractors must be plausible (similar to correct answer but incorrect).
- No explanations, no prose, no markdown — ONLY the JSON array.
- Perfectly follow the JSON array structure.
- The output MUST be valid JSON.
- The JSON MUST be a single array with EXACTLY 5 objects.
- The JSON MUST contain:
    * no trailing commas
    * no comments
    * all keys in double-quotes
    * all strings in double-quotes
- The output MUST be parsable by JSON.parse() without errors.

Content Rules:
- Use realistic texts: emails, postcards, advertisements, simple articles, schedules.
- Avoid proper nouns except common German cities (Berlin, Hamburg, München).
- Passages should be self-contained and understandable from context.


CEFR Level Specific Rules and Question Types:
{level_rules}


Return ONLY the JSON array with 5 objects.
"""
    logger.debug(f"Generated prompt (length: {len(prompt)} characters)")
    logger.info("Reading prompt generated successfully")
    return prompt









def get_writing_prompt(topic, level="A1", item_id_start=1, task_type="email"):
    """Writing-specific generation prompt that returns an ARRAY of 5 writing tasks."""
    logger.info(f"Generating writing prompt: topic='{topic}', level={level}, item_id_start={item_id_start}, task_type={task_type}")
    
    logger.debug(f"Fetching vocabulary for topic '{topic}' at level {level}")
    vocab_list = fetch_vocab_from_vector_db(topic, level)
    logger.info(f"Retrieved {len(vocab_list)} vocabulary items for prompt generation")
    logger.debug(f"Vocabulary items: {vocab_list}")
    
    level_guidelines = f"""
Task Guidelines by Level (Based on Study Plans):

A1 LEVEL:
- Task type: email, postcard, simple message, form filling.
- Focus: Can user write basic personal information and simple sentences?
- Content: greetings, personal details, day-to-day activities, basic preferences, simple directions.
- Grammar: Present tense, Nominative/Accusative/Dative, simple sentences with 'und, aber, oder'. Minimal errors are acceptable. (Source A1 Plan)
- Word count: 30-50 words.
- Evaluation: All 3 content points addressed? Mostly understandable? Basic grammar/spelling OK?

A2 LEVEL:
- Task type: email, postcard, simple letter, message.
- Focus: Can user write about familiar topics with some variety and complexity?
- Content: routines, past/future events (using Perfekt/Präteritum), simple opinions, giving reasons ('weil', 'dass').
- Grammar: Present + past tense (Perfekt, Präteritum, Plusquamperfekt), some complex sentences with basic subordinate clauses. (Source A2 Plan)
- Word count: 60-80 words.
- Evaluation: All 3 points covered? Uses past tense? Demonstrates use of basic complex structures? Mostly correct spelling/grammar?

B1 LEVEL:
- Task type: email, formal/informal letter, short article, blog post, report excerpt.
- Focus: Can user organize ideas, explain reasons, give advice, and discuss hypothetical situations?
- Content: opinions, reasons (using complex conjunctions like 'obwohl', 'so dass'), descriptions of experiences (using a range of past tenses), giving advice. Must address multiple parts of a topic.
- Grammar: Full range of tenses (Präteritum, Plusquamperfekt, Futur I), complex sentences with subordinate and infinitive clauses, use of Passiv Präsens/Perfekt, Konjunktiv II. (Source B1 Plan)
- Word count: 100-150 words.
- Evaluation: Ideas clearly organized? Reasons provided? Appropriate tone/register? Effective use of B1 grammar (e.g., subordinate clauses, Passiv)?

B2 LEVEL:
- Task type: formal letter, article, report, analysis, detailed critique.
- Focus: Can user argue, analyze, hypothesize, and write formally/informally with nuanced language and minimal errors?
- Content: complex opinions, analysis, hypothetical situations (Irreale Wünsche), formal requests, comparison of ideas, summarizing/transforming information (Direkte/Indirekte Rede). Topics are often abstract (science, politics, society).
- Grammar: Full range of tenses/moods (Konjunktiv I/II), complex structures, extensive Passiv, sophisticated vocabulary, correct use of Modalpartikels and Genitive prepositions. (Source B2 Plan)
- Word count: 150-250 words.
- Evaluation: Clear, well-structured argument? Appropriate, sophisticated register? Nuanced vocabulary? Minimal errors in complex B2 structures?
"""

    prompt = f"""
Task:
Generate EXACTLY 5 writing tasks for CEFR {level}.
The output MUST be a SINGLE JSON ARRAY with 5 objects. No text before or after the JSON.


Inputs:
- vocab_list: {json.dumps(vocab_list, ensure_ascii=False)}
- topic: "{topic}"
- start_id: {item_id_start}
- task_type: "{task_type}" (can be: email, postcard, letter, form, message, dialogue)


JSON ARRAY STRUCTURE (exact):
[
  {{
    "id": integer,
    "type": "{task_type}" | "postcard" | "letter" | "email" | "message" | "form",
    "task_title": string,
    "task_description": string,
    "task_instructions": string,
    "context": string,
    "context_translation": string,
    "content_points": [string, string, string],
    "word_count_min": integer,
    "word_count_max": integer,
    "example_response": string,
    "evaluation_criteria": {{
        "task_completion": string,
        "vocabulary_range": string,
        "grammar_complexity": string,
        "coherence_organization": string
    }},
    "metadata": {{
        "level": "{level}",
        "skill": "WRITING",
        "topic": "{topic}",
        "source": "generated",
        "timestamp": "{datetime.now().isoformat()}",
        "task_type": "{task_type}"
    }}
  }},
  ...
]  <-- exactly 5 objects


CRITICAL REQUIREMENTS:
- Start IDs at {item_id_start} and increment sequentially.
- Task must require at least 2-3 words from vocab_list.
- content_points: List 3 key information points that MUST be included in response.
- evaluation_criteria MUST define how to assess: task completion, vocabulary, grammar, coherence.
- No explanations, no prose, no markdown — ONLY the JSON array.
- Perfectly follow the JSON array structure.
- The output MUST be valid JSON.
- The JSON MUST be a single array with EXACTLY 5 objects.
- The JSON MUST contain:
    * no trailing commas
    * no comments
    * all keys in double-quotes
    * all strings in double-quotes
- The output MUST be parsable by JSON.parse() without errors.

Content Rules:
- Use realistic everyday scenarios: work, travel, shopping, family, hobbies, complaints, requests, opinions.
- Provide clear context and situation to make task meaningful.
- Ensure 3 content points are achievable within word limit.
- example_response should be well-written but concise.


{level_guidelines}


Return ONLY the JSON array with 5 objects.
"""
    logger.debug(f"Generated prompt (length: {len(prompt)} characters)")
    logger.info("Writing prompt generated successfully")
    return prompt




def get_speaking_prompt(topic, level="A1", item_id_start=1, interaction_type="interview"):
    """Speaking-specific generation prompt that returns an ARRAY of 6 speaking tasks."""
    logger.info(f"Generating speaking prompt: topic='{topic}', level={level}, item_id_start={item_id_start}, interaction_type={interaction_type}")
    
    logger.debug(f"Fetching vocabulary for topic '{topic}' at level {level}")
    vocab_list = fetch_vocab_from_vector_db(topic, level)
    logger.info(f"Retrieved {len(vocab_list)} vocabulary items for prompt generation")
    logger.debug(f"Vocabulary items: {vocab_list}")

    level_guidelines = f"""
Speaking Task Guidelines by Level (Based on Study Plans):

A1 LEVEL:
- Types: interview, question_answer, basic dialogue, picture description (simple).
- Focus: Can user answer simple questions? Introduce themselves? Use basic phrases? Ask for and give simple directions/times.
- Grammar: Present tense mostly, simple sentences (Hauptsatz), Nominative/Accusative/Dative. (Source A1 Plan)
- Expected duration: 15-30 seconds per sustained response.
- Task: Ask/answer simple personal questions, describe daily routine, ask for directions, state simple preferences.
- Evaluation: Clear pronunciation? Understands question? Uses basic grammar/vocabulary (e.g., correct case for simple objects)?

A2 LEVEL:
- Types: dialogue, interview, question_answer, picture description, roleplay.
- Focus: Can user have short conversations? Describe simple situations? Handle familiar topics (travel, past events) and give simple reasons.
- Grammar: Present + past tense (Perfekt, Präteritum), some complex sentences with basic subordinate clauses ('weil', 'dass'), Reflexive verbs, Passiv Präsens. (Source A2 Plan)
- Expected duration: 30-60 seconds per sustained response.
- Task: Describe a past trip/weekend, give reasons for a preference, describe a product, discuss banking/money matters.
- Evaluation: Fluent with minimal pauses? Uses variety of tenses? Can initiate and maintain short exchanges?

B1 LEVEL:
- Types: dialogue, roleplay, interview, picture_description, short presentation.
- Focus: Can user express opinions, explain reasons, deal with hypothetical situations (Irreales), and handle social/professional interactions (job interview, complaint).
- Grammar: Multiple tenses (Futur I, Passiv, Perfekt/Präteritum), complex sentence structures, subordinate clauses. Must attempt to use Konjunktiv II for hypothetical situations. (Source B1 Plan)
- Expected duration: 1-2 minutes per sustained response.
- Task: Present an opinion on a topic and defend it, describe and analyze a complex image, participate in a consensus-finding discussion, roleplay a complaint/request.
- Evaluation: Fluent with occasional pauses? Natural rhythm? Effective use of B1 grammar (e.g., Konjunktiv II, subordinate clauses)? Can structure a longer turn?

B2 LEVEL:
- Types: dialogue (complex), roleplay, interview (advanced), presentation (long), formal discussion.
- Focus: Can user argue, hypothesize, criticize, discuss abstract and specialized topics (science, politics, society), and speak fluidly for a sustained period?
- Grammar: Full range of tenses/moods (Konjunktiv I/II), sophisticated structures, extensive Passiv, correct use of Modalpartikels. Must be able to perform Direkte/Indirekte Rede transformations in speech. (Source B2 Plan)
- Expected duration: 2-3 minutes per sustained response.
- Task: Discuss advantages/disadvantages of a complex issue, propose a solution to a socio-political problem, analyze a curriculum vitae, present a complex topic using formal language.
- Evaluation: Very fluent with natural pauses? High accuracy in complex B2 structures? Nuanced, specialized vocabulary? Sustained, coherent discourse?
"""

    prompt = f"""
Task:
Generate EXACTLY 6 speaking tasks for CEFR {level}.
The output MUST be a SINGLE JSON ARRAY with 5 objects. No text before or after the JSON.


Inputs:
- vocab_list: {json.dumps(vocab_list, ensure_ascii=False)}
- topic: "{topic}"
- start_id: {item_id_start}
- interaction_type: "{interaction_type}" (can be: interview, dialogue, picture_description, roleplay, presentation, question_answer)


JSON ARRAY STRUCTURE (exact):
[
  {{
    "id": integer,
    "type": "{interaction_type}" | "dialogue" | "picture_description" | "roleplay" | "presentation" | "question_answer",
    "task_title": string,
    "task_description": string,
    "task_instructions": string,
    "prompt": string,
    "prompt_translation": string,
    "follow_up_questions": [string, string, string],
    "follow_up_translations": [string, string, string],
    "acceptable_response_length": "15-30 seconds" | "30-60 seconds" | "1-2 minutes" | "2-3 minutes",
    "vocabulary_required": [vocab from vocab_list],
    "grammar_structures_required": [string],
    "example_response": string,
    "example_response_translation": string,
    "evaluation_criteria": {{
        "fluency_pronunciation": string,
        "vocabulary_accuracy": string,
        "grammar_accuracy": string,
        "task_completion": string,
        "interaction_ability": string
    }},
    "metadata": {{
        "level": "{level}",
        "skill": "SPEAKING",
        "topic": "{topic}",
        "source": "generated",
        "timestamp": "{datetime.now().isoformat()}",
        "interaction_type": "{interaction_type}",
        "expected_duration_seconds": integer
    }}
  }},
  ...
]  <-- exactly 5 objects


CRITICAL REQUIREMENTS:
- Start IDs at {item_id_start} and increment sequentially.
- Task must incorporate 2-3 words from vocab_list in prompt.
- follow_up_questions: Generate 3 related follow-up questions to extend conversation.
- vocabulary_required: List specific vocab from vocab_list that student should use.
- grammar_structures_required: List expected grammar (e.g., "Present tense", "Modal verbs", "Past tense + present").
- example_response: Provide a 1-3 sentence model response at appropriate level.
- No explanations, no prose, no markdown — ONLY the JSON array.
- Perfectly follow the JSON array structure.
- The output MUST be valid JSON.
- The JSON MUST be a single array with EXACTLY 5 objects.
- The JSON MUST contain:
    * no trailing commas
    * no comments
    * all keys in double-quotes
    * all strings in double-quotes
- The output MUST be parsable by JSON.parse() without errors.

Content Rules:
- Use realistic everyday scenarios: introductions, shopping, travel, work, complaints, opinions, advice.
- Provide clear context for roleplay and dialogue tasks.
- Make follow-up questions progressively harder (Q1 simple, Q3 challenging).


{level_guidelines}


Evaluation_Criteria Guidelines:
- fluency_pronunciation: How smooth is speech? Clear pronunciation? Natural rhythm?
- vocabulary_accuracy: Does student use correct vocab? Appropriate to context?
- grammar_accuracy: Are verb tenses correct? Sentence structures sound natural?
- task_completion: Did student answer the question fully? Address all points?
- interaction_ability: Can student engage? Respond to follow-ups naturally?


Return ONLY the JSON array with 5 objects.
"""
    logger.debug(f"Generated prompt (length: {len(prompt)} characters)")
    logger.info("Speaking prompt generated successfully")
    return prompt




# Evaluation for writing and speaking
def evaluate_writing_response(writing_task, user_response):
    """Use LLM to evaluate if writing response is acceptable."""
    return f"""
You are a CEFR German exam grader at level {writing_task['metadata']['level']}.

TASK: {writing_task['task_description']}
INSTRUCTIONS: {writing_task['task_instructions']}
CONTENT POINTS REQUIRED: {', '.join(writing_task['content_points'])}
WORD COUNT RANGE: {writing_task['word_count_min']}-{writing_task['word_count_max']} words
EVALUATION CRITERIA: {json.dumps(writing_task['evaluation_criteria'], ensure_ascii=False)}

USER RESPONSE:
{user_response}

Evaluate the response on these criteria:
1. Task Completion: Are all 3 content points addressed?
2. Word Count: Is response within range?
3. Grammar: Are there major grammar errors?
4. Vocabulary: Is vocabulary appropriate and varied?
5. Organization: Is the text coherent and well-organized?

Respond in JSON:
{{
    "task_completed": true/false,
    "is_acceptable": true/false,
    "score_out_of_10": integer,
    "feedback": "constructive feedback",
    "errors": ["list of main errors"],
    "strengths": ["list of strengths"]
}}
"""


def evaluate_speaking_response(speaking_task, transcribed_audio_text):
    """Use LLM to evaluate if speaking response is acceptable."""
    return f"""
You are a CEFR German exam grader at level {speaking_task['metadata']['level']}.

TASK: {speaking_task['task_description']}
PROMPT: {speaking_task['prompt']}
EXPECTED DURATION: {speaking_task['acceptable_response_length']}
VOCABULARY REQUIRED: {', '.join(speaking_task['vocabulary_required'])}
GRAMMAR STRUCTURES: {', '.join(speaking_task['grammar_structures_required'])}
EVALUATION CRITERIA: {json.dumps(speaking_task['evaluation_criteria'], ensure_ascii=False)}

TRANSCRIBED AUDIO TEXT:
{transcribed_audio_text}

Evaluate the response on these criteria:
1. Task Completion: Did student answer the prompt? Address follow-up questions?
2. Fluency: Natural pace and rhythm? Minimal hesitations?
3. Pronunciation: Is pronunciation clear and mostly correct?
4. Grammar: Are verb tenses and sentence structures correct?
5. Vocabulary: Does student use required vocab? Appropriate level?

Respond in JSON:
{{
    "task_completed": true/false,
    "is_acceptable": true/false,
    "score_out_of_10": integer,
    "pronunciation_clarity": "clear" | "mostly_clear" | "needs_improvement",
    "fluency_level": "fluent" | "mostly_fluent" | "halting",
    "feedback": "constructive feedback",
    "errors": ["list of main errors"],
    "strengths": ["list of strengths"]
}}
"""