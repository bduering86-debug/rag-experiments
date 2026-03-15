# LLM-as-a-Judge Integration

## Übersicht

Das LLM-as-a-Judge System bewertet die Qualität von RAG-generierten Antworten automatisch anhand von 4 Kriterien:

- **F (Faithfulness)**: Treue zum bereitgestellten Kontext
- **R (Relevance)**: Relevanz der Antwort zum Problem
- **C (Completeness)**: Vollständigkeit der Antwort
- **L (Fluency)**: Sprachliche Qualität

## Bewertungsskala

| Score | Bedeutung      | Beschreibung                                                  |
|-------|----------------|---------------------------------------------------------------|
| 1     | Sehr schlecht  | Kriterium nicht erfüllt, gravierende Mängel                   |
| 2     | Schwach        | Teilweise erfüllt, deutliche Fehler oder Auslassungen        |
| 3     | Akzeptabel     | Grundsätzlich erfüllt, erkennbare Schwächen                   |
| 4     | Gut            | Weitgehend erfüllt, nur geringfügige Mängel                   |
| 5     | Sehr gut       | Vollständig erfüllt, keine relevanten Mängel                  |

## Score-Berechnung

```
AnswerQualityScore = 0.35 · F + 0.25 · R + 0.25 · C + 0.15 · L

Normalisierung auf [0, 1]:
AnswerQualityScore[0,1] = (AnswerQualityScore - 1) / 4
```

## Architektur

### 1. `llm_api.py` - Generischer LLM API Client
- Wiederverwendbarer Client für verschiedene LLM APIs
- Unterstützt OpenAI-kompatible APIs (OpenAI, Azure OpenAI, etc.)
- Einfach erweiterbar für andere Anbieter

### 2. `answer_quality.py` - Score-Berechnung
- Validiert Scores (1-5)
- Berechnet gewichteten Raw Score
- Normalisiert auf [0, 1]
- Liefert Interpretation

### 3. `llm_judge.py` - Judge Orchestrator
- Erstellt strukturierte Prompts
- Ruft externes LLM auf
- Parst Bewertungen aus Response
- Berechnet finalen Score

## Konfiguration

### .env Einstellungen

```env
# LLM-as-a-Judge aktivieren/deaktivieren
USE_LLM_JUDGE=false

# API Konfiguration
LLM_JUDGE_API_URL=https://api.openai.com/v1/chat/completions
LLM_JUDGE_API_KEY=sk-...  # Ihr API Key

# Modell-Einstellungen
LLM_JUDGE_MODEL=gpt-4o-mini
LLM_JUDGE_TEMPERATURE=0.1
LLM_JUDGE_MAX_TOKENS=1000
```

### Unterstützte APIs

#### OpenAI
```env
LLM_JUDGE_API_URL=https://api.openai.com/v1/chat/completions
LLM_JUDGE_API_KEY=sk-...
LLM_JUDGE_MODEL=gpt-4o-mini  # oder gpt-4, gpt-4-turbo, etc.
```

#### Azure OpenAI
```env
LLM_JUDGE_API_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment/chat/completions?api-version=2023-05-15
LLM_JUDGE_API_KEY=your-azure-key
LLM_JUDGE_MODEL=gpt-4
```

#### Andere OpenAI-kompatible APIs
Jede API, die das OpenAI Chat Completion Format verwendet, funktioniert.

## Verwendung

### In RAG Answer Orchestrator

Die Integration erfolgt automatisch in `evaluate_testcase()`:

```python
orchestrator = RAGAnswerOrchestrator(
    top_k=10,
    runs_per_testcase=1
)

result = orchestrator.evaluate_testcase(
    testcase=testcase,
    profile="low",
    model="llama3.1:8b-instruct-q4_K_M",
    run=1
)

# Ergebnis enthält Judge-Metriken (falls aktiviert)
print(f"Judge Score: {result['judge_normalized_score']:.4f}")
print(f"Faithfulness: {result['judge_faithfulness']}")
print(f"Relevance: {result['judge_relevance']}")
print(f"Completeness: {result['judge_completeness']}")
print(f"Fluency: {result['judge_fluency']}")
```

### Standalone Verwendung

```python
from rag_csv.utils.llm_judge import LLMJudge

judge = LLMJudge()

result = judge.evaluate(
    ticket_description="Problem-Beschreibung...",
    context="Bereitgestellter KB-Kontext...",
    generated_answer="Generierte Antwort..."
)

if result["success"]:
    print(f"Score: {result['normalized_score']:.4f}")
```

### Nur Score-Berechnung (ohne LLM)

```python
from rag_csv.utils.answer_quality import AnswerQualityCalculator, QualityScores

calculator = AnswerQualityCalculator()

scores = QualityScores(
    faithfulness=4.0,
    relevance=5.0,
    completeness=3.0,
    fluency=4.5
)

result = calculator.calculate(scores)
print(f"Normalized Score: {result['normalized_score']:.4f}")
```

## Testing

```bash
# Test ohne API Key (nur Calculator)
python tests/test_llm_judge.py

# Test mit API Key (vollständig)
# 1. Setze LLM_JUDGE_API_KEY in .env
# 2. Setze USE_LLM_JUDGE=true
python tests/test_llm_judge.py
```

## CSV Output

Die Evaluation-Ergebnisse werden in CSV mit folgenden zusätzlichen Spalten gespeichert:

- `judge_faithfulness`: F-Score (1-5)
- `judge_relevance`: R-Score (1-5)
- `judge_completeness`: C-Score (1-5)
- `judge_fluency`: L-Score (1-5)
- `judge_raw_score`: Gewichteter Score (1-5)
- `judge_normalized_score`: Normalisierter Score (0-1)
- `judge_success`: Erfolg/Fehler
- `judge_error`: Fehlermeldung (falls vorhanden)

## Kosten-Hinweise

- LLM Judge wird **pro Testcase** aufgerufen
- Mit 40 Testcases und 3 Runs = 120 API-Calls
- Geschätzte Tokens pro Call: ~2000 (Prompt) + ~200 (Response)
- **Kosten-Beispiel (gpt-4o-mini)**: 
  - Input: $0.15 / 1M tokens
  - Output: $0.60 / 1M tokens
  - 120 Calls ≈ 264K tokens → ~$0.15

## Best Practices

1. **Development**: Nutze `USE_LLM_JUDGE=false` für schnelle Tests
2. **Testing**: Teste mit wenigen Testcases zuerst
3. **Production**: Aktiviere Judge nur für finale Evaluationen
4. **Modell-Wahl**: `gpt-4o-mini` ist günstig und gut für Bewertungen
5. **Caching**: Überlege API-Response-Caching bei wiederholten Evaluationen

## Erweiterungen

### Eigene Kriterien

Passe `answer_quality.py` an:

```python
class AnswerQualityCalculator:
    WEIGHT_FAITHFULNESS = 0.30
    WEIGHT_RELEVANCE = 0.25
    WEIGHT_COMPLETENESS = 0.25
    WEIGHT_YOUR_METRIC = 0.20  # Neue Gewichtung
```

### Andere LLM Provider

Erweitere `llm_api.py` für spezifische APIs:

```python
class AnthropicClient(LLMAPIClient):
    def chat_completion(self, messages, **kwargs):
        # Anthropic-spezifische Implementierung
        pass
```

## Troubleshooting

### "LLM_JUDGE_API_KEY muss in .env gesetzt sein"
→ Füge API Key in `.env` hinzu

### "401 Unauthorized"
→ Prüfe ob API Key gültig ist

### "Parse-Fehler"
→ LLM hat nicht im erwarteten Format geantwortet
→ Prüfe `judge_response` für Details
→ Eventuell Temperature erhöhen oder Prompt anpassen

### Judge wird nicht aufgerufen
→ Prüfe `USE_LLM_JUDGE=true` in `.env`
→ Prüfe Logs für Initialisierungsfehler
